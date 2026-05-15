//! `lunaris-ai-proxy` daemon entry point.
//!
//! Wires the policy core (`ProxyService`) into a real outbound layer
//! (`ReqwestForwarder`) and exposes `org.lunaris.AIProxy1` on the
//! session D-Bus. Foundation §8.4.6 forbids any AI traffic from
//! leaving the host through any other path.

use std::sync::Arc;

use lunaris_ai_proxy::allowlist::Allowlist;
use lunaris_ai_proxy::audit::{AuditSink, TracingAuditSink};
use lunaris_ai_proxy::catalog::ProviderCatalog;
use lunaris_ai_proxy::forward::ReqwestForwarder;
use lunaris_ai_proxy::peer_auth::{self, PeerAuthError, PeerAuthMap};
use lunaris_ai_proxy::service::{
    CallerAllowlist, ForwardRequest, ProxyError, ProxyService,
};
use zbus::Connection;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let forwarder = Arc::new(ReqwestForwarder::new()?);
    let audit_sink: Arc<dyn AuditSink> = Arc::new(TracingAuditSink);
    let service = Arc::new(ProxyService::new(
        Allowlist::default_lunaris(),
        ProviderCatalog::default_lunaris(),
        CallerAllowlist::default_lunaris(),
        forwarder,
        audit_sink,
    ));

    let peer_map = Arc::new(PeerAuthMap::default_lunaris());
    let dbus = ProxyInterface {
        service: service.clone(),
        peer_map: peer_map.clone(),
    };

    let _connection = zbus::connection::Builder::session()?
        .name("org.lunaris.AIProxy1")?
        .serve_at("/org/lunaris/AIProxy1", dbus)?
        .build()
        .await?;

    tracing::info!("lunaris-ai-proxy: serving org.lunaris.AIProxy1");

    tokio::signal::ctrl_c().await?;
    tracing::info!("lunaris-ai-proxy: shutting down");
    Ok(())
}

/// D-Bus surface (`org.lunaris.AIProxy1`).
struct ProxyInterface {
    service: Arc<ProxyService>,
    peer_map: Arc<PeerAuthMap>,
}

#[zbus::interface(name = "org.lunaris.AIProxy1")]
impl ProxyInterface {
    /// Forward a completion request through the named provider's
    /// catalogued endpoint. The proxy uses its own provider catalog
    /// for endpoint lookup; the caller never supplies a URL.
    async fn forward_completion(
        &self,
        provider_name: &str,
        body_json: &str,
        audit_token: &str,
        #[zbus(header)] header: zbus::message::Header<'_>,
        #[zbus(connection)] connection: &Connection,
    ) -> zbus::fdo::Result<String> {
        let sender = header
            .sender()
            .ok_or_else(|| zbus::fdo::Error::AccessDenied("no sender".to_string()))?
            .to_string();
        let caller = peer_auth::resolve(&sender, connection, &self.peer_map)
            .await
            .map_err(map_peer_auth_error)?;
        let req = ForwardRequest {
            provider_name: provider_name.to_string(),
            body_json: body_json.to_string(),
            audit_token: audit_token.to_string(),
        };
        match self.service.forward(&caller, req).await {
            Ok(outcome) => Ok(serde_json::json!({
                "upstream_status": outcome.upstream_status,
                "body": outcome.body,
            })
            .to_string()),
            Err(err) => Err(map_error(err)),
        }
    }

    /// Return the catalogued provider names. Lists what callers may
    /// pass to `forward_completion`; it does *not* expose the
    /// underlying endpoint URLs.
    async fn list_allowed_providers(&self) -> Vec<String> {
        self.service.allowed_providers()
    }
}

fn map_peer_auth_error(err: PeerAuthError) -> zbus::fdo::Error {
    match err {
        PeerAuthError::NoSender => zbus::fdo::Error::AccessDenied("no sender".to_string()),
        PeerAuthError::PidLookup(detail) => zbus::fdo::Error::AccessDenied(
            format!("peer PID lookup failed: {detail}"),
        ),
        PeerAuthError::ExeLookup { pid, error } => zbus::fdo::Error::AccessDenied(format!(
            "peer exe lookup failed for pid {pid}: {error}"
        )),
        PeerAuthError::ExeNotAllowed { path } => zbus::fdo::Error::AccessDenied(format!(
            "caller executable not allowed: {path}"
        )),
        PeerAuthError::NameOwnershipMismatch {
            name,
            sender,
            owner,
        } => zbus::fdo::Error::AccessDenied(format!(
            "caller {sender} does not own {name} (owner: {owner})"
        )),
    }
}

fn map_error(err: ProxyError) -> zbus::fdo::Error {
    let detail = err.to_string();
    match err.code() {
        "caller-not-allowed" => zbus::fdo::Error::AccessDenied(detail),
        "unknown-provider" => zbus::fdo::Error::InvalidArgs(detail),
        "invalid-url" | "missing-host" => zbus::fdo::Error::Failed(detail),
        "disallowed-scheme" | "host-not-allowed" => zbus::fdo::Error::AccessDenied(detail),
        "proxy-at-capacity" => zbus::fdo::Error::LimitsExceeded(detail),
        "upstream-error" => zbus::fdo::Error::Failed(detail),
        _ => zbus::fdo::Error::Failed(detail),
    }
}
