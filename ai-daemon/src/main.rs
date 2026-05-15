//! `lunaris-ai-daemon` entry point.
//!
//! Wires the service core into a zbus session-bus connection and
//! exposes `org.lunaris.AI1`. Design:
//!
//! * All outbound LLM traffic transits the proxy via
//!   [`ProxiedProvider`] (Foundation §8.4.6). The daemon never
//!   speaks HTTP directly.
//! * Results are not broadcast on the session bus. Callers poll
//!   `take_result(query_id, retrieval_token)` and the daemon checks
//!   both the caller's unique bus name and the per-query retrieval
//!   token before handing back the result text.
//! * The `Enabled` property is read-only over D-Bus. Toggling AI
//!   on/off happens through Settings writing the canonical TOML
//!   config, which the daemon's config watcher picks up.

use std::sync::Arc;

use lunaris_ai_core::graph_query::{AccessTier, QueryScope};
use lunaris_ai_core::graph_schema::GraphSchema;
use lunaris_ai_core::pipeline::{CypherPipeline, GraphQuerier, QueryRunner};
use lunaris_ai_core::provider::AIProvider;
use lunaris_ai_daemon::config_watch;
use lunaris_ai_daemon::graph_adapter::OsSdkGraphQuerier;
use lunaris_ai_daemon::peer::{self, PeerError};
use lunaris_ai_daemon::registry::{AuthError, CompletionOutcome};
use lunaris_ai_daemon::service::{AiDaemonService, QueryError};
use lunaris_ai_providers::proxied::{ProxiedConfig, ProxiedProvider};
use zbus::Connection;

const BUS_NAME: &str = "org.lunaris.AI1";
const OBJECT_PATH: &str = "/org/lunaris/AI1";

/// Resolve the Knowledge Daemon query socket the same way every
/// other Lunaris client does: an explicit
/// `LUNARIS_DAEMON_SOCKET` override wins; otherwise the per-user
/// runtime path `$XDG_RUNTIME_DIR/lunaris/knowledge.sock` is used
/// when it exists (the daemon listens there in an unprivileged
/// session); the system path `/run/lunaris/knowledge.sock` is the
/// final fallback.
fn resolve_knowledge_socket() -> String {
    if let Ok(explicit) = std::env::var("LUNARIS_DAEMON_SOCKET") {
        if !explicit.is_empty() {
            return explicit;
        }
    }
    if let Ok(xdg) = std::env::var("XDG_RUNTIME_DIR") {
        if !xdg.is_empty() {
            let runtime = format!("{xdg}/lunaris/knowledge.sock");
            if std::path::Path::new(&runtime).exists() {
                return runtime;
            }
        }
    }
    "/run/lunaris/knowledge.sock".to_string()
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    // Read ai.toml once at startup. `provider` is applied here;
    // `enabled` is applied after the service is built and then kept
    // live by the config watcher (Phase 9-α S7).
    let settings = config_watch::load_ai_settings();
    tracing::info!(
        enabled = settings.enabled,
        provider = %settings.provider,
        "loaded ai.toml"
    );

    // One session-bus connection for the daemon's whole lifetime: it
    // owns `org.lunaris.AI1` and is also the connection the proxied
    // provider forwards on. The proxy authorises a forward by the
    // calling connection owning that name, so the provider and the
    // name must live on the same connection.
    let connection = zbus::Connection::session().await?;

    // Foundation §8.4.6: outbound LLM traffic goes through ai-proxy.
    let provider: Arc<dyn AIProvider> = Arc::new(
        ProxiedProvider::with_connection(
            ProxiedConfig {
                name: settings.provider.clone(),
                model: "llama3:8b".to_string(),
                audit_token: "ai-daemon-default-token".to_string(),
            },
            &connection,
        )
        .await?,
    );

    // Graph queries run against the Knowledge Daemon. The pipeline
    // turns NL into a validated structured query, compiles Cypher,
    // executes it here, then formats the result back to NL.
    let knowledge_socket = resolve_knowledge_socket();
    tracing::info!(socket = %knowledge_socket, "knowledge daemon socket");
    let graph: Arc<dyn GraphQuerier> =
        Arc::new(OsSdkGraphQuerier::new(knowledge_socket));
    let runner: Arc<dyn QueryRunner> =
        Arc::new(CypherPipeline::new(provider, graph));

    // Phase 9-α applies the Minimal scope (no graph access). The
    // user-selectable access tier needs the per-caller capability
    // model (S16) and the Settings tier slider (S24); until those
    // land, `Minimal` is the honest fixed scope. The S7 enable
    // toggle below makes the daemon switchable on/off, but the
    // graph-read scope stays Minimal.
    let scope = QueryScope::for_tier(
        AccessTier::Minimal,
        &GraphSchema::knowledge_graph(),
    );
    let service = Arc::new(AiDaemonService::new(runner, scope));

    // Apply ai.toml's `enabled` at startup, then keep it live: the
    // watcher re-applies it whenever Settings rewrites the file.
    service.set_enabled(settings.enabled);
    config_watch::spawn_config_watch(service.clone());

    // Auto-sweep terminal records once per minute. The handle is
    // kept alive for the daemon's lifetime; aborting it on shutdown
    // is fine because ctrl_c().await is the only exit path.
    let _sweep = service.spawn_sweep_task();

    let dbus = AiInterface {
        service: service.clone(),
    };

    // Register the interface, then claim the well-known name on the
    // same connection the provider forwards on. The interface is up
    // before the name is claimed so a client cannot reach the name
    // before the object is served.
    connection.object_server().at(OBJECT_PATH, dbus).await?;
    connection.request_name(BUS_NAME).await?;

    tracing::info!(bus = BUS_NAME, path = OBJECT_PATH, "lunaris-ai-daemon serving");

    tokio::signal::ctrl_c().await?;
    tracing::info!("lunaris-ai-daemon shutting down");
    Ok(())
}

/// D-Bus surface (`org.lunaris.AI1`).
struct AiInterface {
    service: Arc<AiDaemonService>,
}

#[zbus::interface(name = "org.lunaris.AI1")]
impl AiInterface {
    /// Submit a new query. Returns a `(query_id, retrieval_token)`
    /// pair as a JSON object. The caller must store both and present
    /// them on every follow-up method; the daemon also verifies the
    /// follow-up's D-Bus sender matches the submitter.
    async fn query(
        &self,
        prompt: &str,
        _context_hints: &str,
        #[zbus(header)] header: zbus::message::Header<'_>,
        #[zbus(connection)] connection: &Connection,
    ) -> zbus::fdo::Result<String> {
        // Resolve the caller's stable executable identity for
        // rate-limit accounting. Fails closed if the PID/exe cannot
        // be resolved.
        let caller = peer::resolve(&header, connection)
            .await
            .map_err(map_peer_error)?;
        match self.service.query(prompt.to_string(), caller).await {
            Ok(handle) => Ok(serde_json::json!({
                "query_id": handle.query_id,
                "retrieval_token": handle.retrieval_token,
            })
            .to_string()),
            Err(QueryError::Disabled) => Err(zbus::fdo::Error::AccessDenied(
                "ai layer is disabled".to_string(),
            )),
            Err(QueryError::TooManyInflight) => Err(zbus::fdo::Error::LimitsExceeded(
                "too many in-flight queries for this caller".to_string(),
            )),
            Err(QueryError::GlobalCapacityReached) => Err(zbus::fdo::Error::LimitsExceeded(
                "daemon at global query capacity".to_string(),
            )),
            Err(QueryError::PromptTooLarge(n)) => Err(zbus::fdo::Error::LimitsExceeded(
                format!("prompt too large: {n} bytes"),
            )),
            Err(QueryError::NoGraphAccess) => Err(zbus::fdo::Error::NotSupported(
                "ai layer has no graph access configured".to_string(),
            )),
        }
    }

    /// Poll a query for completion. Returns a JSON envelope of the
    /// form `{ "status": "...", ... }`. Result text is only included
    /// for the single-shot `completed` status; subsequent polls
    /// return `drained`.
    async fn take_result(
        &self,
        query_id: &str,
        retrieval_token: &str,
        #[zbus(header)] header: zbus::message::Header<'_>,
    ) -> zbus::fdo::Result<String> {
        let caller = sender(&header)?;
        let outcome = self
            .service
            .take_result(query_id, &caller, retrieval_token)
            .await
            .map_err(map_auth_error)?;
        Ok(serialise_outcome(outcome))
    }

    /// Cancel an in-flight query. Returns true if the query existed,
    /// was not already terminated, and the caller passed authz.
    async fn cancel(
        &self,
        query_id: &str,
        retrieval_token: &str,
        #[zbus(header)] header: zbus::message::Header<'_>,
    ) -> zbus::fdo::Result<bool> {
        let caller = sender(&header)?;
        self.service
            .cancel(query_id, &caller, retrieval_token)
            .await
            .map_err(map_auth_error)
    }

    /// Whether the daemon is currently accepting new queries. Read
    /// only; writers must update the canonical TOML config and the
    /// daemon picks it up through its config watcher (S7).
    #[zbus(property)]
    fn enabled(&self) -> bool {
        self.service.is_enabled()
    }
}

fn sender(header: &zbus::message::Header<'_>) -> zbus::fdo::Result<String> {
    header
        .sender()
        .map(|s| s.to_string())
        .ok_or_else(|| zbus::fdo::Error::AccessDenied("message has no sender".to_string()))
}

fn map_peer_error(err: PeerError) -> zbus::fdo::Error {
    match err {
        PeerError::NoSender => {
            zbus::fdo::Error::AccessDenied("message has no sender".to_string())
        }
        PeerError::PidLookup(detail) => {
            zbus::fdo::Error::AccessDenied(format!("caller PID lookup failed: {detail}"))
        }
        PeerError::ExeLookup { pid, error } => zbus::fdo::Error::AccessDenied(format!(
            "caller exe lookup failed for pid {pid}: {error}"
        )),
    }
}

fn map_auth_error(err: AuthError) -> zbus::fdo::Error {
    match err {
        AuthError::UnknownQuery => zbus::fdo::Error::InvalidArgs("unknown query".to_string()),
        AuthError::CallerMismatch => {
            zbus::fdo::Error::AccessDenied("caller does not match submitter".to_string())
        }
        AuthError::TokenMismatch => {
            zbus::fdo::Error::AccessDenied("retrieval token mismatch".to_string())
        }
    }
}

fn serialise_outcome(outcome: CompletionOutcome) -> String {
    let value = match outcome {
        CompletionOutcome::Pending => serde_json::json!({ "status": "pending" }),
        CompletionOutcome::InProgress => serde_json::json!({ "status": "in-progress" }),
        CompletionOutcome::Completed { result } => {
            serde_json::json!({ "status": "completed", "result": result })
        }
        CompletionOutcome::Drained => serde_json::json!({ "status": "drained" }),
        CompletionOutcome::Failed { code, reason } => serde_json::json!({
            "status": "failed",
            "code": code,
            "reason": reason,
        }),
        CompletionOutcome::Cancelled => serde_json::json!({ "status": "cancelled" }),
    };
    value.to_string()
}
