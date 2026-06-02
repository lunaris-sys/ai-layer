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

use lunaris_ai_core::audit::{AuditSink, LedgerAuditSink};
use lunaris_ai_core::capability::access_tier_from_level;
use lunaris_ai_core::graph_query::QueryScope;
use lunaris_ai_core::graph_schema::GraphSchema;
use lunaris_ai_core::pipeline::{CypherPipeline, GraphQuerier, QueryRunner};
use lunaris_ai_core::provider::AIProvider;
use lunaris_ai_daemon::authz::AuthorizationStore;
use lunaris_ai_daemon::config_watch;
use lunaris_ai_daemon::graph_adapter::OsSdkGraphQuerier;
use lunaris_ai_daemon::mcp_discovery::McpDiscovery;
use lunaris_ai_daemon::peer::{self, PeerError};
use lunaris_ai_daemon::registry::{AuthError, CompletionOutcome};
use lunaris_ai_daemon::service::{AiDaemonService, QueryError};
use lunaris_ai_providers::proxied::{ProxiedConfig, ProxiedProvider};
use os_sdk::UnixEventConsumer;
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

/// Resolve the Event Bus consumer socket the same way: an explicit
/// `LUNARIS_CONSUMER_SOCKET` override wins; otherwise the per-user
/// runtime path is used when it exists; the system path is the
/// final fallback.
fn resolve_event_consumer_socket() -> String {
    if let Ok(explicit) = std::env::var("LUNARIS_CONSUMER_SOCKET") {
        if !explicit.is_empty() {
            return explicit;
        }
    }
    if let Ok(xdg) = std::env::var("XDG_RUNTIME_DIR") {
        if !xdg.is_empty() {
            let runtime = format!("{xdg}/lunaris/event-bus-consumer.sock");
            if std::path::Path::new(&runtime).exists() {
                return runtime;
            }
        }
    }
    "/run/lunaris/event-bus-consumer.sock".to_string()
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

    // The audit sink submits to `lunaris-auditd` over its ingest
    // socket. It is shared: the service gates every query on a
    // dispatch entry, and the MCP discovery layer audits tool calls
    // through the same ledger.
    let audit: Arc<dyn AuditSink> = Arc::new(LedgerAuditSink::at_default_socket());

    // The service is constructed fail-closed: disabled, with the Minimal
    // (no graph access) scope. The config watcher is the sole owner of
    // the admission state (enabled flag + read scope from `ai.toml`'s
    // `access_level`, 0..=4, Foundation §8.4); it publishes the
    // configured admission once its file watch is armed and keeps it live
    // on every change. Starting fail-closed means there is no window in
    // which a stale startup snapshot serves access before the watcher is
    // live. The Settings tier slider that writes `access_level` is S24.
    let service = Arc::new(AiDaemonService::new(
        runner,
        QueryScope::for_tier(access_tier_from_level(0), &GraphSchema::knowledge_graph()),
        audit.clone(),
    ));
    config_watch::spawn_config_watch(service.clone());

    // Auto-sweep terminal records once per minute. The handle is
    // kept alive for the daemon's lifetime; aborting it on shutdown
    // is fine because ctrl_c().await is the only exit path.
    let _sweep = service.spawn_sweep_task();

    // Per-session authorization for MCP action servers. Grants live
    // here only; nothing is persisted, and the store is dropped with
    // the process at session end.
    let authz = Arc::new(AuthorizationStore::new());

    let dbus = AiInterface {
        service: service.clone(),
        authz: authz.clone(),
    };

    // Register the interface, then claim the well-known name on the
    // same connection the provider forwards on. The interface is up
    // before the name is claimed so a client cannot reach the name
    // before the object is served.
    connection.object_server().at(OBJECT_PATH, dbus).await?;
    connection.request_name(BUS_NAME).await?;

    tracing::info!(bus = BUS_NAME, path = OBJECT_PATH, "lunaris-ai-daemon serving");

    // Discover Tier-1 module MCP servers over the Event Bus `module.`
    // namespace. `run` subscribes (retrying if the bus is late),
    // reconciles against the sockets already on disk, and tracks
    // installs and removals for the rest of the session.
    let discovery = Arc::new(McpDiscovery::new(audit.clone()));
    let consumer = UnixEventConsumer::new(resolve_event_consumer_socket());
    tokio::spawn(discovery.run(consumer));

    tokio::signal::ctrl_c().await?;
    tracing::info!("lunaris-ai-daemon shutting down");
    Ok(())
}

/// D-Bus surface (`org.lunaris.AI1`).
struct AiInterface {
    service: Arc<AiDaemonService>,
    authz: Arc<AuthorizationStore>,
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
            Err(QueryError::AuditUnavailable) => Err(zbus::fdo::Error::Failed(
                "audit log unavailable; query refused".to_string(),
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

    /// Answer an open authorization prompt.
    ///
    /// Only the desktop shell may call this: the `AuthorizationPrompt`
    /// signal that carries a prompt id is a session-bus broadcast, so
    /// without a caller check any peer that observed the id could
    /// approve a scope itself. The caller's executable is resolved
    /// and checked against the trusted shell binary before the
    /// decision is recorded.
    ///
    /// Returns `true` if a matching pending prompt existed.
    async fn respond_authorization(
        &self,
        prompt_id: &str,
        granted: bool,
        #[zbus(header)] header: zbus::message::Header<'_>,
        #[zbus(connection)] connection: &Connection,
    ) -> zbus::fdo::Result<bool> {
        let caller = peer::resolve(&header, connection)
            .await
            .map_err(map_peer_error)?;
        if !is_trusted_shell(&caller.stable_id) {
            return Err(zbus::fdo::Error::AccessDenied(
                "only the desktop shell may answer authorization prompts"
                    .to_string(),
            ));
        }
        match uuid::Uuid::parse_str(prompt_id) {
            Ok(id) => Ok(self.authz.resolve(id, granted).await),
            Err(_) => Ok(false),
        }
    }

    /// Emitted when a scope needs the user's authorization. The
    /// payload is only a prompt id and a scope label, never query
    /// content. The prompt id is not a bearer token: a response is
    /// authorised by the caller's identity in `respond_authorization`,
    /// not by knowing the id. Phase 9-δ's tool dispatch emits this
    /// once it can request authorization for a real tool call.
    #[zbus(signal)]
    async fn authorization_prompt(
        emitter: &zbus::object_server::SignalEmitter<'_>,
        prompt_id: &str,
        scope: &str,
    ) -> zbus::Result<()>;
}

/// Canonical install paths of the desktop shell binary. Only a
/// process running one of these may answer authorization prompts.
const TRUSTED_SHELL_BINS: &[&str] = &[
    "/usr/bin/lunaris-desktop-shell",
    "/usr/lib/lunaris/libexec/lunaris-desktop-shell",
];

/// Whether `exe_path` is the trusted desktop shell.
///
/// In debug builds a `LUNARIS_AI_TRUSTED_SHELL_BIN` env var adds a
/// dev path (the repo-relative `cargo tauri dev` binary). The
/// override is compiled out of release builds so it cannot become
/// part of the production trust boundary.
fn is_trusted_shell(exe_path: &str) -> bool {
    if TRUSTED_SHELL_BINS.contains(&exe_path) {
        return true;
    }
    #[cfg(debug_assertions)]
    if let Ok(dev) = std::env::var("LUNARIS_AI_TRUSTED_SHELL_BIN") {
        if !dev.is_empty() && dev == exe_path {
            return true;
        }
    }
    false
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
