//! MCP server discovery for the AI daemon.
//!
//! Tier-1 `mcp.server` modules are hosted by `lunaris-modulesd`,
//! which fronts each with a Unix socket under
//! `$XDG_RUNTIME_DIR/lunaris/mcp/modules/` and announces it on the
//! Event Bus. This module keeps the daemon's [`McpClient`] in step
//! with that feed.
//!
//! Trust model: the Event Bus does not authenticate event *content*
//! (any same-uid producer can emit `module.installed`), and the
//! modules directory is writable by any same-uid process. Discovery
//! therefore trusts neither surface blindly. Before registering a
//! server it (1) rejects module ids that could escape the socket
//! directory and (2) resolves the socket's server peer via
//! `SO_PEERCRED` and requires it to be `modulesd`. A forged event or
//! a stray socket cannot make the daemon adopt an imposter server.
//!
//! Per foundation §5.7 a third-party module MCP server is always
//! treated as an action server until it carries a Security Audit
//! Badge, so every module is registered with [`ServerClass::Action`].

use std::sync::Arc;
use std::time::Duration;

use lunaris_ai_core::mcp::{McpClient, ServerClass, ServerId};
use lunaris_permissions::identity::app_id_from_pid;
use os_sdk::event_consumer::{EventConsumer, UnixEventConsumer};
use os_sdk::mcp::{is_safe_module_id, mcp_module_socket_path};
use tokio::net::UnixStream;
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

/// Event-type prefix the discovery loop subscribes to. Prefix-match
/// semantics: this catches `module.installed`, `module.removed`, and
/// any future `module.*` event.
const MODULE_NAMESPACE: &str = "module.";

/// Resolved `app_id` of the canonically-installed module runtime
/// daemon. `lunaris-permissions` maps `/usr/bin/lunaris-modulesd` to
/// this; a module MCP socket served by anything else is an imposter.
const MODULESD_APP_ID: &str = "modulesd";

/// Wait budget for opening and handshaking a module socket. A bad
/// socket that accepts but stalls cannot wedge discovery.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);

/// Backoff between Event Bus subscribe attempts. The bus may come up
/// after the daemon during a normal boot; discovery keeps retrying
/// rather than disabling itself permanently.
const SUBSCRIBE_RETRY: Duration = Duration::from_secs(5);

/// Keeps the daemon's [`McpClient`] connected to the set of
/// currently-hosted Tier-1 module MCP servers.
pub struct McpDiscovery {
    client: Arc<Mutex<McpClient>>,
}

impl McpDiscovery {
    /// Build a discovery handle around a fresh, empty client.
    pub fn new() -> Self {
        Self {
            client: Arc::new(Mutex::new(McpClient::new())),
        }
    }

    /// The shared MCP client. The query path dispatches tool calls
    /// through this once AI-side tool routing lands; until then the
    /// discovery loop is its only writer.
    pub fn client(&self) -> Arc<Mutex<McpClient>> {
        Arc::clone(&self.client)
    }

    /// Subscribe to the `module.` Event Bus namespace and keep the
    /// client in step. Subscription is retried until it succeeds, and
    /// re-established if the feed later closes; an existing-socket
    /// reconciliation runs on every (re)subscribe. Runs forever.
    pub async fn run(self: Arc<Self>, consumer: UnixEventConsumer) {
        loop {
            let mut rx = match consumer
                .subscribe(vec![MODULE_NAMESPACE.to_string()])
                .await
            {
                Ok(rx) => rx,
                Err(err) => {
                    warn!(
                        "mcp discovery: event bus subscribe failed: {err}; \
                         retrying in {}s",
                        SUBSCRIBE_RETRY.as_secs()
                    );
                    tokio::time::sleep(SUBSCRIBE_RETRY).await;
                    continue;
                }
            };
            info!("mcp discovery: subscribed to the module.* event namespace");
            // Reconcile *after* subscribing: an install or remove that
            // lands during the scan queues in `rx` and is drained by
            // the loop below, so the startup gap drops no event.
            self.scan_existing().await;
            while let Some(event) = rx.recv().await {
                self.handle_event(&event.r#type, &event.payload).await;
            }
            warn!("mcp discovery: event feed closed; re-subscribing");
        }
    }

    /// Dispatch one `module.*` event.
    async fn handle_event(&self, event_type: &str, payload: &[u8]) {
        // modulesd carries the module id as the raw UTF-8 payload.
        let module_id = match std::str::from_utf8(payload) {
            Ok(id) if !id.is_empty() => id,
            _ => {
                warn!(
                    event_type,
                    "mcp discovery: module event with no valid module id"
                );
                return;
            }
        };
        match event_type {
            "module.installed" => self.connect_module(module_id).await,
            "module.removed" => self.disconnect_module(module_id).await,
            other => debug!(
                event_type = other,
                "mcp discovery: non-discovery module event ignored"
            ),
        }
    }

    /// Connect to every module MCP socket already present on disk.
    /// Covers modules hosted before the daemon started, and re-runs
    /// on every resubscribe as a reconciliation pass.
    async fn scan_existing(&self) {
        let Some(dir) = mcp_module_socket_path("placeholder")
            .parent()
            .map(|p| p.to_path_buf())
        else {
            return;
        };
        let entries = match std::fs::read_dir(&dir) {
            Ok(entries) => entries,
            // A missing directory just means no module has been
            // hosted yet; that is not an error.
            Err(_) => return,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("sock") {
                continue;
            }
            if let Some(module_id) = path.file_stem().and_then(|s| s.to_str()) {
                self.connect_module(module_id).await;
            }
        }
    }

    /// Connect the client to one module's MCP socket, after checking
    /// the id is path-safe and the socket is actually served by
    /// modulesd. Any failure is logged, not fatal.
    async fn connect_module(&self, module_id: &str) {
        // (1) An id that fails this check could format a socket path
        // outside the modules directory. modulesd applies the same
        // gate before binding; discovery does not trust an id it has
        // not validated itself.
        if !is_safe_module_id(module_id) {
            warn!(module = module_id, "mcp discovery: unsafe module id, ignored");
            return;
        }
        let path = mcp_module_socket_path(module_id);

        // Open the stream directly so the server peer can be checked
        // before the rmcp handshake runs.
        let stream = match tokio::time::timeout(
            CONNECT_TIMEOUT,
            UnixStream::connect(&path),
        )
        .await
        {
            Ok(Ok(stream)) => stream,
            Ok(Err(err)) => {
                warn!(
                    module = module_id,
                    "mcp discovery: module server unavailable: {err}"
                );
                return;
            }
            Err(_elapsed) => {
                warn!(module = module_id, "mcp discovery: connect timed out");
                return;
            }
        };

        // (2) Authenticate the server end. A stray socket planted by
        // another same-uid process is served by that process, not
        // modulesd, and is refused here.
        if !peer_is_modulesd(&stream) {
            warn!(
                module = module_id,
                "mcp discovery: socket is not served by modulesd; refusing"
            );
            return;
        }

        let mut client = self.client.lock().await;
        match tokio::time::timeout(
            CONNECT_TIMEOUT,
            client.connect_stream(
                ServerId(module_id.to_string()),
                stream,
                ServerClass::Action,
            ),
        )
        .await
        {
            Ok(Ok(())) => info!(module = module_id, "mcp discovery: connected"),
            Ok(Err(err)) => warn!(
                module = module_id,
                "mcp discovery: handshake failed: {err}"
            ),
            Err(_elapsed) => warn!(
                module = module_id,
                "mcp discovery: handshake timed out"
            ),
        }
    }

    /// Drop the client's connection to one module.
    async fn disconnect_module(&self, module_id: &str) {
        self.client
            .lock()
            .await
            .disconnect(&ServerId(module_id.to_string()));
        info!(module = module_id, "mcp discovery: disconnected");
    }
}

impl Default for McpDiscovery {
    fn default() -> Self {
        Self::new()
    }
}

/// Whether the peer that bound `stream`'s server end is `modulesd`.
///
/// Uses `SO_PEERCRED` on the live connection — the credentials of the
/// process actually `accept()`ing — so it cannot be spoofed by a
/// path swap. In debug builds every component runs from a cargo
/// target directory and resolves to a `dev.*` id, so those pass too.
fn peer_is_modulesd(stream: &UnixStream) -> bool {
    let Ok(cred) = stream.peer_cred() else {
        return false;
    };
    let Some(pid) = cred.pid() else {
        return false;
    };
    if pid < 0 {
        return false;
    }
    let Ok(app_id) = app_id_from_pid(pid as u32) else {
        return false;
    };
    app_id == MODULESD_APP_ID
        || (cfg!(debug_assertions) && app_id.starts_with("dev."))
}
