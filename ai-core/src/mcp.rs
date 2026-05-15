//! MCP client for the AI layer.
//!
//! Lunaris exposes application capabilities as MCP tools. This module
//! is the consuming side: a single [`McpClient`], embedded in the AI
//! daemon, holds one connection per MCP server and dispatches
//! `tools/list` and `tools/call` over it.
//!
//! The protocol itself is `rmcp`, the upstream MCP SDK. Local servers
//! are reached over a Unix socket; `rmcp` carries JSON-RPC over the
//! stream and Lunaris does not extend the wire format.
//!
//! On top of `rmcp` this module adds the two Lunaris-specific guards
//! from the foundation:
//!
//! * **Call-chain depth.** Every call carries a [`CallChain`]; a call
//!   past the depth limit is refused before dispatch so a server
//!   that calls another server cannot recurse without bound.
//! * **Always-confirm classification.** A hardcoded set of
//!   high-impact action shapes ([`AlwaysConfirm`]) is checked before
//!   a call is issued, so an irreversible action always reaches a
//!   confirmation prompt regardless of any standing authorization.

use std::collections::HashMap;

use rmcp::model::CallToolRequestParams;
use rmcp::service::{RoleClient, RunningService};
use rmcp::ServiceExt;
use tokio::net::UnixStream;
use uuid::Uuid;

/// Default maximum MCP call-chain depth. Legitimate nested calls
/// rarely exceed two or three levels; five is a conservative cap.
pub const DEFAULT_MAX_DEPTH: u8 = 5;

/// Hard ceiling on the call-chain depth. No configuration may raise
/// the limit above this, however it is set.
pub const MAX_DEPTH_CEILING: u8 = 10;

/// Identifier of a connected MCP server.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ServerId(pub String);

impl ServerId {
    /// Borrow the identifier string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// A tool exposed by an MCP server, as seen by the client.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolDef {
    /// Tool name, used as the `tools/call` target.
    pub name: String,
    /// Human-readable description, if the server provides one.
    pub description: Option<String>,
}

/// Tracks one MCP call chain.
///
/// A chain starts at depth 1 with [`CallChain::root`]. When an MCP
/// call triggers a nested MCP call, the nested call uses
/// [`CallChain::nested`], which keeps the chain id and increments the
/// depth. The depth bound is enforced by [`McpClient::call_tool`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallChain {
    /// Identifier shared by every call in the chain.
    pub id: Uuid,
    /// Current depth, starting at 1.
    pub depth: u8,
}

impl CallChain {
    /// Start a fresh chain at depth 1.
    pub fn root() -> Self {
        Self {
            id: Uuid::new_v4(),
            depth: 1,
        }
    }

    /// Derive the chain for a call nested inside this one.
    pub fn nested(&self) -> Self {
        Self {
            id: self.id,
            depth: self.depth.saturating_add(1),
        }
    }
}

/// Errors raised by the MCP client.
#[derive(Debug, thiserror::Error)]
pub enum McpError {
    /// The server's socket could not be reached or the handshake
    /// failed.
    #[error("mcp server '{0}' unavailable")]
    ServerUnavailable(String),
    /// No connection is registered for the given server id.
    #[error("unknown mcp server '{0}'")]
    UnknownServer(String),
    /// The call-chain depth limit was reached. Mirrors the MCP
    /// JSON-RPC error `-32099` (`mcp_depth_exceeded`).
    #[error("mcp call chain depth {depth} exceeds maximum {max}")]
    DepthExceeded {
        /// Depth the rejected call would have run at.
        depth: u8,
        /// Configured maximum.
        max: u8,
    },
    /// The server returned an error or the transport failed.
    #[error("mcp call failed: {0}")]
    CallFailed(String),
    /// The server reported the tool call itself as an error.
    #[error("mcp tool reported an error: {0}")]
    ToolError(String),
}

/// Permission class of an MCP server.
///
/// Read-only servers are usable within the enabled AI layer without
/// a prompt (foundation Table 7). Action servers mutate state and
/// require a live per-session authorization grant before a call
/// (Table 8). Third-party module servers are always registered as
/// `Action` until they carry a verified Security Audit Badge,
/// regardless of what their manifest claims.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServerClass {
    /// Reading does not change state.
    ReadOnly,
    /// Calls can mutate state; per-session authorization required.
    Action,
}

/// Outcome of the pre-call permission check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CallDecision {
    /// The call may be dispatched now.
    Allow,
    /// The server is an action server and no authorization grant is
    /// held; the user must authorize the scope first.
    NeedsAuthorization,
    /// The tool is in the hardcoded always-confirm set; an explicit
    /// confirmation is required even when an authorization grant is
    /// held.
    NeedsConfirmation(AlwaysConfirmReason),
}

/// One live connection to an MCP server.
pub struct McpConnection {
    service: RunningService<RoleClient, ()>,
    class: ServerClass,
}

/// One audit record for an MCP call. The full audit ledger with
/// hash-chain integrity is Phase 9-γ; until then records are emitted
/// as structured `tracing` events, the same policy-only stub the
/// rest of the AI layer's auditing uses.
#[derive(Debug, Clone)]
pub struct McpAuditRecord {
    /// Identifier shared by every call in the chain.
    pub call_chain_id: Uuid,
    /// Depth this call ran at.
    pub depth: u8,
    /// Server the call targeted.
    pub server: String,
    /// Tool name.
    pub tool: String,
    /// Coarse outcome label (`ok`, `tool-error`, `failed`,
    /// `depth-exceeded`, ...).
    pub outcome: &'static str,
}

impl McpAuditRecord {
    /// Emit the record. Tool arguments are deliberately not
    /// included (PII risk per foundation §8.4.7).
    fn emit(&self) {
        tracing::info!(
            call_chain_id = %self.call_chain_id,
            depth = self.depth,
            server = %self.server,
            tool = %self.tool,
            outcome = self.outcome,
            "mcp call audited"
        );
    }
}

/// The AI layer's MCP client.
///
/// Holds one [`McpConnection`] per server. Cheap to construct;
/// connections are added with [`connect`](McpClient::connect) and
/// dropped with [`disconnect`](McpClient::disconnect).
pub struct McpClient {
    servers: HashMap<ServerId, McpConnection>,
    max_depth: u8,
}

impl Default for McpClient {
    fn default() -> Self {
        Self::new()
    }
}

impl McpClient {
    /// Build a client with the default depth limit.
    pub fn new() -> Self {
        Self {
            servers: HashMap::new(),
            max_depth: DEFAULT_MAX_DEPTH,
        }
    }

    /// Build a client with an explicit depth limit. The limit is
    /// clamped to `[1, MAX_DEPTH_CEILING]`.
    pub fn with_max_depth(max_depth: u8) -> Self {
        Self {
            servers: HashMap::new(),
            max_depth: max_depth.clamp(1, MAX_DEPTH_CEILING),
        }
    }

    /// The effective call-chain depth limit.
    pub fn max_depth(&self) -> u8 {
        self.max_depth
    }

    /// Number of connected servers.
    pub fn len(&self) -> usize {
        self.servers.len()
    }

    /// Whether any server is connected.
    pub fn is_empty(&self) -> bool {
        self.servers.is_empty()
    }

    /// Connect to an MCP server over its Unix socket and run the MCP
    /// handshake. Replaces any existing connection for `id`.
    ///
    /// `class` declares whether the server is read-only or an action
    /// server. The discovery layer passes `ServerClass::Action` for
    /// every third-party module server regardless of its manifest.
    pub async fn connect(
        &mut self,
        id: ServerId,
        socket_path: &str,
        class: ServerClass,
    ) -> Result<(), McpError> {
        let stream = UnixStream::connect(socket_path).await.map_err(|err| {
            McpError::ServerUnavailable(format!("{}: {err}", id.0))
        })?;
        self.connect_stream(id, stream, class).await
    }

    /// Connect over an already-open stream. Used by tests and by the
    /// in-process module hosting path.
    pub async fn connect_stream(
        &mut self,
        id: ServerId,
        stream: UnixStream,
        class: ServerClass,
    ) -> Result<(), McpError> {
        let service = ().serve(stream).await.map_err(|err| {
            McpError::ServerUnavailable(format!("{}: {err}", id.0))
        })?;
        self.servers
            .insert(id, McpConnection { service, class });
        Ok(())
    }

    /// The permission class a connected server was registered with.
    pub fn server_class(&self, id: &ServerId) -> Option<ServerClass> {
        self.servers.get(id).map(|c| c.class)
    }

    /// Decide whether a tool call may proceed.
    ///
    /// `has_grant` is whether the caller holds a live per-session
    /// authorization grant covering this server. The check order is
    /// deliberate: the always-confirm classifier wins over a held
    /// grant, so an irreversible action always reaches a
    /// confirmation prompt.
    pub fn decide(
        &self,
        id: &ServerId,
        tool: &str,
        has_grant: bool,
    ) -> Result<CallDecision, McpError> {
        let conn = self
            .servers
            .get(id)
            .ok_or_else(|| McpError::UnknownServer(id.0.clone()))?;

        if let Some(reason) = AlwaysConfirm::classify(tool) {
            return Ok(CallDecision::NeedsConfirmation(reason));
        }
        match conn.class {
            ServerClass::ReadOnly => Ok(CallDecision::Allow),
            ServerClass::Action => {
                if has_grant {
                    Ok(CallDecision::Allow)
                } else {
                    Ok(CallDecision::NeedsAuthorization)
                }
            }
        }
    }

    /// Drop the connection to a server. Tools from it stop being
    /// callable; the rmcp service is cancelled on drop.
    pub fn disconnect(&mut self, id: &ServerId) {
        self.servers.remove(id);
    }

    /// List the tools a connected server exposes.
    pub async fn list_tools(&self, id: &ServerId) -> Result<Vec<ToolDef>, McpError> {
        let conn = self
            .servers
            .get(id)
            .ok_or_else(|| McpError::UnknownServer(id.0.clone()))?;
        let tools = conn
            .service
            .list_all_tools()
            .await
            .map_err(|err| McpError::CallFailed(err.to_string()))?;
        Ok(tools
            .into_iter()
            .map(|t| ToolDef {
                name: t.name.to_string(),
                description: t.description.map(|d| d.to_string()),
            })
            .collect())
    }

    /// Call a tool on a connected server.
    ///
    /// `chain` carries the call-chain id and depth; a call past the
    /// configured depth limit is refused before any dispatch. The
    /// tool result is flattened to its text content.
    pub async fn call_tool(
        &self,
        id: &ServerId,
        tool: &str,
        arguments: serde_json::Value,
        chain: &CallChain,
    ) -> Result<String, McpError> {
        // Helper to emit one audit record for this call.
        let audit = |outcome: &'static str| {
            McpAuditRecord {
                call_chain_id: chain.id,
                depth: chain.depth,
                server: id.0.clone(),
                tool: tool.to_string(),
                outcome,
            }
            .emit();
        };

        if chain.depth > self.max_depth {
            audit("depth-exceeded");
            return Err(McpError::DepthExceeded {
                depth: chain.depth,
                max: self.max_depth,
            });
        }
        let conn = match self.servers.get(id) {
            Some(conn) => conn,
            None => {
                audit("unknown-server");
                return Err(McpError::UnknownServer(id.0.clone()));
            }
        };

        let arguments = match arguments {
            serde_json::Value::Object(map) => Some(map),
            serde_json::Value::Null => None,
            other => {
                // A non-object, non-null argument is not a valid MCP
                // argument set; surface it rather than silently
                // dropping the call.
                audit("bad-arguments");
                return Err(McpError::CallFailed(format!(
                    "tool arguments must be a JSON object, got {other}"
                )));
            }
        };

        let mut params = CallToolRequestParams::new(tool.to_string());
        if let Some(map) = arguments {
            params = params.with_arguments(map);
        }
        let result = match conn.service.call_tool(params).await {
            Ok(result) => result,
            Err(err) => {
                audit("failed");
                return Err(McpError::CallFailed(err.to_string()));
            }
        };

        let text = flatten_content(&result.content);
        if result.is_error.unwrap_or(false) {
            audit("tool-error");
            return Err(McpError::ToolError(text));
        }
        audit("ok");
        Ok(text)
    }
}

/// Flatten an MCP content list to a plain string. Text parts are
/// concatenated; non-text parts are noted by kind so the caller
/// still sees that something was returned.
fn flatten_content(content: &[rmcp::model::Content]) -> String {
    let mut out = String::new();
    for part in content {
        if let Some(text) = part.as_text() {
            out.push_str(&text.text);
        } else {
            out.push_str("[non-text content]");
        }
    }
    out
}

/// Category of a high-impact action that always requires explicit
/// user confirmation, regardless of any standing session
/// authorization. The set is fixed by the foundation and is not
/// user-configurable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlwaysConfirmReason {
    /// Permanent deletion of a file.
    FileDeletion,
    /// Sending a message outside the machine (email, messenger).
    ExternalMessage,
    /// Installing or removing a system package.
    PackageChange,
    /// Writing system configuration outside `~/.config`.
    SystemConfigWrite,
    /// A command run with elevated privileges.
    ElevatedCommand,
    /// A generic command-execution tool. Such a tool can do anything,
    /// so its specific effect cannot be judged from the name; it is
    /// confirmed unconditionally and fails closed.
    GenericExecution,
}

/// Classifier for the hardcoded always-confirm action set.
///
/// Phase 9-β classifies by tool name. It splits the name into word
/// segments, honouring `_` `-` `.` `/` separators *and* camelCase
/// boundaries, so `delete_file`, `deleteFile`, and
/// `filesystem.remove_file` all classify alike. Generic execution
/// tools (`run`, `exec`, `shell`, ...) are confirmed unconditionally
/// because the name alone cannot tell a harmless call from a
/// destructive one.
///
/// It is intentionally broad: a false positive only adds a
/// confirmation prompt, while a false negative would let an
/// irreversible action through silently. Phase 9-γ refines this with
/// call-argument inspection.
pub struct AlwaysConfirm;

impl AlwaysConfirm {
    /// Classify a tool call. Returns the reason it must be confirmed,
    /// or `None` if it is not in the always-confirm set.
    pub fn classify(tool: &str) -> Option<AlwaysConfirmReason> {
        let segments = name_segments(tool);
        for segment in &segments {
            let reason = match segment.as_str() {
                "delete" | "remove" | "trash" | "rm" | "unlink" | "rmdir" => {
                    Some(AlwaysConfirmReason::FileDeletion)
                }
                "send" | "email" | "mail" | "message" | "post" | "publish" => {
                    Some(AlwaysConfirmReason::ExternalMessage)
                }
                "install" | "uninstall" => Some(AlwaysConfirmReason::PackageChange),
                "sudo" | "elevate" | "pkexec" | "doas" => {
                    Some(AlwaysConfirmReason::ElevatedCommand)
                }
                "run" | "exec" | "execute" | "command" | "shell" | "spawn"
                | "eval" => Some(AlwaysConfirmReason::GenericExecution),
                _ => None,
            };
            if reason.is_some() {
                return reason;
            }
        }
        // A `config` word paired with a write-intent word is a system
        // configuration write, even when the two arrive as separate
        // segments (`write_config`, `set_system_config`, `configWrite`).
        let has = |w: &str| segments.iter().any(|s| s == w);
        if has("config")
            && (has("write") || has("set") || has("update")
                || has("edit") || has("system"))
        {
            return Some(AlwaysConfirmReason::SystemConfigWrite);
        }
        None
    }
}

/// Split a tool name into lowercase word segments, breaking on
/// `_` `-` `.` `/` ` ` separators and on camelCase boundaries. So
/// `deleteFile` and `filesystem.remove_file` both yield the word
/// `delete` / `remove` the classifier matches on.
fn name_segments(name: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut prev_was_lower_or_digit = false;
    for ch in name.chars() {
        if matches!(ch, '_' | '-' | '.' | '/' | ' ') {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
            prev_was_lower_or_digit = false;
            continue;
        }
        // An uppercase letter after a lowercase/digit starts a new
        // camelCase word.
        if ch.is_ascii_uppercase() && prev_was_lower_or_digit && !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
        cur.push(ch.to_ascii_lowercase());
        prev_was_lower_or_digit = ch.is_ascii_lowercase() || ch.is_ascii_digit();
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use rmcp::handler::server::router::tool::ToolRouter;
    use rmcp::{tool, tool_handler, tool_router, ServerHandler};
    use tokio::net::UnixListener;

    #[derive(Clone)]
    struct TestServer {
        tool_router: ToolRouter<Self>,
    }

    #[tool_router(router = tool_router)]
    impl TestServer {
        fn new() -> Self {
            Self {
                tool_router: Self::tool_router(),
            }
        }

        /// A harmless read-style tool.
        #[tool(name = "greeting")]
        async fn greeting(&self) -> Result<String, String> {
            Ok("hello from mcp".to_string())
        }

        /// A destructive-looking tool, to exercise always-confirm.
        #[tool(name = "delete_file")]
        async fn delete_file(&self) -> Result<String, String> {
            Ok("deleted".to_string())
        }
    }

    #[tool_handler(router = self.tool_router)]
    impl ServerHandler for TestServer {}

    /// Bind a Unix socket, accept one connection, and serve a
    /// `TestServer` on it. Returns the socket path and the server
    /// task handle.
    async fn spawn_test_server() -> (String, tokio::task::JoinHandle<()>) {
        let dir = std::env::temp_dir()
            .join(format!("lunaris-mcp-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let socket = dir.join("server.sock");
        let listener = UnixListener::bind(&socket).expect("bind socket");
        let path = socket.to_string_lossy().to_string();
        let handle = tokio::spawn(async move {
            if let Ok((stream, _)) = listener.accept().await {
                if let Ok(server) = TestServer::new().serve(stream).await {
                    let _ = server.waiting().await;
                }
            }
        });
        (path, handle)
    }

    #[tokio::test]
    async fn list_tools_returns_server_tools() {
        let (path, _server) = spawn_test_server().await;
        let mut client = McpClient::new();
        let id = ServerId("test".to_string());
        client.connect(id.clone(), &path, ServerClass::ReadOnly).await.expect("connect");

        let tools = client.list_tools(&id).await.expect("list tools");
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"greeting"), "tools: {names:?}");
        assert!(names.contains(&"delete_file"), "tools: {names:?}");
    }

    #[tokio::test]
    async fn call_tool_round_trips() {
        let (path, _server) = spawn_test_server().await;
        let mut client = McpClient::new();
        let id = ServerId("test".to_string());
        client.connect(id.clone(), &path, ServerClass::ReadOnly).await.expect("connect");

        let out = client
            .call_tool(&id, "greeting", serde_json::json!({}), &CallChain::root())
            .await
            .expect("call tool");
        assert!(out.contains("hello from mcp"), "got: {out}");
    }

    #[tokio::test]
    async fn depth_limit_rejects_before_dispatch() {
        let (path, _server) = spawn_test_server().await;
        let mut client = McpClient::with_max_depth(5);
        let id = ServerId("test".to_string());
        client.connect(id.clone(), &path, ServerClass::ReadOnly).await.expect("connect");

        // Build a chain at depth 6: root is 1, five nests reach 6.
        let mut chain = CallChain::root();
        for _ in 0..5 {
            chain = chain.nested();
        }
        let err = client
            .call_tool(&id, "greeting", serde_json::json!({}), &chain)
            .await
            .expect_err("depth 6 must be rejected");
        assert!(
            matches!(err, McpError::DepthExceeded { depth: 6, max: 5 }),
            "got: {err:?}"
        );
    }

    #[tokio::test]
    async fn unknown_server_is_rejected() {
        let client = McpClient::new();
        let err = client
            .list_tools(&ServerId("nonexistent".to_string()))
            .await
            .expect_err("unknown server");
        assert!(matches!(err, McpError::UnknownServer(_)));
    }

    #[tokio::test]
    async fn connect_to_missing_socket_is_unavailable() {
        let mut client = McpClient::new();
        let err = client
            .connect(
                ServerId("gone".to_string()),
                "/run/lunaris/mcp/nope.sock",
                ServerClass::ReadOnly,
            )
            .await
            .expect_err("missing socket");
        assert!(matches!(err, McpError::ServerUnavailable(_)));
    }

    #[test]
    fn always_confirm_classifies_destructive_tools() {
        assert_eq!(
            AlwaysConfirm::classify("delete_file"),
            Some(AlwaysConfirmReason::FileDeletion)
        );
        assert_eq!(
            AlwaysConfirm::classify("send_email"),
            Some(AlwaysConfirmReason::ExternalMessage)
        );
        assert_eq!(
            AlwaysConfirm::classify("install_package"),
            Some(AlwaysConfirmReason::PackageChange)
        );
        assert_eq!(
            AlwaysConfirm::classify("sudo_run"),
            Some(AlwaysConfirmReason::ElevatedCommand)
        );
        assert_eq!(
            AlwaysConfirm::classify("write_config"),
            Some(AlwaysConfirmReason::SystemConfigWrite)
        );
        // Read-style tools must not trip the classifier.
        assert_eq!(AlwaysConfirm::classify("list_directory"), None);
        assert_eq!(AlwaysConfirm::classify("get_note"), None);
        assert_eq!(AlwaysConfirm::classify("read_file"), None);
        // A near-miss substring must not match: `undelete` is a
        // distinct word, `configure` is not `config`.
        assert_eq!(AlwaysConfirm::classify("undelete"), None);
        assert_eq!(AlwaysConfirm::classify("configure_view"), None);
    }

    #[test]
    fn always_confirm_handles_camelcase_and_namespaced_names() {
        // camelCase boundaries split into words.
        assert_eq!(
            AlwaysConfirm::classify("deleteFile"),
            Some(AlwaysConfirmReason::FileDeletion)
        );
        assert_eq!(
            AlwaysConfirm::classify("sendEmail"),
            Some(AlwaysConfirmReason::ExternalMessage)
        );
        // Namespaced names split on `.` and `/`.
        assert_eq!(
            AlwaysConfirm::classify("filesystem.remove_file"),
            Some(AlwaysConfirmReason::FileDeletion)
        );
        assert_eq!(
            AlwaysConfirm::classify("pkg/uninstall"),
            Some(AlwaysConfirmReason::PackageChange)
        );
        // A generic execution tool is confirmed unconditionally; its
        // effect cannot be judged from the name.
        assert_eq!(
            AlwaysConfirm::classify("run_command"),
            Some(AlwaysConfirmReason::GenericExecution)
        );
        assert_eq!(
            AlwaysConfirm::classify("shell"),
            Some(AlwaysConfirmReason::GenericExecution)
        );
        assert_eq!(
            AlwaysConfirm::classify("evalExpression"),
            Some(AlwaysConfirmReason::GenericExecution)
        );
    }

    #[test]
    fn name_segments_splits_separators_and_camelcase() {
        assert_eq!(name_segments("deleteFile"), ["delete", "file"]);
        assert_eq!(
            name_segments("filesystem.remove_file"),
            ["filesystem", "remove", "file"]
        );
        assert_eq!(name_segments("HTTPGet"), ["httpget"]);
        assert_eq!(name_segments("get2Notes"), ["get2", "notes"]);
        assert_eq!(name_segments(""), [] as [&str; 0]);
    }

    #[test]
    fn call_chain_nesting_increments_depth_keeps_id() {
        let root = CallChain::root();
        assert_eq!(root.depth, 1);
        let nested = root.nested();
        assert_eq!(nested.depth, 2);
        assert_eq!(nested.id, root.id, "nested call keeps the chain id");
    }

    #[test]
    fn with_max_depth_clamps_to_ceiling() {
        assert_eq!(McpClient::with_max_depth(99).max_depth(), MAX_DEPTH_CEILING);
        assert_eq!(McpClient::with_max_depth(0).max_depth(), 1);
        assert_eq!(McpClient::with_max_depth(7).max_depth(), 7);
    }

    #[tokio::test]
    async fn read_only_server_tool_is_allowed_without_grant() {
        let (path, _server) = spawn_test_server().await;
        let mut client = McpClient::new();
        let id = ServerId("ro".to_string());
        client
            .connect(id.clone(), &path, ServerClass::ReadOnly)
            .await
            .expect("connect");
        assert_eq!(
            client.decide(&id, "greeting", false).expect("decide"),
            CallDecision::Allow
        );
    }

    #[tokio::test]
    async fn action_server_tool_needs_authorization() {
        let (path, _server) = spawn_test_server().await;
        let mut client = McpClient::new();
        let id = ServerId("act".to_string());
        client
            .connect(id.clone(), &path, ServerClass::Action)
            .await
            .expect("connect");
        // No grant: the action server's tool must not dispatch.
        assert_eq!(
            client.decide(&id, "greeting", false).expect("decide"),
            CallDecision::NeedsAuthorization
        );
        // With a grant: the same tool is allowed.
        assert_eq!(
            client.decide(&id, "greeting", true).expect("decide"),
            CallDecision::Allow
        );
    }

    #[tokio::test]
    async fn always_confirm_tool_confirms_even_with_grant() {
        let (path, _server) = spawn_test_server().await;
        let mut client = McpClient::new();
        let id = ServerId("act".to_string());
        client
            .connect(id.clone(), &path, ServerClass::Action)
            .await
            .expect("connect");
        // delete_file is in the hardcoded always-confirm set; holding
        // an authorization grant must not bypass the confirmation.
        let decision = client.decide(&id, "delete_file", true).expect("decide");
        assert!(
            matches!(
                decision,
                CallDecision::NeedsConfirmation(AlwaysConfirmReason::FileDeletion)
            ),
            "got: {decision:?}"
        );
    }

    #[tokio::test]
    async fn decide_on_unknown_server_errors() {
        let client = McpClient::new();
        let err = client
            .decide(&ServerId("nope".to_string()), "x", false)
            .expect_err("unknown server");
        assert!(matches!(err, McpError::UnknownServer(_)));
    }

    #[tokio::test]
    async fn server_class_is_recorded_on_connect() {
        // The discovery layer registers third-party module servers
        // as `Action` regardless of what their manifest claims; the
        // client just records and trusts the class it was given.
        let (path, _server) = spawn_test_server().await;
        let mut client = McpClient::new();
        let id = ServerId("module".to_string());
        client
            .connect(id.clone(), &path, ServerClass::Action)
            .await
            .expect("connect");
        assert_eq!(client.server_class(&id), Some(ServerClass::Action));
    }
}
