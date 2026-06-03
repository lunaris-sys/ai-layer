//! Production [`GraphHandle`]: a read-only Knowledge Graph client.
//!
//! Wraps `os_sdk::UnixGraphClient` so a behaviour handler reads through the
//! object-safe [`GraphHandle`] seam. Read *scope* is enforced by the
//! knowledge daemon against the agent's configured access tier, and the
//! dispatcher refuses any behaviour whose declared read tier exceeds the
//! grant before the handler runs.

use std::collections::HashMap;

use os_sdk::{GraphClient, UnixGraphClient};

use crate::seams::{GraphError, GraphHandle};

/// The knowledge daemon's query socket. The daemon resolves the real path
/// from `LUNARIS_KNOWLEDGE_SOCKET` (with this as the fallback).
pub const DEFAULT_GRAPH_SOCKET: &str = "/run/lunaris/knowledge.sock";

/// A [`GraphHandle`] backed by the knowledge daemon's read-only Cypher
/// socket.
///
/// Each query uses a **fresh** client (and therefore a fresh connection).
/// `UnixGraphClient` caches a persistent socket and only resets it on an
/// explicit I/O error, not when a future is dropped; under the dispatcher's
/// per-handler timeout a query future can be cancelled mid-I/O, which would
/// leave a shared cached stream desynchronised and corrupt the *next*
/// query. A per-query client makes a timed-out query drop only its own
/// connection, so cancellation is safe. (The agent's query rate is low, so
/// the reconnect cost is acceptable.)
pub struct UnixGraph {
    socket_path: String,
}

impl UnixGraph {
    /// Build a graph handle for the given query socket.
    pub fn new(socket_path: impl Into<String>) -> Self {
        Self {
            socket_path: socket_path.into(),
        }
    }
}

#[async_trait::async_trait]
impl GraphHandle for UnixGraph {
    async fn query(
        &self,
        cypher: &str,
    ) -> Result<Vec<HashMap<String, serde_json::Value>>, GraphError> {
        UnixGraphClient::new(self.socket_path.clone())
            .query(cypher, HashMap::new())
            .await
            .map_err(|e| GraphError::Failed(e.to_string()))
    }
}
