//! Production [`GraphHandle`]: a read-only Knowledge Graph client.
//!
//! Wraps `os_sdk::UnixGraphClient` so a behaviour handler reads through the
//! object-safe [`GraphHandle`] seam. Read *scope* is enforced by the
//! knowledge daemon against the agent's configured access tier, and the
//! dispatcher refuses any behaviour whose declared read tier exceeds the
//! grant before the handler runs.

use std::collections::HashMap;

use os_sdk::{RelationWriteOutcome, UnixGraphClient};

use crate::executor::{RelationWrite, RelationWriter, WriteError, WriteOutcome};
use crate::seams::{GraphError, GraphHandle};

/// The knowledge daemon's query socket. The daemon resolves the real path
/// from `LUNARIS_KNOWLEDGE_SOCKET` (with this as the fallback).
pub const DEFAULT_GRAPH_SOCKET: &str = "/run/lunaris/knowledge.sock";

/// A [`GraphHandle`] backed by the knowledge daemon's read-only Cypher
/// socket.
///
/// Each query uses a **fresh** client (and therefore a fresh connection).
/// `UnixGraphClient` is itself cancellation-safe (it takes its cached stream
/// out of the mutex for the duration of a round trip, so a dropped future
/// never leaves a desynchronised socket for the next query), but a per-query
/// client keeps this handle stateless and isolates any per-connection error
/// to the single query that hit it. The agent's query rate is low, so the
/// reconnect cost is acceptable.
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
        // Use the daemon's typed structured-row mode, not the legacy text
        // mode: a behaviour reads `row["id"]` as a properly-typed value, and a
        // graph string containing a delimiter or newline cannot forge a row.
        UnixGraphClient::new(self.socket_path.clone())
            .query_rows(cypher)
            .await
            .map_err(|e| GraphError::Failed(e.to_string()))
    }
}

/// Production [`RelationWriter`]: writes through the knowledge daemon's write
/// socket via `os_sdk::UnixGraphClient::create_relation`.
///
/// Like [`UnixGraph`], each write uses a fresh client (fresh connection); the
/// agent's write rate is low. The daemon authorises the relation against the
/// agent's permission profile and persists it idempotently, so a retry after a
/// transport failure re-confirms the edge rather than duplicating it.
pub struct UnixRelationWriter {
    socket_path: String,
}

impl UnixRelationWriter {
    /// Build a relation writer for the given daemon write socket (the same
    /// socket path as the query socket; the daemon multiplexes read and write
    /// modes on it).
    pub fn new(socket_path: impl Into<String>) -> Self {
        Self {
            socket_path: socket_path.into(),
        }
    }
}

#[async_trait::async_trait]
impl RelationWriter for UnixRelationWriter {
    async fn write_relation(&self, write: &RelationWrite) -> Result<WriteOutcome, WriteError> {
        let outcome = UnixGraphClient::new(self.socket_path.clone())
            .create_relation(
                &write.from_type,
                &write.from_id,
                &write.to_type,
                &write.to_id,
                &write.relation_type,
            )
            .await
            .map_err(|e| WriteError::Failed(e.to_string()))?;
        Ok(match outcome {
            RelationWriteOutcome::Created => WriteOutcome::Created,
            RelationWriteOutcome::AlreadyExists => WriteOutcome::AlreadyExists,
        })
    }
}
