//! Production [`GraphHandle`]: a read-only Knowledge Graph client.
//!
//! Wraps `os_sdk::UnixGraphClient` so a behaviour handler reads through the
//! object-safe [`GraphHandle`] seam. Read *scope* is enforced by the
//! knowledge daemon against the agent's configured access tier, and the
//! dispatcher refuses any behaviour whose declared read tier exceeds the
//! grant before the handler runs.

use std::collections::HashMap;

use os_sdk::{QueryError, RelationWriteOutcome, UnixGraphClient};

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
            .map_err(map_write_error)?;
        Ok(match outcome {
            RelationWriteOutcome::Created => WriteOutcome::Created,
            RelationWriteOutcome::AlreadyExists => WriteOutcome::AlreadyExists,
        })
    }
}

/// Classify an os-sdk write error by whether the relation could have committed,
/// conservatively: only a **reliable** definite no-commit is `Failed`, and
/// anything that could have left a committed write upstream is `Indeterminate`.
///
/// `PermissionDenied` is the one reliable definite no-commit: the daemon
/// authorises before it writes, so an auth rejection means nothing was created
/// (and it is persistent, so reconciliation must not keep treating it as
/// maybe-committed). Everything else is treated as commit-unknown, because the
/// coarse `QueryError` cannot separate the phases within it: `ConnectionFailed`
/// covers both a pre-send connect failure (no commit) and a post-send drop
/// (unknown); `InvalidQuery` covers both a daemon `ERROR:` rejection (no commit)
/// and an undecodable/oversized response *after* the daemon processed the
/// request (unknown). Defaulting those to `Indeterminate` is the safe direction:
/// a false indeterminate is reconciled by the next run's re-validation (an
/// already-committed edge fails the `Not(EdgeExists)` proof; an absent one is
/// re-written), whereas a false `Failed` would discard a real commit. Precise
/// per-phase certainty needs durable operation identity (an idempotency key) and
/// a reconciliation query, the deferred executor-design follow-up; this is the
/// honest conservative mapping until then.
fn map_write_error(e: QueryError) -> WriteError {
    match e {
        QueryError::PermissionDenied => WriteError::Failed(e.to_string()),
        QueryError::ConnectionFailed(msg) => WriteError::Indeterminate(msg),
        QueryError::InvalidQuery(msg) => WriteError::Indeterminate(msg),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_errors_are_classified_by_commit_certainty() {
        // Only a reliable daemon auth rejection is a definite no-commit.
        assert!(matches!(
            map_write_error(QueryError::PermissionDenied),
            WriteError::Failed(_)
        ));
        // A transport failure is commit-unknown (the request may have been sent).
        assert!(matches!(
            map_write_error(QueryError::ConnectionFailed("reset".into())),
            WriteError::Indeterminate(_)
        ));
        // An InvalidQuery conflates a daemon rejection with a post-commit decode
        // failure, so it is conservatively commit-unknown, never a false Failed.
        assert!(matches!(
            map_write_error(QueryError::InvalidQuery("unexpected response".into())),
            WriteError::Indeterminate(_)
        ));
    }
}
