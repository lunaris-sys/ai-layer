//! Knowledge Graph adapter.
//!
//! Bridges the os-sdk [`UnixGraphClient`] (which talks the Knowledge
//! Daemon's Unix-socket Cypher protocol) onto the ai-core
//! [`GraphQuerier`] trait the pipeline depends on.
//!
//! The os-sdk `GraphClient` trait uses return-position `impl Trait`,
//! which is not object-safe; `GraphQuerier` uses `async_trait` so the
//! pipeline can hold it behind an `Arc<dyn _>`. This adapter is the
//! thin glue between the two.

use std::collections::HashMap;

use async_trait::async_trait;
use lunaris_ai_core::pipeline::{GraphQuerier, GraphQueryError, GraphRow};
use os_sdk::graph::{GraphClient, QueryError, UnixGraphClient};

/// [`GraphQuerier`] backed by the os-sdk Unix-socket graph client.
pub struct OsSdkGraphQuerier {
    client: UnixGraphClient,
}

impl OsSdkGraphQuerier {
    /// Build an adapter pointing at the Knowledge Daemon socket.
    pub fn new(socket_path: impl Into<String>) -> Self {
        Self {
            client: UnixGraphClient::new(socket_path),
        }
    }
}

#[async_trait]
impl GraphQuerier for OsSdkGraphQuerier {
    async fn run(&self, cypher: &str) -> Result<Vec<GraphRow>, GraphQueryError> {
        self.client
            .query(cypher, HashMap::new())
            .await
            .map_err(|err| match err {
                QueryError::ConnectionFailed(msg) => GraphQueryError::Unreachable(msg),
                QueryError::InvalidQuery(msg) => GraphQueryError::Rejected(msg),
                QueryError::PermissionDenied => {
                    GraphQueryError::Rejected("permission denied".to_string())
                }
            })
    }
}
