//! Per-call audit emission.
//!
//! The full audit-log subsystem (hash-chain ledger, anomaly detector
//! consumption) lands in Phase 9-γ S13. Until then the proxy emits
//! structured [`tracing`] events through a pluggable
//! [`AuditSink`] so callers can collect them in tests, and the daemon
//! binary can swap in the eventual ledger client without touching
//! call sites.

use async_trait::async_trait;
use serde::Serialize;

/// A single outbound-call audit record.
#[derive(Debug, Clone, Serialize)]
pub struct AuditRecord {
    /// Capability token presented by the caller. The proxy does not
    /// validate this; it just records it so the audit ledger can
    /// cross-reference the issuing daemon later.
    pub audit_token: String,
    /// Provider name from the routing config.
    pub provider_name: String,
    /// Canonical host portion of the endpoint URL after allowlist
    /// normalisation. None if the call was rejected before host
    /// extraction.
    pub host: Option<String>,
    /// Outcome of the call.
    pub outcome: AuditOutcome,
}

/// Outcome label captured in the audit record.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum AuditOutcome {
    /// Request reached the upstream and was forwarded back.
    Forwarded {
        /// HTTP status code returned by the upstream.
        upstream_status: u16,
    },
    /// Rejected before any outbound call.
    RejectedByPolicy {
        /// Stable error code matching
        /// [`crate::service::ProxyError`] variants.
        code: String,
    },
    /// Outbound call started but failed transport-side.
    UpstreamError {
        /// Free-form error string from the transport layer.
        detail: String,
    },
}

/// Sink for audit records. The library code (and the unit tests)
/// depends on this trait, not on the concrete logging implementation.
#[async_trait]
pub trait AuditSink: Send + Sync {
    /// Record one audit entry.
    async fn record(&self, record: AuditRecord);
}

/// Default sink. Emits a structured `tracing` event at INFO level.
#[derive(Debug, Default, Clone)]
pub struct TracingAuditSink;

#[async_trait]
impl AuditSink for TracingAuditSink {
    async fn record(&self, record: AuditRecord) {
        tracing::info!(
            audit_token = %record.audit_token,
            provider = %record.provider_name,
            host = ?record.host,
            outcome = ?record.outcome,
            "ai-proxy outbound call audited"
        );
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::*;
    use std::sync::Arc;
    use tokio::sync::Mutex;

    /// Audit sink that buffers records in memory.
    #[derive(Debug, Default, Clone)]
    pub struct CollectingAuditSink {
        records: Arc<Mutex<Vec<AuditRecord>>>,
    }

    impl CollectingAuditSink {
        pub fn new() -> Self {
            Self::default()
        }

        pub async fn snapshot(&self) -> Vec<AuditRecord> {
            self.records.lock().await.clone()
        }
    }

    #[async_trait]
    impl AuditSink for CollectingAuditSink {
        async fn record(&self, record: AuditRecord) {
            self.records.lock().await.push(record);
        }
    }
}
