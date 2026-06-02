//! Per-call audit emission.
//!
//! Every outbound provider call the proxy makes is recorded in the
//! system audit ledger written by `lunaris-auditd` (foundation
//! §8.4.7). The service depends on the shared [`audit_proto::AuditSink`]
//! trait — the same one the AI daemon uses, so the two components
//! share one trust level — and the daemon binary wires
//! [`audit_proto::LedgerAuditSink`]. Unit tests collect the submitted
//! ingest requests in memory.
//!
//! [`AuditRecord`] is the proxy's internal description of one call;
//! [`AuditRecord::to_ingest_request`] maps it to the content-free wire
//! type at the sink boundary.
//!
//! Auditing the *outbound call* is fail-closed: the proxy commits a
//! pre-forward entry before the request leaves the host and refuses
//! the call if that entry cannot be recorded (foundation §8.4.6, no
//! un-audited AI network activity). The post-call status entry and
//! the rejection entries — where nothing left the host — are
//! best-effort.

use serde::Serialize;

pub use audit_proto::{AuditSink, LedgerAuditSink};

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
    /// The pre-forward entry: an outbound call to the host is about to
    /// be made. This is the fail-closed gate entry, committed before
    /// the request leaves the host.
    Forwarding,
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

impl AuditRecord {
    /// Map this record to an `audit-proto` ingest request for the
    /// system audit ledger.
    ///
    /// The Structural record is content-free: `subject` is the
    /// outbound host (a coarse network destination), or the provider
    /// name when the call was rejected before the host was resolved.
    /// The HTTP status, when known, is folded into the coarse
    /// `outcome` label. The capability token is deliberately dropped:
    /// `lunaris-auditd` attributes every entry from the connection's
    /// kernel-attested peer credentials, never from a request field.
    pub fn to_ingest_request(&self) -> audit_proto::IngestRequest {
        let outcome = match &self.outcome {
            AuditOutcome::Forwarding => "forwarding".to_string(),
            AuditOutcome::Forwarded { upstream_status } => {
                format!("forwarded-{upstream_status}")
            }
            AuditOutcome::RejectedByPolicy { code } => code.clone(),
            AuditOutcome::UpstreamError { .. } => "upstream-error".to_string(),
        };
        // The subject must be a TRUSTED identifier. `host` is derived
        // from the proxy-owned provider catalog after lookup, so it is
        // safe. `provider_name` is the raw caller-supplied request
        // argument and is deliberately NOT used here: a rejected call
        // (caller-not-allowed, unknown-provider) has no resolved host,
        // and copying the caller's string into the always-recorded
        // Structural tier would let any session-bus peer inject
        // content/noise under the ai-proxy actor. Such entries get a
        // fixed subject; the outcome label carries the rejection code.
        let subject = self
            .host
            .clone()
            .unwrap_or_else(|| "network-call".to_string());
        audit_proto::IngestRequest {
            kind: audit_proto::AuditKind::NetworkCall,
            structural: audit_proto::StructuralRecord {
                subject,
                node_types: Vec::new(),
                relations: Vec::new(),
                result_count: None,
                duration_ms: None,
                outcome,
                depth: None,
            },
            forensic: None,
            call_chain_id: None,
            project_id: None,
        }
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::*;
    use async_trait::async_trait;
    use audit_proto::client::AuditClientError;
    use audit_proto::IngestRequest;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::Arc;
    use tokio::sync::Mutex;

    /// Audit sink that buffers submitted ingest requests in memory.
    /// `failing()` rejects every submit so the proxy's fail-closed
    /// path can be exercised.
    #[derive(Clone)]
    pub struct CollectingAuditSink {
        requests: Arc<Mutex<Vec<IngestRequest>>>,
        accepting: Arc<AtomicBool>,
        next_index: Arc<AtomicU64>,
    }

    impl Default for CollectingAuditSink {
        fn default() -> Self {
            Self {
                requests: Arc::new(Mutex::new(Vec::new())),
                accepting: Arc::new(AtomicBool::new(true)),
                next_index: Arc::new(AtomicU64::new(0)),
            }
        }
    }

    impl CollectingAuditSink {
        pub fn new() -> Self {
            Self::default()
        }

        /// A sink that rejects every submit.
        pub fn failing() -> Self {
            let sink = Self::default();
            sink.accepting.store(false, Ordering::SeqCst);
            sink
        }

        /// The ingest requests submitted so far, in order.
        pub async fn snapshot(&self) -> Vec<IngestRequest> {
            self.requests.lock().await.clone()
        }
    }

    #[async_trait]
    impl AuditSink for CollectingAuditSink {
        async fn submit(&self, event: IngestRequest) -> Result<u64, AuditClientError> {
            if !self.accepting.load(Ordering::SeqCst) {
                return Err(AuditClientError::Unavailable("collecting sink: failing".into()));
            }
            self.requests.lock().await.push(event);
            Ok(self.next_index.fetch_add(1, Ordering::SeqCst))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forwarded_record_maps_to_a_content_free_network_call() {
        let rec = AuditRecord {
            audit_token: "tok".to_string(),
            provider_name: "ollama-default".to_string(),
            host: Some("api.example.com".to_string()),
            outcome: AuditOutcome::Forwarded {
                upstream_status: 200,
            },
        };
        let req = rec.to_ingest_request();
        assert_eq!(req.kind, audit_proto::AuditKind::NetworkCall);
        // The host is the coarse subject; the status folds into the
        // outcome label. No request body, no token.
        assert_eq!(req.structural.subject, "api.example.com");
        assert_eq!(req.structural.outcome, "forwarded-200");
        assert!(req.forensic.is_none());
    }

    #[test]
    fn forwarding_record_is_the_pre_forward_gate_entry() {
        let rec = AuditRecord {
            audit_token: "tok".to_string(),
            provider_name: "ollama-default".to_string(),
            host: Some("api.example.com".to_string()),
            outcome: AuditOutcome::Forwarding,
        };
        let req = rec.to_ingest_request();
        assert_eq!(req.structural.subject, "api.example.com");
        assert_eq!(req.structural.outcome, "forwarding");
    }

    #[test]
    fn rejected_record_uses_a_fixed_subject_not_the_caller_provider_name() {
        // A rejected call has no resolved host. The caller-supplied
        // provider_name (here a content-looking string) must NOT reach
        // the Structural subject — only a fixed label does.
        let rec = AuditRecord {
            audit_token: "tok".to_string(),
            provider_name: "please log this secret sentence as my provider".to_string(),
            host: None,
            outcome: AuditOutcome::RejectedByPolicy {
                code: "unknown-provider".to_string(),
            },
        };
        let req = rec.to_ingest_request();
        assert_eq!(req.structural.subject, "network-call");
        assert_ne!(req.structural.subject, rec.provider_name);
        assert_eq!(req.structural.outcome, "unknown-provider");
    }
}
