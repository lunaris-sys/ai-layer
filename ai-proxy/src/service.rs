//! Proxy service core.
//!
//! [`ProxyService`] holds the allowlist, the trusted provider
//! catalog, the caller allowlist, the audit sink, and the outbound
//! forwarder. The D-Bus surface in `main.rs` is a thin wrapper that
//! converts D-Bus method calls into [`ProxyService::forward`] calls
//! and back. Keeping the service detached from the D-Bus layer keeps
//! every policy decision exercised in unit tests.
//!
//! ## Trust boundaries (Foundation §8.4.6)
//!
//! 1. **The caller does not supply the endpoint URL.** Callers
//!    identify the upstream by a provider *name*. The URL is looked
//!    up from the proxy-owned [`ProviderCatalog`].
//! 2. **The proxy verifies its callers.** Only the
//!    [`CallerAllowlist`] (defaulted to `ai-daemon` + `ai-agent`)
//!    may invoke the proxy. Anything else is rejected before any
//!    outbound work.
//! 3. **The allowlist hostname check still runs**, applied to the
//!    catalogued URL, so a misconfigured catalog cannot smuggle a
//!    new host past the proxy without an explicit allowlist update.

use std::collections::BTreeSet;
use std::sync::Arc;

use crate::allowlist::{Allowlist, AllowlistDecision, RejectReason};
use crate::audit::{AuditOutcome, AuditRecord, AuditSink};
use crate::catalog::ProviderCatalog;
use crate::forward::{ForwardError, Forwarder};

/// Stable error codes returned to D-Bus callers. The audit log
/// records the same `code()` string so logs and caller-side errors
/// align.
#[derive(Debug, thiserror::Error)]
pub enum ProxyError {
    /// Caller is not in the proxy's caller allowlist.
    #[error("caller not allowed: {caller}")]
    CallerNotAllowed {
        /// Unique caller identifier as reported by D-Bus.
        caller: String,
    },
    /// `provider_name` not present in the trusted catalog.
    #[error("unknown provider: {provider}")]
    UnknownProvider {
        /// Provider name from the request.
        provider: String,
    },
    /// Allowlist rejected the catalogued URL.
    #[error("allowlist: {0:?}")]
    Allowlist(RejectReason),
    /// The proxy is already forwarding its maximum number of
    /// concurrent upstream calls.
    #[error("proxy at concurrency capacity")]
    AtCapacity,
    /// The pre-forward audit entry could not be committed, so the
    /// outbound call was refused before the request left the host.
    /// Foundation §8.4.6: no un-audited AI network activity.
    #[error("audit log unavailable")]
    AuditUnavailable,
    /// Upstream call failed transport-side.
    #[error("upstream: {0}")]
    Upstream(#[from] ForwardError),
}

impl ProxyError {
    /// Stable kebab-case error code used in audit records and as the
    /// `org.lunaris.AIProxy1.<Code>` D-Bus error name.
    pub fn code(&self) -> &'static str {
        match self {
            ProxyError::CallerNotAllowed { .. } => "caller-not-allowed",
            ProxyError::UnknownProvider { .. } => "unknown-provider",
            ProxyError::Allowlist(RejectReason::InvalidUrl) => "invalid-url",
            ProxyError::Allowlist(RejectReason::MissingHost) => "missing-host",
            ProxyError::Allowlist(RejectReason::DisallowedScheme { .. }) => "disallowed-scheme",
            ProxyError::Allowlist(RejectReason::HostNotAllowed { .. }) => "host-not-allowed",
            ProxyError::AtCapacity => "proxy-at-capacity",
            ProxyError::AuditUnavailable => "audit-unavailable",
            ProxyError::Upstream(_) => "upstream-error",
        }
    }
}

/// Default ceiling on concurrent upstream forwards. A backstop so
/// that even if a daemon's own in-flight accounting is bypassed
/// (for example by submit/cancel churn) the proxy still bounds the
/// real outbound work it performs.
pub const DEFAULT_MAX_INFLIGHT: usize = 8;

/// RAII slot in the proxy's concurrency counter. Decrements on drop,
/// so every `forward` return path releases its slot.
struct InflightGuard(std::sync::Arc<std::sync::atomic::AtomicUsize>);

impl Drop for InflightGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
    }
}

/// Caller identity passed into [`ProxyService::forward`] by the D-Bus
/// layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallerIdentity {
    /// Well-known bus name of the caller, if it owns one. The D-Bus
    /// layer fills this in from the message header.
    pub well_known_bus_name: Option<String>,
    /// Unique bus name (`":1.42"`) of the caller. Always present
    /// because every connection has one.
    pub unique_bus_name: String,
}

impl CallerIdentity {
    /// Compact identifier used in audit records and error messages.
    pub fn label(&self) -> &str {
        self.well_known_bus_name
            .as_deref()
            .unwrap_or(&self.unique_bus_name)
    }
}

/// Set of bus names permitted to invoke the proxy.
#[derive(Debug, Clone)]
pub struct CallerAllowlist {
    well_known_names: BTreeSet<String>,
}

impl CallerAllowlist {
    /// Build from any iterable of well-known names. The unique bus
    /// names (`":1.NN"`) are not allowlisted; only well-known names
    /// are stable across daemon restarts.
    pub fn new<I, S>(names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            well_known_names: names.into_iter().map(Into::into).collect(),
        }
    }

    /// The default Lunaris caller allowlist: only the AI daemons.
    pub fn default_lunaris() -> Self {
        Self::new(["org.lunaris.AI1", "org.lunaris.AIAgent1"])
    }

    /// Whether the caller is permitted.
    pub fn permits(&self, caller: &CallerIdentity) -> bool {
        match &caller.well_known_bus_name {
            Some(name) => self.well_known_names.contains(name),
            None => false,
        }
    }

    /// Iterator over allowed well-known names. Used by
    /// `list_allowed_endpoints` only for logging; callers cannot
    /// enumerate this list over D-Bus.
    pub fn allowed_names(&self) -> impl Iterator<Item = &str> {
        self.well_known_names.iter().map(String::as_str)
    }
}

/// Input to a single forwarded call.
#[derive(Debug, Clone)]
pub struct ForwardRequest {
    /// Provider catalog key. Maps onto a trusted endpoint URL.
    pub provider_name: String,
    /// JSON body to POST. The proxy does not re-serialise it.
    pub body_json: String,
    /// Capability token presented by the caller. Recorded in the
    /// audit log; not interpreted here.
    pub audit_token: String,
}

/// Output of a forwarded call.
#[derive(Debug, Clone)]
pub struct ForwardOutcome {
    /// HTTP status the upstream returned.
    pub upstream_status: u16,
    /// Upstream response body.
    pub body: String,
}

/// Proxy service. Holds the policy plus the wired-in dependencies
/// (forwarder + audit sink).
pub struct ProxyService {
    allowlist: Allowlist,
    catalog: ProviderCatalog,
    caller_allowlist: CallerAllowlist,
    forwarder: Arc<dyn Forwarder>,
    audit_sink: Arc<dyn AuditSink>,
    inflight: Arc<std::sync::atomic::AtomicUsize>,
    max_inflight: usize,
}

impl ProxyService {
    /// Build the service with the default concurrency ceiling.
    pub fn new(
        allowlist: Allowlist,
        catalog: ProviderCatalog,
        caller_allowlist: CallerAllowlist,
        forwarder: Arc<dyn Forwarder>,
        audit_sink: Arc<dyn AuditSink>,
    ) -> Self {
        Self::with_max_inflight(
            allowlist,
            catalog,
            caller_allowlist,
            forwarder,
            audit_sink,
            DEFAULT_MAX_INFLIGHT,
        )
    }

    /// Build with an explicit concurrency ceiling. Tests use a small
    /// ceiling to exercise the at-capacity path.
    pub fn with_max_inflight(
        allowlist: Allowlist,
        catalog: ProviderCatalog,
        caller_allowlist: CallerAllowlist,
        forwarder: Arc<dyn Forwarder>,
        audit_sink: Arc<dyn AuditSink>,
        max_inflight: usize,
    ) -> Self {
        Self {
            allowlist,
            catalog,
            caller_allowlist,
            forwarder,
            audit_sink,
            inflight: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            max_inflight,
        }
    }

    /// Names of catalogued providers the proxy will accept. Returned
    /// to the D-Bus surface for `list_allowed_endpoints`.
    pub fn allowed_providers(&self) -> Vec<String> {
        self.catalog.names().map(str::to_string).collect()
    }

    /// Run a single forward call. The audit sink is invoked
    /// regardless of outcome.
    pub async fn forward(
        &self,
        caller: &CallerIdentity,
        req: ForwardRequest,
    ) -> Result<ForwardOutcome, ProxyError> {
        // 1. Caller allowlist.
        if !self.caller_allowlist.permits(caller) {
            let err = ProxyError::CallerNotAllowed {
                caller: caller.label().to_string(),
            };
            self.audit_best_effort(
                &req,
                None,
                AuditOutcome::RejectedByPolicy {
                    code: err.code().to_string(),
                },
            )
            .await;
            return Err(err);
        }

        // 2. Catalog lookup.
        let entry = match self.catalog.get(&req.provider_name) {
            Some(entry) => entry,
            None => {
                let err = ProxyError::UnknownProvider {
                    provider: req.provider_name.clone(),
                };
                self.audit_best_effort(
                    &req,
                    None,
                    AuditOutcome::RejectedByPolicy {
                        code: err.code().to_string(),
                    },
                )
                .await;
                return Err(err);
            }
        };
        let endpoint_url = entry.endpoint_url.clone();

        // 3. Allowlist on the catalogued URL (defence in depth).
        let host = match self.allowlist.check(&endpoint_url) {
            AllowlistDecision::Allowed { host } => host,
            AllowlistDecision::Rejected(reason) => {
                let err = ProxyError::Allowlist(reason);
                self.audit_best_effort(
                    &req,
                    None,
                    AuditOutcome::RejectedByPolicy {
                        code: err.code().to_string(),
                    },
                )
                .await;
                return Err(err);
            }
        };

        // 4. Reserve a concurrency slot before doing the real
        //    outbound work. The guard releases it on every return
        //    path below. If the proxy is already at its ceiling the
        //    call is refused here, so a flood of forwards cannot
        //    multiply real upstream traffic.
        let prev = self
            .inflight
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let _slot = InflightGuard(self.inflight.clone());
        if prev >= self.max_inflight {
            let err = ProxyError::AtCapacity;
            self.audit_best_effort(
                &req,
                Some(&host),
                AuditOutcome::RejectedByPolicy {
                    code: err.code().to_string(),
                },
            )
            .await;
            return Err(err);
        }

        // 5. Audit-before-action gate (foundation §8.4.6). The proxy
        //    is the network egress chokepoint, so it must record the
        //    outbound call *before* it leaves the host and refuse the
        //    call if the ledger cannot record it. On this early return
        //    `_slot` drops and releases the concurrency slot.
        self.audit_forwarding_gate(&req, &host).await?;

        // 6. Forward. The status entry is best-effort: the call has
        //    already happened, so a ledger hiccup here does not undo
        //    it; the pre-forward entry already satisfies §8.4.6.
        match self.forwarder.post(&endpoint_url, &req.body_json).await {
            Ok(result) => {
                self.audit_best_effort(
                    &req,
                    Some(&host),
                    AuditOutcome::Forwarded {
                        upstream_status: result.status,
                    },
                )
                .await;
                Ok(ForwardOutcome {
                    upstream_status: result.status,
                    body: result.body,
                })
            }
            Err(err) => {
                let detail = err.to_string();
                self.audit_best_effort(
                    &req,
                    Some(&host),
                    AuditOutcome::UpstreamError { detail },
                )
                .await;
                Err(ProxyError::Upstream(err))
            }
        }
    }

    /// Commit the fail-closed pre-forward entry. Returns
    /// `Err(ProxyError::AuditUnavailable)` if the ledger cannot record
    /// it, so the caller refuses the forward rather than letting an
    /// unaudited request leave the host.
    async fn audit_forwarding_gate(
        &self,
        req: &ForwardRequest,
        host: &str,
    ) -> Result<(), ProxyError> {
        let record = AuditRecord {
            audit_token: req.audit_token.clone(),
            provider_name: req.provider_name.clone(),
            host: Some(host.to_string()),
            outcome: AuditOutcome::Forwarding,
        };
        self.audit_sink
            .submit(record.to_ingest_request())
            .await
            .map(|_| ())
            .map_err(|err| {
                tracing::warn!(
                    "ai-proxy forward refused: audit log unavailable: {err}"
                );
                ProxyError::AuditUnavailable
            })
    }

    /// Record one audit entry best-effort: a ledger failure is logged,
    /// not propagated. Used for rejections (nothing left the host) and
    /// for the post-forward status entry (the call already happened).
    async fn audit_best_effort(
        &self,
        req: &ForwardRequest,
        host: Option<&str>,
        outcome: AuditOutcome,
    ) {
        let record = AuditRecord {
            audit_token: req.audit_token.clone(),
            provider_name: req.provider_name.clone(),
            host: host.map(str::to_string),
            outcome,
        };
        if let Err(err) = self.audit_sink.submit(record.to_ingest_request()).await {
            tracing::warn!("ai-proxy audit submit failed: {err}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audit::test_support::CollectingAuditSink;
    use crate::forward::test_support::StubForwarder;
    use crate::forward::ForwardResult;
    use async_trait::async_trait;

    fn ai_daemon_caller() -> CallerIdentity {
        CallerIdentity {
            well_known_bus_name: Some("org.lunaris.AI1".to_string()),
            unique_bus_name: ":1.42".to_string(),
        }
    }

    fn service_with(
        forwarder: Arc<StubForwarder>,
        sink: Arc<CollectingAuditSink>,
    ) -> ProxyService {
        ProxyService::new(
            Allowlist::default_lunaris(),
            ProviderCatalog::default_lunaris(),
            CallerAllowlist::default_lunaris(),
            forwarder as Arc<dyn Forwarder>,
            sink as Arc<dyn AuditSink>,
        )
    }

    #[tokio::test]
    async fn happy_path_forwards_via_catalog_url() {
        let forwarder = Arc::new(StubForwarder::new(vec![Ok(ForwardResult {
            status: 200,
            body: r#"{"ok":true}"#.to_string(),
        })]));
        let sink = Arc::new(CollectingAuditSink::new());
        let svc = service_with(forwarder.clone(), sink.clone());

        let out = svc
            .forward(
                &ai_daemon_caller(),
                ForwardRequest {
                    provider_name: "ollama-default".to_string(),
                    body_json: "{}".to_string(),
                    audit_token: "tok-1".to_string(),
                },
            )
            .await
            .expect("ok");
        assert_eq!(out.upstream_status, 200);

        let calls = forwarder.calls.lock().await;
        assert_eq!(calls.len(), 1);
        // The forwarder must have been called with the *catalogued*
        // URL, not with anything the caller supplied.
        assert_eq!(calls[0].0, "http://localhost:11434/v1/chat/completions");

        let records = sink.snapshot().await;
        // Two entries: the fail-closed pre-forward gate, then the
        // best-effort status entry.
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].structural.outcome, "forwarding");
        assert_eq!(records[0].structural.subject, "localhost");
        assert_eq!(records[1].structural.outcome, "forwarded-200");
    }

    #[tokio::test]
    async fn caller_not_in_allowlist_is_rejected() {
        let forwarder = Arc::new(StubForwarder::new(vec![]));
        let sink = Arc::new(CollectingAuditSink::new());
        let svc = service_with(forwarder.clone(), sink.clone());

        let unknown = CallerIdentity {
            well_known_bus_name: Some("com.example.evil".to_string()),
            unique_bus_name: ":1.7".to_string(),
        };
        let err = svc
            .forward(
                &unknown,
                ForwardRequest {
                    provider_name: "ollama-default".to_string(),
                    body_json: "{}".to_string(),
                    audit_token: "tok-2".to_string(),
                },
            )
            .await
            .expect_err("reject");
        assert_eq!(err.code(), "caller-not-allowed");
        assert!(forwarder.calls.lock().await.is_empty());
    }

    #[tokio::test]
    async fn caller_with_no_well_known_name_is_rejected() {
        let forwarder = Arc::new(StubForwarder::new(vec![]));
        let sink = Arc::new(CollectingAuditSink::new());
        let svc = service_with(forwarder.clone(), sink.clone());

        let anon = CallerIdentity {
            well_known_bus_name: None,
            unique_bus_name: ":1.9".to_string(),
        };
        let err = svc
            .forward(
                &anon,
                ForwardRequest {
                    provider_name: "ollama-default".to_string(),
                    body_json: "{}".to_string(),
                    audit_token: "tok-3".to_string(),
                },
            )
            .await
            .expect_err("reject");
        assert_eq!(err.code(), "caller-not-allowed");
        assert!(forwarder.calls.lock().await.is_empty());
    }

    #[tokio::test]
    async fn unknown_provider_is_rejected() {
        let forwarder = Arc::new(StubForwarder::new(vec![]));
        let sink = Arc::new(CollectingAuditSink::new());
        let svc = service_with(forwarder.clone(), sink.clone());

        let err = svc
            .forward(
                &ai_daemon_caller(),
                ForwardRequest {
                    provider_name: "imaginary".to_string(),
                    body_json: "{}".to_string(),
                    audit_token: "tok-4".to_string(),
                },
            )
            .await
            .expect_err("reject");
        assert_eq!(err.code(), "unknown-provider");
        assert!(forwarder.calls.lock().await.is_empty());
    }

    #[tokio::test]
    async fn upstream_transport_error_audits_upstream_error() {
        let forwarder = Arc::new(StubForwarder::new(vec![Err(ForwardError::Transport(
            "connection refused".to_string(),
        ))]));
        let sink = Arc::new(CollectingAuditSink::new());
        let svc = service_with(forwarder.clone(), sink.clone());

        let err = svc
            .forward(
                &ai_daemon_caller(),
                ForwardRequest {
                    provider_name: "ollama-default".to_string(),
                    body_json: "{}".to_string(),
                    audit_token: "tok-5".to_string(),
                },
            )
            .await
            .expect_err("fail");
        assert_eq!(err.code(), "upstream-error");
        let records = sink.snapshot().await;
        // Pre-forward gate entry, then the best-effort error entry.
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].structural.outcome, "forwarding");
        assert_eq!(records[1].structural.outcome, "upstream-error");
        assert_eq!(records[1].structural.subject, "localhost");
    }

    #[tokio::test]
    async fn allowed_providers_lists_catalog_entries() {
        let forwarder = Arc::new(StubForwarder::new(vec![]));
        let sink = Arc::new(CollectingAuditSink::new());
        let svc = service_with(forwarder, sink);
        let providers = svc.allowed_providers();
        // The default catalog ships only the local provider.
        assert_eq!(providers, vec!["ollama-default".to_string()]);
    }

    /// Forwarder that parks every call on a notify until released,
    /// so a test can hold a forward in flight deterministically.
    struct GatedForwarder {
        gate: Arc<tokio::sync::Notify>,
    }

    #[async_trait]
    impl Forwarder for GatedForwarder {
        async fn post(
            &self,
            _endpoint_url: &str,
            _body_json: &str,
        ) -> Result<ForwardResult, ForwardError> {
            self.gate.notified().await;
            Ok(ForwardResult {
                status: 200,
                body: "{}".to_string(),
            })
        }
    }

    #[tokio::test]
    async fn concurrent_forwards_past_the_ceiling_are_refused() {
        // A ceiling of one: the first forward parks in the gated
        // forwarder holding the only slot; a second concurrent
        // forward must be refused rather than reaching upstream.
        let gate = Arc::new(tokio::sync::Notify::new());
        let svc = Arc::new(ProxyService::with_max_inflight(
            Allowlist::default_lunaris(),
            ProviderCatalog::default_lunaris(),
            CallerAllowlist::default_lunaris(),
            Arc::new(GatedForwarder { gate: gate.clone() }) as Arc<dyn Forwarder>,
            Arc::new(CollectingAuditSink::new()) as Arc<dyn AuditSink>,
            1,
        ));

        let svc_a = svc.clone();
        let first = tokio::spawn(async move {
            svc_a
                .forward(
                    &ai_daemon_caller(),
                    ForwardRequest {
                        provider_name: "ollama-default".to_string(),
                        body_json: "{}".to_string(),
                        audit_token: "tok-a".to_string(),
                    },
                )
                .await
        });

        // Let the first forward enter the gated post() and hold the
        // slot before the second one runs.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let err = svc
            .forward(
                &ai_daemon_caller(),
                ForwardRequest {
                    provider_name: "ollama-default".to_string(),
                    body_json: "{}".to_string(),
                    audit_token: "tok-b".to_string(),
                },
            )
            .await
            .expect_err("second forward over ceiling");
        assert_eq!(err.code(), "proxy-at-capacity");

        // Release the first forward; once its slot frees, a new
        // forward is admitted again.
        gate.notify_one();
        first.await.unwrap().expect("first forward completes");
        gate.notify_one();
        svc.forward(
            &ai_daemon_caller(),
            ForwardRequest {
                provider_name: "ollama-default".to_string(),
                body_json: "{}".to_string(),
                audit_token: "tok-c".to_string(),
            },
        )
        .await
        .expect("forward admitted after slot freed");
    }

    #[tokio::test]
    async fn forward_is_refused_when_audit_is_unavailable() {
        // The audit ledger is down. The proxy must refuse the forward
        // before any request leaves the host (foundation §8.4.6),
        // even though the caller, provider, and allowlist all pass.
        let forwarder = Arc::new(StubForwarder::new(vec![Ok(ForwardResult {
            status: 200,
            body: "{}".to_string(),
        })]));
        let sink = Arc::new(CollectingAuditSink::failing());
        let svc = ProxyService::new(
            Allowlist::default_lunaris(),
            ProviderCatalog::default_lunaris(),
            CallerAllowlist::default_lunaris(),
            forwarder.clone() as Arc<dyn Forwarder>,
            sink as Arc<dyn AuditSink>,
        );
        let err = svc
            .forward(
                &ai_daemon_caller(),
                ForwardRequest {
                    provider_name: "ollama-default".to_string(),
                    body_json: "{}".to_string(),
                    audit_token: "tok".to_string(),
                },
            )
            .await
            .expect_err("audit-unavailable must refuse the forward");
        assert_eq!(err.code(), "audit-unavailable");
        // The request never left the host: the forwarder was not called.
        assert!(forwarder.calls.lock().await.is_empty());
    }
}
