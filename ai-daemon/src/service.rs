//! Dispatch pipeline + service surface.
//!
//! [`AiDaemonService`] is the in-process API the D-Bus layer wraps.
//! The daemon is poll-based: callers submit a query, get back
//! `(query_id, retrieval_token)`, and poll
//! `take_result(query_id, retrieval_token)` until the outcome is
//! terminal. Polling rather than broadcasting matters: a result
//! signal on the session bus would leak the answer to every
//! listener, not just the caller.
//!
//! The dispatcher runs each prompt through a
//! [`QueryRunner`] (in production the `ai-core` `CypherPipeline`):
//! natural language to a structured graph query, validated and
//! compiled to Cypher by the daemon, executed against the Knowledge
//! Graph, then formatted back to natural language. The pipeline's
//! provider calls transit `ai-proxy` (Foundation §8.4.6).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use lunaris_ai_core::graph_query::QueryScope;
use lunaris_ai_core::pipeline::QueryRunner;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

use crate::registry::{AuthError, CompletionOutcome, CreatedQuery, QueryRegistry, QueryStatus};

/// Per-caller in-flight cap. Matches the modulesd network host's
/// per-module concurrency budget for symmetry.
pub const DEFAULT_MAX_INFLIGHT_PER_CALLER: usize = 4;

/// Daemon-wide in-flight cap. Backstop so the sum of all callers
/// cannot drive unbounded provider work even if many distinct apps
/// each stay under their per-caller cap.
pub const DEFAULT_MAX_INFLIGHT_GLOBAL: usize = 32;

/// Hard ceiling on the prompt size accepted at the D-Bus boundary.
/// 64 KiB comfortably covers chat prompts plus inlined context;
/// documents are summarised by the dispatcher before reaching the
/// provider, so larger inputs are out of scope.
pub const DEFAULT_MAX_PROMPT_BYTES: usize = 64 * 1024;

/// Identity of a query submitter, resolved by the D-Bus layer.
///
/// The two fields serve two distinct security purposes and must not
/// be conflated:
///
/// * `unique_bus_name` is connection-precise. It authorises result
///   retrieval so a sibling connection of the same app cannot poll
///   another connection's query (paired with the retrieval token).
/// * `stable_id` is the caller's executable path. It is the
///   rate-limit key, because a caller could otherwise open many
///   D-Bus connections and multiply its quota.
#[derive(Debug, Clone)]
pub struct CallerIdentity {
    /// Unique D-Bus name of the caller's connection (`:1.42`).
    pub unique_bus_name: String,
    /// Stable per-application identity (executable path).
    pub stable_id: String,
}

/// Errors that `query()` can return synchronously.
#[derive(Debug, thiserror::Error)]
pub enum QueryError {
    /// The daemon is currently disabled.
    #[error("ai disabled")]
    Disabled,
    /// Caller already has [`DEFAULT_MAX_INFLIGHT_PER_CALLER`] queries
    /// running.
    #[error("too many in-flight queries for this caller")]
    TooManyInflight,
    /// The daemon-wide in-flight cap is reached.
    #[error("daemon at global query capacity")]
    GlobalCapacityReached,
    /// Prompt exceeds [`DEFAULT_MAX_PROMPT_BYTES`].
    #[error("prompt too large: {0} bytes")]
    PromptTooLarge(usize),
    /// The daemon's capability scope permits no graph access, so no
    /// query can succeed. Rejected synchronously before any provider
    /// call so an impossible query never burns an LLM round-trip.
    #[error("ai layer has no graph access configured")]
    NoGraphAccess,
}

impl QueryError {
    /// Stable kebab-case error code.
    pub fn code(&self) -> &'static str {
        match self {
            QueryError::Disabled => "ai-disabled",
            QueryError::TooManyInflight => "too-many-inflight",
            QueryError::GlobalCapacityReached => "global-capacity-reached",
            QueryError::PromptTooLarge(_) => "prompt-too-large",
            QueryError::NoGraphAccess => "no-graph-access",
        }
    }
}

/// Handle returned to the D-Bus surface from `query()`.
#[derive(Debug, Clone)]
pub struct QueryHandle {
    /// Stable query identifier.
    pub query_id: String,
    /// One-shot retrieval token. The caller must store this and pass
    /// it on every follow-up method.
    pub retrieval_token: String,
}

/// Outcome of an [`InflightTracker::try_acquire`] attempt.
#[derive(Debug, PartialEq, Eq)]
enum AcquireResult {
    /// A slot was reserved; the caller must `release` it later.
    Acquired,
    /// The per-caller cap is reached.
    CallerFull,
    /// The daemon-wide cap is reached.
    GlobalFull,
}

/// Concurrency tracker. Keyed on the caller's stable executable
/// identity so extra D-Bus connections cannot multiply a caller's
/// quota, plus a daemon-wide counter as a backstop.
#[derive(Debug, Default)]
struct InflightTracker {
    by_stable_id: HashMap<String, usize>,
    global: usize,
}

impl InflightTracker {
    fn try_acquire(
        &mut self,
        stable_id: &str,
        per_caller_cap: usize,
        global_cap: usize,
    ) -> AcquireResult {
        if self.global >= global_cap {
            return AcquireResult::GlobalFull;
        }
        let entry = self.by_stable_id.entry(stable_id.to_string()).or_insert(0);
        if *entry >= per_caller_cap {
            return AcquireResult::CallerFull;
        }
        *entry += 1;
        self.global += 1;
        AcquireResult::Acquired
    }

    fn release(&mut self, stable_id: &str) {
        if let Some(count) = self.by_stable_id.get_mut(stable_id) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                self.by_stable_id.remove(stable_id);
            }
        }
        self.global = self.global.saturating_sub(1);
    }
}

/// Daemon service.
#[derive(Clone)]
pub struct AiDaemonService {
    registry: QueryRegistry,
    runner: Arc<dyn QueryRunner>,
    /// Capability scope applied to every query. Injected explicitly;
    /// there is deliberately no implicit full-access default.
    /// Phase 9-α uses a single daemon-wide scope; Phase 9-γ S16
    /// derives it per-caller from the 5 read tiers.
    scope: QueryScope,
    enabled: Arc<std::sync::atomic::AtomicBool>,
    inflight: Arc<Mutex<InflightTracker>>,
    max_inflight_per_caller: usize,
    max_inflight_global: usize,
    max_prompt_bytes: usize,
}

impl AiDaemonService {
    /// Build a service over a query runner with the default limits.
    /// The capability `scope` is supplied by the caller; there is no
    /// implicit full-access default.
    pub fn new(runner: Arc<dyn QueryRunner>, scope: QueryScope) -> Self {
        Self::with_limits(
            runner,
            scope,
            DEFAULT_MAX_INFLIGHT_PER_CALLER,
            DEFAULT_MAX_INFLIGHT_GLOBAL,
            DEFAULT_MAX_PROMPT_BYTES,
        )
    }

    /// Build with explicit limits. Tests use this constructor to
    /// exercise the cap paths without pumping out dozens of queries.
    pub fn with_limits(
        runner: Arc<dyn QueryRunner>,
        scope: QueryScope,
        max_inflight_per_caller: usize,
        max_inflight_global: usize,
        max_prompt_bytes: usize,
    ) -> Self {
        Self {
            registry: QueryRegistry::new(),
            runner,
            scope,
            // Fail closed: the daemon starts disabled and accepts no
            // queries until Settings explicitly enables the AI layer.
            // The AI layer is opt-in per Foundation §5.1-5.2; a
            // freshly started daemon must not serve graph reads to
            // session-bus callers on its own.
            enabled: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            inflight: Arc::new(Mutex::new(InflightTracker::default())),
            max_inflight_per_caller,
            max_inflight_global,
            max_prompt_bytes,
        }
    }

    /// Spawn the periodic sweep task. Drops terminated records older
    /// than the registry's retention window. The caller owns the
    /// returned [`tokio::task::JoinHandle`] and can abort it on
    /// shutdown.
    pub fn spawn_sweep_task(&self) -> tokio::task::JoinHandle<()> {
        let registry = self.registry.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60));
            loop {
                interval.tick().await;
                registry.sweep().await;
            }
        })
    }

    /// Borrow the registry. Exposed for diagnostics and tests.
    pub fn registry(&self) -> &QueryRegistry {
        &self.registry
    }

    /// Toggle the daemon's accept state. The D-Bus surface does not
    /// expose a writer for this property; only `Settings`, via the
    /// TOML config watcher (Phase 9-α S7), should change it.
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled
            .store(enabled, std::sync::atomic::Ordering::SeqCst);
    }

    /// Whether the daemon is currently accepting new queries.
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Submit a query for dispatch. Returns the query handle (id +
    /// retrieval token) immediately; the result is consumable through
    /// [`take_result`](Self::take_result) once the dispatcher finishes.
    ///
    /// Bounded by [`Self::max_prompt_bytes`] (input cap) and by the
    /// per-caller in-flight count (concurrent dispatch cap). Both
    /// limits surface as typed [`QueryError`] variants so the D-Bus
    /// surface can map them to stable error codes.
    pub async fn query(
        &self,
        prompt: String,
        caller: CallerIdentity,
    ) -> Result<QueryHandle, QueryError> {
        if !self.is_enabled() {
            return Err(QueryError::Disabled);
        }
        // An empty scope (Minimal tier) cannot satisfy any query.
        // Reject here so the pipeline never runs and no provider
        // call is spent on a query that would always fail validation.
        if self.scope.is_empty() {
            return Err(QueryError::NoGraphAccess);
        }
        if prompt.len() > self.max_prompt_bytes {
            return Err(QueryError::PromptTooLarge(prompt.len()));
        }
        match self.inflight.lock().await.try_acquire(
            &caller.stable_id,
            self.max_inflight_per_caller,
            self.max_inflight_global,
        ) {
            AcquireResult::Acquired => {}
            AcquireResult::CallerFull => return Err(QueryError::TooManyInflight),
            AcquireResult::GlobalFull => return Err(QueryError::GlobalCapacityReached),
        }
        let CreatedQuery {
            query_id,
            retrieval_token,
            cancel,
        } = self
            .registry
            .create(caller.unique_bus_name.clone())
            .await;

        let svc = self.clone();
        let qid = query_id.clone();
        let stable_id = caller.stable_id.clone();
        tokio::spawn(async move {
            svc.dispatch(qid, prompt, cancel).await;
            svc.inflight.lock().await.release(&stable_id);
        });

        Ok(QueryHandle {
            query_id,
            retrieval_token,
        })
    }

    /// Authorised cancel.
    pub async fn cancel(
        &self,
        query_id: &str,
        caller_unique_bus_name: &str,
        retrieval_token: &str,
    ) -> Result<bool, AuthError> {
        self.registry
            .cancel(query_id, caller_unique_bus_name, retrieval_token)
            .await
    }

    /// Authorised status snapshot.
    pub async fn status(
        &self,
        query_id: &str,
        caller_unique_bus_name: &str,
        retrieval_token: &str,
    ) -> Result<QueryStatus, AuthError> {
        self.registry
            .status_authorised(query_id, caller_unique_bus_name, retrieval_token)
            .await
    }

    /// Authorised result retrieval. Single-shot for `Completed`
    /// outcomes.
    pub async fn take_result(
        &self,
        query_id: &str,
        caller_unique_bus_name: &str,
        retrieval_token: &str,
    ) -> Result<CompletionOutcome, AuthError> {
        self.registry
            .take_result(query_id, caller_unique_bus_name, retrieval_token)
            .await
    }

    async fn dispatch(&self, query_id: String, prompt: String, cancel: CancellationToken) {
        self.registry.mark_in_progress(&query_id).await;

        let runner_call = self.runner.run_query(&prompt, &self.scope);

        let result = tokio::select! {
            biased;
            _ = cancel.cancelled() => {
                // Cancellation already updates status in the registry
                // via `cancel()`. Nothing further to do.
                return;
            }
            res = runner_call => res,
        };

        match result {
            Ok(answer) => {
                self.registry.mark_completed(&query_id, answer).await;
            }
            Err(failure) => {
                self.registry
                    .mark_failed(&query_id, &failure.code, &failure.reason)
                    .await;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use lunaris_ai_core::pipeline::RunFailure;
    use std::time::Duration;
    use tokio::sync::Notify;

    /// Query-runner stub. `gate`, when set, parks the call until
    /// notified so tests can hold queries in flight deterministically.
    struct StubRunner {
        reply: Result<String, RunFailure>,
        gate: Option<Arc<Notify>>,
    }

    #[async_trait]
    impl QueryRunner for StubRunner {
        async fn run_query(
            &self,
            _prompt: &str,
            _scope: &QueryScope,
        ) -> Result<String, RunFailure> {
            if let Some(gate) = &self.gate {
                gate.notified().await;
            }
            self.reply.clone()
        }
    }

    /// Poll the daemon for an outcome that is not `Pending` / `InProgress`.
    async fn wait_for_terminal(
        svc: &AiDaemonService,
        h: &QueryHandle,
        caller: &str,
    ) -> CompletionOutcome {
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        loop {
            let outcome = svc
                .take_result(&h.query_id, caller, &h.retrieval_token)
                .await
                .expect("authz");
            match &outcome {
                CompletionOutcome::Pending | CompletionOutcome::InProgress => {
                    if std::time::Instant::now() > deadline {
                        panic!("timed out waiting for terminal outcome");
                    }
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
                _ => return outcome,
            }
        }
    }

    /// Build a [`CallerIdentity`] for tests.
    fn caller_id(unique: &str, stable: &str) -> CallerIdentity {
        CallerIdentity {
            unique_bus_name: unique.to_string(),
            stable_id: stable.to_string(),
        }
    }

    /// Full-access scope for service tests. These tests exercise
    /// dispatch / caps / registry behaviour; scope enforcement is
    /// covered in `ai-core`'s `graph_query` tests.
    fn full_scope() -> lunaris_ai_core::graph_query::QueryScope {
        lunaris_ai_core::graph_query::QueryScope::full(
            &lunaris_ai_core::graph_schema::GraphSchema::knowledge_graph(),
        )
    }

    /// Enable a freshly built service. The daemon is fail-closed by
    /// default; tests that exercise the query path flip it on
    /// explicitly, just as Settings does in production.
    fn enable(svc: AiDaemonService) -> AiDaemonService {
        svc.set_enabled(true);
        svc
    }

    #[tokio::test]
    async fn happy_path_returns_completed_result() {
        let svc = enable(AiDaemonService::new(Arc::new(StubRunner {
            reply: Ok("hello".to_string()),
            gate: None,
        }), full_scope()));
        let h = svc
            .query("hi".to_string(), caller_id(":1.42", "/usr/bin/app-a"))
            .await
            .unwrap();
        let outcome = wait_for_terminal(&svc, &h, ":1.42").await;
        match outcome {
            CompletionOutcome::Completed { result } => assert_eq!(result, "hello"),
            other => panic!("expected completed, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn other_caller_cannot_read_result() {
        let svc = enable(AiDaemonService::new(Arc::new(StubRunner {
            reply: Ok("secret".to_string()),
            gate: None,
        }), full_scope()));
        let h = svc
            .query("hi".to_string(), caller_id(":1.42", "/usr/bin/app-a"))
            .await
            .unwrap();
        // Original connection drains it first to make the test
        // deterministic, then a different connection tries.
        let _ = wait_for_terminal(&svc, &h, ":1.42").await;
        let err = svc
            .take_result(&h.query_id, ":1.99", &h.retrieval_token)
            .await
            .expect_err("must reject");
        assert_eq!(err, AuthError::CallerMismatch);
    }

    #[tokio::test]
    async fn wrong_token_is_rejected_even_for_correct_caller() {
        let svc = enable(AiDaemonService::new(Arc::new(StubRunner {
            reply: Ok("secret".to_string()),
            gate: None,
        }), full_scope()));
        let h = svc
            .query("hi".to_string(), caller_id(":1.42", "/usr/bin/app-a"))
            .await
            .unwrap();
        let err = svc
            .take_result(&h.query_id, ":1.42", "deadbeef")
            .await
            .expect_err("must reject");
        assert_eq!(err, AuthError::TokenMismatch);
    }

    #[tokio::test]
    async fn cancel_aborts_dispatch_and_reports_cancelled() {
        let gate = Arc::new(Notify::new());
        let svc = enable(AiDaemonService::new(Arc::new(StubRunner {
            reply: Ok("never".to_string()),
            gate: Some(gate.clone()),
        }), full_scope()));
        let h = svc
            .query("hi".to_string(), caller_id(":1.42", "/usr/bin/app-a"))
            .await
            .unwrap();
        assert!(svc
            .cancel(&h.query_id, ":1.42", &h.retrieval_token)
            .await
            .unwrap());
        // Release the gate so the dispatch task drops out promptly.
        gate.notify_one();
        // Status should already be Cancelled.
        let outcome = svc
            .take_result(&h.query_id, ":1.42", &h.retrieval_token)
            .await
            .unwrap();
        assert!(matches!(outcome, CompletionOutcome::Cancelled));
    }

    #[tokio::test]
    async fn empty_scope_rejects_before_running_the_pipeline() {
        // An enabled daemon with the Minimal (empty) scope must
        // reject synchronously, not burn a provider call on a query
        // that would always fail.
        use lunaris_ai_core::graph_query::{AccessTier, QueryScope};
        use lunaris_ai_core::graph_schema::GraphSchema;

        let minimal = QueryScope::for_tier(
            AccessTier::Minimal,
            &GraphSchema::knowledge_graph(),
        );
        let svc = enable(AiDaemonService::new(
            Arc::new(StubRunner {
                reply: Ok("never reached".to_string()),
                gate: None,
            }),
            minimal,
        ));
        let err = svc
            .query("hi".to_string(), caller_id(":1.42", "/usr/bin/app-a"))
            .await
            .expect_err("empty scope rejects");
        assert_eq!(err.code(), "no-graph-access");
    }

    #[tokio::test]
    async fn fresh_daemon_is_disabled_and_rejects_queries() {
        // A freshly built service is fail-closed: it serves no
        // queries until Settings enables the AI layer.
        let runner = Arc::new(StubRunner {
            reply: Ok("never".to_string()),
            gate: None,
        });
        let svc = AiDaemonService::new(runner, full_scope());
        // No enable() call: the default must be disabled.
        let err = svc
            .query("hi".to_string(), caller_id(":1.42", "/usr/bin/app-a"))
            .await
            .expect_err("fresh daemon rejects");
        assert_eq!(err.code(), "ai-disabled");
    }

    #[tokio::test]
    async fn disabled_service_rejects_synchronously() {
        let svc = enable(AiDaemonService::new(Arc::new(StubRunner {
            reply: Ok("never".to_string()),
            gate: None,
        }), full_scope()));
        svc.set_enabled(false);
        let err = svc
            .query("hi".to_string(), caller_id(":1.42", "/usr/bin/app-a"))
            .await
            .expect_err("rejected");
        assert_eq!(err.code(), "ai-disabled");
    }

    #[tokio::test]
    async fn prompt_size_cap_is_enforced() {
        let svc = enable(AiDaemonService::with_limits(
            Arc::new(StubRunner {
                reply: Ok("ok".to_string()),
                gate: None,
            }),
            full_scope(),
            4,
            DEFAULT_MAX_INFLIGHT_GLOBAL,
            16,
        ));
        let prompt = "x".repeat(17);
        let err = svc
            .query(prompt, caller_id(":1.42", "/usr/bin/app-a"))
            .await
            .expect_err("rejected");
        match err {
            QueryError::PromptTooLarge(n) => assert_eq!(n, 17),
            other => panic!("expected PromptTooLarge, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn per_caller_cap_is_keyed_on_stable_id_not_connection() {
        // A caller that opens several D-Bus connections (distinct
        // unique names) but runs the same executable must NOT
        // multiply its quota.
        let gate = Arc::new(Notify::new());
        let svc = enable(AiDaemonService::with_limits(
            Arc::new(StubRunner {
                reply: Ok("ok".to_string()),
                gate: Some(gate.clone()),
            }),
            full_scope(),
            2,
            DEFAULT_MAX_INFLIGHT_GLOBAL,
            DEFAULT_MAX_PROMPT_BYTES,
        ));
        let stable = "/usr/bin/app-a";
        // Two queries from two *different* connections of the same app.
        let _a = svc
            .query("hi".to_string(), caller_id(":1.10", stable))
            .await
            .unwrap();
        let _b = svc
            .query("hi".to_string(), caller_id(":1.11", stable))
            .await
            .unwrap();
        // A third connection of the same app is over the per-caller cap.
        let err = svc
            .query("hi".to_string(), caller_id(":1.12", stable))
            .await
            .expect_err("over cap");
        assert!(matches!(err, QueryError::TooManyInflight));
        // Release both, then the cap admits a new query again.
        gate.notify_one();
        gate.notify_one();
        for _ in 0..50 {
            if svc
                .inflight
                .lock()
                .await
                .by_stable_id
                .get(stable)
                .copied()
                .unwrap_or(0)
                == 0
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        let _c = svc
            .query("hi".to_string(), caller_id(":1.13", stable))
            .await
            .expect("ok after release");
    }

    #[tokio::test]
    async fn global_cap_backstops_many_distinct_callers() {
        // Even when every caller stays under its own per-caller cap,
        // the daemon-wide cap must bound total in-flight work.
        let gate = Arc::new(Notify::new());
        let svc = enable(AiDaemonService::with_limits(
            Arc::new(StubRunner {
                reply: Ok("ok".to_string()),
                gate: Some(gate.clone()),
            }),
            full_scope(),
            4,
            2, // tiny global cap
            DEFAULT_MAX_PROMPT_BYTES,
        ));
        let _a = svc
            .query("hi".to_string(), caller_id(":1.1", "/usr/bin/app-a"))
            .await
            .unwrap();
        let _b = svc
            .query("hi".to_string(), caller_id(":1.2", "/usr/bin/app-b"))
            .await
            .unwrap();
        // Third distinct caller, still under its per-caller cap, but
        // the global cap is full.
        let err = svc
            .query("hi".to_string(), caller_id(":1.3", "/usr/bin/app-c"))
            .await
            .expect_err("global full");
        assert!(matches!(err, QueryError::GlobalCapacityReached));
        assert_eq!(err.code(), "global-capacity-reached");
        gate.notify_one();
        gate.notify_one();
    }

    #[tokio::test]
    async fn runner_failure_surfaces_as_failed_outcome() {
        let svc = enable(AiDaemonService::new(Arc::new(StubRunner {
            reply: Err(RunFailure {
                code: "graph-error".to_string(),
                reason: "knowledge graph unreachable".to_string(),
            }),
            gate: None,
        }), full_scope()));
        let h = svc
            .query("hi".to_string(), caller_id(":1.42", "/usr/bin/app-a"))
            .await
            .unwrap();
        let outcome = wait_for_terminal(&svc, &h, ":1.42").await;
        match outcome {
            CompletionOutcome::Failed { code, .. } => assert_eq!(code, "graph-error"),
            other => panic!("expected failed, got {other:?}"),
        }
    }
}
