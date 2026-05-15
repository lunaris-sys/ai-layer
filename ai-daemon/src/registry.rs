//! In-memory query registry.
//!
//! Tracks every in-flight query so the daemon can:
//!
//! * answer `cancel(query_id)` from the D-Bus surface,
//! * verify that the caller invoking `get_result` / `get_status` /
//!   `cancel` is the same caller that submitted the query,
//! * present a status to Settings on demand,
//! * enforce per-identity rate limits in Phase 9-γ S15.
//!
//! ## Authorisation model
//!
//! Each created record carries two independently-checked guards:
//!
//! 1. The submitting D-Bus caller (unique bus name). Captured at
//!    `create()` time. The D-Bus surface compares this against
//!    `Header::sender()` on every follow-up call.
//! 2. A per-query *retrieval token*: 256 bits of CSPRNG, returned to
//!    the caller from `query()` and required as an argument on every
//!    follow-up method. The registry stores a SHA-256 hash of the
//!    token, never the plaintext. This means a process that observed
//!    the audit log or memory dump cannot replay tokens.
//!
//! Both must match for sensitive operations
//! ([`get_result`](QueryRegistry::take_result),
//! [`cancel`](QueryRegistry::cancel),
//! [`status`](QueryRegistry::status_authorised)).
//!
//! ## Lifecycle retention
//!
//! Once a query terminates the entry stays for a short retention
//! window so late `cancel` / `get_result` calls do not race against
//! cleanup. The result text is consumed on `take_result` so the
//! daemon does not keep it pinned in memory beyond the caller's
//! single-shot retrieval.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use rand::RngCore;
use sha2::{Digest, Sha256};
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

/// Retention for terminated records that still hold an unretrieved
/// result (`Completed`). Generous so a slow or suspended caller that
/// polls infrequently does not lose an expensive answer.
const DEFAULT_COMPLETED_RETENTION: Duration = Duration::from_secs(3600);

/// Retention for terminated records whose payload is already gone
/// (`Drained`, `Failed`, `Cancelled`). Short, just long enough to
/// absorb a late follow-up call racing against cleanup.
const DEFAULT_DRAINED_RETENTION: Duration = Duration::from_secs(30);

/// Lifecycle status of a tracked query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryStatus {
    /// The query has been accepted and queued.
    Pending,
    /// The dispatch pipeline is actively running.
    InProgress,
    /// The query finished successfully. The result is consumable
    /// exactly once through `take_result`.
    Completed,
    /// The query has been completed and the result already
    /// retrieved.
    Drained,
    /// The query failed.
    Failed {
        /// Stable error code matching
        /// [`crate::service::QueryError::code`].
        code: String,
        /// Human-readable detail.
        reason: String,
    },
    /// The query was cancelled by the caller.
    Cancelled,
}

/// Errors returned by authorised registry operations.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum AuthError {
    /// No record exists for the supplied query id.
    #[error("unknown query")]
    UnknownQuery,
    /// The caller does not match the original submitter.
    #[error("caller does not match query submitter")]
    CallerMismatch,
    /// The retrieval token does not match the stored hash.
    #[error("retrieval token mismatch")]
    TokenMismatch,
}

/// Result-payload outcome surfaced by `take_result`.
#[derive(Debug, Clone)]
pub enum CompletionOutcome {
    /// The query is still in flight.
    Pending,
    /// The query is in flight but in progress.
    InProgress,
    /// The query succeeded. The result text is consumed by the
    /// caller; subsequent calls return `Drained`.
    Completed {
        /// Final result text.
        result: String,
    },
    /// The query already had its result consumed.
    Drained,
    /// The query failed.
    Failed {
        /// Stable error code.
        code: String,
        /// Human-readable detail.
        reason: String,
    },
    /// The query was cancelled.
    Cancelled,
}

#[derive(Debug, Clone)]
struct QueryRecord {
    status: QueryStatus,
    cancel: CancellationToken,
    submitter_unique_bus_name: String,
    token_hash: [u8; 32],
    result: Option<String>,
    terminated_at: Option<Instant>,
}

/// Thread-safe registry.
#[derive(Debug, Clone)]
pub struct QueryRegistry {
    inner: Arc<Mutex<HashMap<String, QueryRecord>>>,
    /// Retention for `Completed` records with an unretrieved result.
    completed_retention: Duration,
    /// Retention for `Drained` / `Failed` / `Cancelled` records.
    drained_retention: Duration,
}

impl Default for QueryRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Bundle returned by [`QueryRegistry::create`].
#[derive(Debug, Clone)]
pub struct CreatedQuery {
    /// Stable identifier the caller uses on subsequent calls.
    pub query_id: String,
    /// Retrieval token plaintext. The caller must store this; the
    /// registry only keeps its SHA-256 hash.
    pub retrieval_token: String,
    /// Cancellation token observed by the dispatch task.
    pub cancel: CancellationToken,
}

impl QueryRegistry {
    /// Build an empty registry with the default retention windows.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(HashMap::new())),
            completed_retention: DEFAULT_COMPLETED_RETENTION,
            drained_retention: DEFAULT_DRAINED_RETENTION,
        }
    }

    /// Build with explicit retention windows. Tests use short
    /// windows to exercise the sweep without real-time waits.
    pub fn with_retention(completed: Duration, drained: Duration) -> Self {
        Self {
            inner: Arc::new(Mutex::new(HashMap::new())),
            completed_retention: completed,
            drained_retention: drained,
        }
    }

    /// Create a new record. `submitter_unique_bus_name` is the
    /// unique D-Bus name (`:1.42`) of the caller; the daemon's D-Bus
    /// surface fills it in from the message header. The returned
    /// `retrieval_token` is sent back to the caller and required on
    /// every follow-up method.
    pub async fn create(&self, submitter_unique_bus_name: String) -> CreatedQuery {
        let query_id = uuid::Uuid::new_v4().to_string();
        let mut token_bytes = [0u8; 32];
        rand::thread_rng().fill_bytes(&mut token_bytes);
        let retrieval_token = hex::encode(token_bytes);
        let token_hash = hash_token(&retrieval_token);
        let cancel = CancellationToken::new();
        let record = QueryRecord {
            status: QueryStatus::Pending,
            cancel: cancel.clone(),
            submitter_unique_bus_name,
            token_hash,
            result: None,
            terminated_at: None,
        };
        self.inner.lock().await.insert(query_id.clone(), record);
        CreatedQuery {
            query_id,
            retrieval_token,
            cancel,
        }
    }

    /// Promote a pending record to `InProgress`. No-op if the record
    /// no longer exists.
    pub async fn mark_in_progress(&self, query_id: &str) {
        if let Some(rec) = self.inner.lock().await.get_mut(query_id) {
            if matches!(rec.status, QueryStatus::Pending) {
                rec.status = QueryStatus::InProgress;
            }
        }
    }

    /// Store the successful result. The transition fires only if
    /// the record is still in the in-flight state set
    /// (`Pending` or `InProgress`). If the record is already
    /// terminal, most importantly `Cancelled`, the result is dropped
    /// and the cancellation stays final. This closes the race where
    /// a provider call resolves after a cancel has already won.
    pub async fn mark_completed(&self, query_id: &str, result: String) {
        if let Some(rec) = self.inner.lock().await.get_mut(query_id) {
            if !is_in_flight(&rec.status) {
                return;
            }
            rec.status = QueryStatus::Completed;
            rec.result = Some(result);
            rec.terminated_at = Some(Instant::now());
        }
    }

    /// Mark the record as failed with a stable error code. Same
    /// terminal-status guard as [`mark_completed`].
    pub async fn mark_failed(&self, query_id: &str, code: &str, reason: &str) {
        if let Some(rec) = self.inner.lock().await.get_mut(query_id) {
            if !is_in_flight(&rec.status) {
                return;
            }
            rec.status = QueryStatus::Failed {
                code: code.to_string(),
                reason: reason.to_string(),
            };
            rec.terminated_at = Some(Instant::now());
        }
    }

    /// Authorised cancel. Validates caller + token, then signals
    /// cancellation. Returns `true` if the record existed, was not
    /// already terminated, and the caller passed authz.
    pub async fn cancel(
        &self,
        query_id: &str,
        caller_unique_bus_name: &str,
        retrieval_token: &str,
    ) -> Result<bool, AuthError> {
        let mut guard = self.inner.lock().await;
        let Some(rec) = guard.get_mut(query_id) else {
            return Err(AuthError::UnknownQuery);
        };
        check_authz(rec, caller_unique_bus_name, retrieval_token)?;
        if matches!(
            rec.status,
            QueryStatus::Completed
                | QueryStatus::Drained
                | QueryStatus::Failed { .. }
                | QueryStatus::Cancelled
        ) {
            return Ok(false);
        }
        rec.cancel.cancel();
        rec.status = QueryStatus::Cancelled;
        rec.terminated_at = Some(Instant::now());
        Ok(true)
    }

    /// Authorised status snapshot.
    pub async fn status_authorised(
        &self,
        query_id: &str,
        caller_unique_bus_name: &str,
        retrieval_token: &str,
    ) -> Result<QueryStatus, AuthError> {
        let guard = self.inner.lock().await;
        let rec = guard.get(query_id).ok_or(AuthError::UnknownQuery)?;
        check_authz(rec, caller_unique_bus_name, retrieval_token)?;
        Ok(rec.status.clone())
    }

    /// Authorised result retrieval. Consumes the stored result on
    /// `Completed`, returns `Drained` afterwards. The caller can
    /// invoke this repeatedly to poll for completion without leaking
    /// the result text to anyone other than the original submitter.
    pub async fn take_result(
        &self,
        query_id: &str,
        caller_unique_bus_name: &str,
        retrieval_token: &str,
    ) -> Result<CompletionOutcome, AuthError> {
        let mut guard = self.inner.lock().await;
        let rec = guard.get_mut(query_id).ok_or(AuthError::UnknownQuery)?;
        check_authz(rec, caller_unique_bus_name, retrieval_token)?;
        let outcome = match rec.status.clone() {
            QueryStatus::Pending => CompletionOutcome::Pending,
            QueryStatus::InProgress => CompletionOutcome::InProgress,
            QueryStatus::Completed => {
                let result = rec.result.take().unwrap_or_default();
                rec.status = QueryStatus::Drained;
                CompletionOutcome::Completed { result }
            }
            QueryStatus::Drained => CompletionOutcome::Drained,
            QueryStatus::Failed { code, reason } => CompletionOutcome::Failed { code, reason },
            QueryStatus::Cancelled => CompletionOutcome::Cancelled,
        };
        Ok(outcome)
    }

    /// Drop terminated records once their retention window expires.
    ///
    /// `Completed` records still hold an unretrieved result and use
    /// the long [`completed_retention`](Self::completed_retention)
    /// window so a slow poller does not lose its answer. Every other
    /// terminal status has no payload left and uses the short
    /// [`drained_retention`](Self::drained_retention) window.
    /// In-flight records (`terminated_at == None`) are never swept.
    pub async fn sweep(&self) {
        let now = Instant::now();
        self.inner.lock().await.retain(|_, rec| {
            let Some(terminated) = rec.terminated_at else {
                return true;
            };
            let retention = match rec.status {
                QueryStatus::Completed => self.completed_retention,
                _ => self.drained_retention,
            };
            now.saturating_duration_since(terminated) < retention
        });
    }

    /// Number of records currently tracked.
    pub async fn len(&self) -> usize {
        self.inner.lock().await.len()
    }
}

fn is_in_flight(status: &QueryStatus) -> bool {
    matches!(status, QueryStatus::Pending | QueryStatus::InProgress)
}

fn hash_token(token: &str) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(token.as_bytes());
    let digest = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&digest);
    out
}

fn check_authz(
    rec: &QueryRecord,
    caller_unique_bus_name: &str,
    retrieval_token: &str,
) -> Result<(), AuthError> {
    if rec.submitter_unique_bus_name != caller_unique_bus_name {
        return Err(AuthError::CallerMismatch);
    }
    let supplied = hash_token(retrieval_token);
    // Constant-time comparison to keep the token guard out of reach
    // of timing oracles.
    let mut diff: u8 = 0;
    for (a, b) in rec.token_hash.iter().zip(supplied.iter()) {
        diff |= a ^ b;
    }
    if diff != 0 {
        return Err(AuthError::TokenMismatch);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn caller() -> &'static str {
        ":1.42"
    }

    #[tokio::test]
    async fn create_yields_unique_ids_and_tokens() {
        let reg = QueryRegistry::new();
        let a = reg.create(caller().to_string()).await;
        let b = reg.create(caller().to_string()).await;
        assert_ne!(a.query_id, b.query_id);
        assert_ne!(a.retrieval_token, b.retrieval_token);
        // 256-bit hex.
        assert_eq!(a.retrieval_token.len(), 64);
        assert!(a.retrieval_token.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[tokio::test]
    async fn take_result_returns_completed_then_drained() {
        let reg = QueryRegistry::new();
        let c = reg.create(caller().to_string()).await;
        reg.mark_in_progress(&c.query_id).await;
        reg.mark_completed(&c.query_id, "42".to_string()).await;
        let first = reg
            .take_result(&c.query_id, caller(), &c.retrieval_token)
            .await
            .unwrap();
        assert!(matches!(first, CompletionOutcome::Completed { ref result } if result == "42"));
        let second = reg
            .take_result(&c.query_id, caller(), &c.retrieval_token)
            .await
            .unwrap();
        assert!(matches!(second, CompletionOutcome::Drained));
    }

    #[tokio::test]
    async fn other_caller_cannot_retrieve_result() {
        let reg = QueryRegistry::new();
        let c = reg.create(caller().to_string()).await;
        reg.mark_completed(&c.query_id, "secret".to_string()).await;
        let err = reg
            .take_result(&c.query_id, ":1.99", &c.retrieval_token)
            .await
            .expect_err("must reject");
        assert_eq!(err, AuthError::CallerMismatch);
    }

    #[tokio::test]
    async fn wrong_token_is_rejected_even_for_correct_caller() {
        let reg = QueryRegistry::new();
        let c = reg.create(caller().to_string()).await;
        reg.mark_completed(&c.query_id, "secret".to_string()).await;
        let err = reg
            .take_result(&c.query_id, caller(), "deadbeef")
            .await
            .expect_err("must reject");
        assert_eq!(err, AuthError::TokenMismatch);
    }

    #[tokio::test]
    async fn cancel_requires_correct_caller_and_token() {
        let reg = QueryRegistry::new();
        let c = reg.create(caller().to_string()).await;
        let err = reg
            .cancel(&c.query_id, ":1.99", &c.retrieval_token)
            .await
            .expect_err("wrong caller");
        assert_eq!(err, AuthError::CallerMismatch);
        let err = reg
            .cancel(&c.query_id, caller(), "deadbeef")
            .await
            .expect_err("wrong token");
        assert_eq!(err, AuthError::TokenMismatch);
        assert!(reg
            .cancel(&c.query_id, caller(), &c.retrieval_token)
            .await
            .unwrap());
        assert!(c.cancel.is_cancelled());
    }

    #[tokio::test]
    async fn unknown_query_yields_unknown_error() {
        let reg = QueryRegistry::new();
        let err = reg
            .take_result("missing", caller(), "deadbeef")
            .await
            .expect_err("unknown");
        assert_eq!(err, AuthError::UnknownQuery);
    }

    #[tokio::test]
    async fn failed_status_is_visible_through_outcome() {
        let reg = QueryRegistry::new();
        let c = reg.create(caller().to_string()).await;
        reg.mark_failed(&c.query_id, "provider-timeout", "timed out").await;
        let outcome = reg
            .take_result(&c.query_id, caller(), &c.retrieval_token)
            .await
            .unwrap();
        match outcome {
            CompletionOutcome::Failed { code, reason } => {
                assert_eq!(code, "provider-timeout");
                assert_eq!(reason, "timed out");
            }
            other => panic!("expected failure, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn cancel_then_complete_keeps_cancelled_state() {
        // Simulate the race where the dispatcher resolves
        // successfully after a cancel has already won. The complete
        // must NOT overwrite the cancellation; the
        // caller was already told the query was cancelled and the
        // result must stay unreachable.
        let reg = QueryRegistry::new();
        let c = reg.create(caller().to_string()).await;
        // Cancellation wins first.
        assert!(reg
            .cancel(&c.query_id, caller(), &c.retrieval_token)
            .await
            .unwrap());
        // Provider future resolves after cancel.
        reg.mark_completed(&c.query_id, "leaked".to_string()).await;
        // Status still Cancelled, result not retrievable.
        let outcome = reg
            .take_result(&c.query_id, caller(), &c.retrieval_token)
            .await
            .unwrap();
        assert!(matches!(outcome, CompletionOutcome::Cancelled));
    }

    #[tokio::test]
    async fn cancel_then_fail_keeps_cancelled_state() {
        let reg = QueryRegistry::new();
        let c = reg.create(caller().to_string()).await;
        assert!(reg
            .cancel(&c.query_id, caller(), &c.retrieval_token)
            .await
            .unwrap());
        // Provider future resolves with error after cancel.
        reg.mark_failed(&c.query_id, "provider-timeout", "late").await;
        let outcome = reg
            .take_result(&c.query_id, caller(), &c.retrieval_token)
            .await
            .unwrap();
        assert!(matches!(outcome, CompletionOutcome::Cancelled));
    }

    #[tokio::test]
    async fn sweep_keeps_recent_terminated_records() {
        let reg = QueryRegistry::new();
        let c = reg.create(caller().to_string()).await;
        reg.mark_completed(&c.query_id, "x".to_string()).await;
        reg.sweep().await;
        assert_eq!(reg.len().await, 1);
    }

    #[tokio::test]
    async fn sweep_keeps_completed_but_drops_drained() {
        // A Completed record holding an unretrieved result must
        // survive even when the drained window is zero, because the
        // caller has not had a chance to poll yet.
        let reg = QueryRegistry::with_retention(Duration::from_secs(3600), Duration::ZERO);
        let c = reg.create(caller().to_string()).await;
        reg.mark_completed(&c.query_id, "expensive answer".to_string()).await;
        reg.sweep().await;
        assert_eq!(reg.len().await, 1, "completed result must not be swept");
        // Caller finally drains it; now it is eligible for the short
        // window and the next sweep removes it.
        let _ = reg
            .take_result(&c.query_id, caller(), &c.retrieval_token)
            .await
            .unwrap();
        reg.sweep().await;
        assert_eq!(reg.len().await, 0, "drained record swept on short window");
    }

    #[tokio::test]
    async fn sweep_drops_failed_and_cancelled_on_short_window() {
        let reg = QueryRegistry::with_retention(Duration::from_secs(3600), Duration::ZERO);
        let failed = reg.create(caller().to_string()).await;
        reg.mark_failed(&failed.query_id, "provider-timeout", "late").await;
        let cancelled = reg.create(caller().to_string()).await;
        reg.cancel(&cancelled.query_id, caller(), &cancelled.retrieval_token)
            .await
            .unwrap();
        reg.sweep().await;
        assert_eq!(reg.len().await, 0, "failed + cancelled swept immediately");
    }
}
