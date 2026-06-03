//! Seam discipline: the boundary traits through which the agent reaches
//! the outside world.
//!
//! The rule (design-doc P13, established in build phase B0): the execution
//! engine must touch the outside world *only* through a small set of
//! trait "seams", never through direct calls. That is what makes the agent
//! testable — an event/idle/stochastic agent is otherwise nearly
//! impossible to drive deterministically. Each seam has a production
//! implementation and a test double:
//!
//! | Seam            | production            | test double        |
//! |-----------------|-----------------------|--------------------|
//! | [`Clock`]       | [`SystemClock`]       | [`ManualClock`]    |
//! | `GraphHandle`   | `UnixGraphClient`     | `MockGraphClient`  |
//! | `TriggerSource` | Event Bus subscriber  | injected queue     |
//! | provider        | `ai_core::provider`   | mock provider      |
//! | [`GateObserver`]| audit/inspection sink | recording observer |
//!
//! **Scope of B0.** B0 ships the manifest contract and the seam *rule*. It
//! defines the seams whose shape is already settled and concretely useful
//! — [`Clock`] (so no engine code ever calls `SystemTime::now()` directly)
//! and [`GateObserver`] (it observes the existing
//! [`lunaris_ai_core::capability::ActionDecision`]). `TriggerSource` and
//! `GraphHandle` are intentionally *not* defined here yet: their method
//! shape is determined by their first consumer (the router in B1, the
//! engine in B2), and inventing trait signatures with no caller is
//! speculative generality. They land with that consumer, behind this same
//! discipline. The provider seam already exists as
//! [`lunaris_ai_core::provider::AIProvider`] and is reused as-is.

use std::collections::BTreeMap;
use std::future::Future;
use std::time::SystemTime;

use lunaris_ai_core::capability::ActionDecision;

/// A source of wall-clock time. The engine reads time only through this,
/// so tests can pin or advance it ([`ManualClock`]) and idle/timing
/// behaviour becomes deterministic. No engine code calls
/// `SystemTime::now()` directly.
///
/// (Monotonic time for loop wall-clock budgets is a separate concern that
/// arrives with the engine in B2; this seam covers the wall-clock
/// decisions the router/idle layer make.)
pub trait Clock: Send + Sync {
    /// The current wall-clock time.
    fn now(&self) -> SystemTime;
}

/// The production clock: the real system wall clock.
#[derive(Debug, Default, Clone, Copy)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> SystemTime {
        SystemTime::now()
    }
}

/// A test clock whose time only moves when the test moves it. Lets the
/// idle/timing paths be exercised without real time passing.
#[derive(Debug)]
pub struct ManualClock {
    now: std::sync::Mutex<SystemTime>,
}

impl ManualClock {
    /// Create a manual clock fixed at `start`.
    pub fn new(start: SystemTime) -> Self {
        Self {
            now: std::sync::Mutex::new(start),
        }
    }

    /// Move the clock forward by `delta`.
    pub fn advance(&self, delta: std::time::Duration) {
        let mut now = self.now.lock().expect("ManualClock mutex poisoned");
        *now += delta;
    }

    /// Set the clock to an absolute time.
    pub fn set(&self, when: SystemTime) {
        *self.now.lock().expect("ManualClock mutex poisoned") = when;
    }
}

impl Clock for ManualClock {
    fn now(&self) -> SystemTime {
        *self.now.lock().expect("ManualClock mutex poisoned")
    }
}

/// Observes every capability-gate decision. This is the read-only tap the
/// audit sink, the anomaly detector, and the P13 inspection plane attach
/// to — the engine reports each [`ActionDecision`] here rather than
/// logging directly, so observation is a seam, not scattered side effects.
pub trait GateObserver: Send + Sync {
    /// Called with the gate's decision for one proposed action.
    fn observed(&self, decision: &ActionDecision);
}

/// A no-op observer (the default when nothing is attached).
#[derive(Debug, Default, Clone, Copy)]
pub struct NullObserver;

impl GateObserver for NullObserver {
    fn observed(&self, _decision: &ActionDecision) {}
}

/// A decoded Event Bus event the router and engine act on: its type plus a
/// flat map of payload fields. The production [`TriggerSource`] decodes the
/// prost `Event` (type + the per-payload fields the router needs); tests
/// construct these directly.
#[derive(Debug, Clone)]
pub struct AgentEvent {
    /// The event's stable id (from the Event envelope), used to correlate
    /// the resulting gate/audit entries.
    pub id: String,
    /// The event type string, e.g. `file.opened`.
    pub event_type: String,
    /// The payload fields the router/filters read (e.g. `path`, `app_id`).
    pub fields: BTreeMap<String, String>,
    /// Whether this event carries (or was triggered by) external content —
    /// a trusted origin fact the [`TriggerSource`] decoder stamps from the
    /// event source (S18-A origin tagging). Any action triggered by it must
    /// be confirmed (prompt-injection containment), so the production
    /// decoder defaults an *unknown* origin to `true` (fail-safe); a local
    /// system event (e.g. `file.opened` from the kernel layer) is `false`.
    pub external_content: bool,
}

/// The source of trigger events the engine consumes. The production impl
/// wraps the Event Bus consumer (decoding each frame into an
/// [`AgentEvent`]); tests inject a queue. Returning `impl Future + Send`
/// rather than a bare `async fn` keeps the auto-trait bound explicit (and
/// avoids the `async_fn_in_trait` lint) for use behind the engine loop.
pub trait TriggerSource {
    /// The next event, or `None` when the source is exhausted / closed.
    fn recv(&mut self) -> impl Future<Output = Option<AgentEvent>> + Send;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    #[test]
    fn manual_clock_only_moves_when_told() {
        let t0 = SystemTime::UNIX_EPOCH;
        let clock = ManualClock::new(t0);
        assert_eq!(clock.now(), t0);
        clock.advance(Duration::from_secs(3600));
        assert_eq!(clock.now(), t0 + Duration::from_secs(3600));
    }

    #[test]
    fn system_clock_is_monotonic_enough() {
        let clock = SystemClock;
        let a = clock.now();
        let b = clock.now();
        assert!(b >= a);
    }

    // A recording observer doubles as the test double for the seam.
    struct Counter(AtomicUsize);
    impl GateObserver for Counter {
        fn observed(&self, _decision: &ActionDecision) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn gate_observer_receives_decisions() {
        let counter = Counter(AtomicUsize::new(0));
        counter.observed(&ActionDecision::Proceed);
        counter.observed(&ActionDecision::Proceed);
        assert_eq!(counter.0.load(Ordering::SeqCst), 2);
        // NullObserver is the inert default.
        NullObserver.observed(&ActionDecision::Proceed);
    }
}
