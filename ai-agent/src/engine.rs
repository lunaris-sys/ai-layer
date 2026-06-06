//! Execution engine: the deterministic B1 dispatch spine.
//!
//! It consumes trigger events from a [`TriggerSource`], routes each to the
//! enabled behaviours that match ([`crate::router`]), runs each matched
//! **workflow** behaviour's code handler, and passes any action the
//! handler proposes through the [`Gate`] (capability + fail-closed audit).
//! The result of each dispatch is a [`DispatchOutcome`] (for logging now,
//! and for the P9 surfaces — Waypointer Suggestions / notifications —
//! later).
//!
//! Scope of this increment: the spine, end-to-end and testable with an
//! injected source + stub handlers + a mock audit sink. Deliberately *not*
//! here yet: `kind: agent` behaviours (the bounded LLM loop is B2); the
//! real `auto-tag-by-project` handler (it reads the graph through a
//! `GraphHandle` seam that lands with it); the production `TriggerSource`
//! over `UnixEventConsumer` + prost-`Event` decoding; per-behaviour burst
//! coalescing (gap G1); and the `main.rs` daemon wiring. Each is a
//! follow-up that slots behind these same seams.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use futures::FutureExt as _;
use lunaris_ai_core::capability::{AccessTier, ActionDecision, BaselineMode};
use lunaris_ai_core::provider::{AIProvider, CompletionRequest};

use crate::agentic::{build_agent_prompt, external_screen_text, parse_agent_step, AgentStep};
use lunaris_ai_classifier::{screen, ClassifierPolicy, InjectionClassifier, Verdict};
use crate::behaviour::{Behaviour, BehaviourKind, ReadScope};
use crate::compaction::{self, CompactionPolicy, TranscriptEntry};
use crate::gate::{ActionContext, DecisionReason, Gate, GateError, ProposedAction};
use crate::executor::DryRunReport;
use crate::loader::LoadedBehaviour;
use crate::registry::plan_for;
/// Re-exported so the public [`DispatchOutcome::Decided`] can carry a plan while
/// the trusted registry module itself stays crate-private.
pub use crate::registry::ExecutionPlan;
use crate::router::matching_behaviours;
use crate::seams::{AgentEvent, Clock, DeniedGraph, GraphHandle, TriggerSource};

/// The trusted app id the agent acts as for now. Proper per-app resolution
/// (from the tool binding / the behaviour identity) lands later; until then
/// the agent acts as itself, and execution is capped to confirmation
/// regardless, so this never widens authority.
const AGENT_APP_ID: &str = "org.lunaris.agent";

/// Compute the dry-run executor's plan for a gate decision, to carry on the
/// dispatch outcome (the bin logs the successful plan from there). Suggest-mode:
/// this performs no write. The manual `Propose` flow plans nothing; an action
/// the executor cannot plan is warned at the source, never guessed, and yields
/// `None`.
fn plan_dry_run(
    behaviour: &str,
    action: &ProposedAction,
    decision: ActionDecision,
    tool_scope: &[String],
) -> Option<DryRunReport> {
    match crate::executor::dry_run(action, decision, tool_scope) {
        Ok(report) => report,
        Err(e) => {
            tracing::warn!(
                behaviour = %behaviour,
                "dry-run executor could not plan write: {e}"
            );
            None
        }
    }
}

/// Wall-clock bound on a single workflow handler run. A handler that blocks
/// or runs away is abandoned with a Failed outcome rather than stalling the
/// loop. (Agent-loop budgets, which are per-step, arrive separately.)
const HANDLER_TIMEOUT: Duration = Duration::from_secs(10);

/// Default per-behaviour coalescing window (gap G1). A burst of identical
/// events for one behaviour within this window fires it once. Short by design:
/// long enough to collapse a "x100 in a second" storm, short enough not to
/// suppress a deliberate re-trigger seconds later. Tunable via
/// [`Dispatcher::with_coalesce_window`]; per-behaviour tuning from the manifest
/// is a follow-up.
const DEFAULT_COALESCE_WINDOW: Duration = Duration::from_secs(1);

/// Hard cap on the coalescer's tracking map (gap G1), so a storm of distinct
/// events (many unique paths within one window, e.g. a build or a `find`)
/// cannot grow it without bound. At the cap, stale entries are pruned; if it is
/// still full of fresh distinct events, the map is cleared (coalescing forgets
/// recent entries, never dropping a distinct dispatch). Comparable to the
/// kernel-layer normaliser's dedup-map cleanup threshold.
const MAX_COALESCE_ENTRIES: usize = 4096;

/// Upper bound on the external text the injection screen (S17) will run the
/// classifier over. Event metadata (paths, titles) is tiny; a payload past this
/// is not normal and would force the classifier to score many windows, so it is
/// blocked (fail closed) without running inference. Bounds the per-event
/// classifier work against a hostile oversized field.
const MAX_SCREEN_BYTES: usize = 64 * 1024;

/// How many times the agent loop tolerates the identical [`ProgressKey`]
/// consecutively before stopping for no progress. Counts repeats after the
/// first sighting (here: same key, again, again -> stop on the third).
const MAX_NO_PROGRESS_REPEATS: u32 = 2;

/// Upper bound on one injection-screen scoring call. ONNX inference on bounded
/// input is milliseconds, so this is generous headroom; its job is to stop a
/// wedged or pathologically slow classifier from pinning the dispatch loop (and
/// delaying config revocation / shutdown) indefinitely. A timeout fails closed
/// (the content is blocked).
const SCREEN_TIMEOUT: Duration = Duration::from_secs(5);

/// The identity of one bounded-loop step for the no-progress guard (technique
/// adapted from goose's tool_monitor, Apache-2.0). The same key repeated means
/// the model is stuck, since the loop neither executes actions nor observes
/// state between steps, so nothing changes to justify the repeat. Keyed
/// differently by outcome. A refusal keys on the tool only, so a model cannot
/// dodge the guard by re-wording the rationale of an action the gate keeps
/// refusing. An acceptance keys on the tool *and* summary, so the same in-scope
/// tool proposed for genuinely different work (a different summary) counts as
/// progress, while an exact duplicate suggestion does not. Any different key
/// resets the streak.
#[derive(Debug, Clone, PartialEq, Eq)]
enum ProgressKey {
    /// An accepted (gated) proposal: tool + summary.
    Accepted(String, String),
    /// A refused proposal: tool only.
    Refused(String),
}

/// What a workflow handler decides for a matched event.
#[derive(Debug, Clone)]
pub enum HandlerOutcome {
    /// Propose an action; it is gated before being surfaced/executed.
    Propose(ProposedAction),
    /// Reached a terminal condition with no action (e.g. `no_matching_project`).
    Terminal(String),
}

/// A handler that failed to produce an outcome. The dispatcher records it
/// and moves on to the next behaviour — one bad handler never stalls the
/// loop.
#[derive(Debug, thiserror::Error)]
#[error("handler failed: {0}")]
pub struct HandlerError(pub String);

/// A workflow behaviour's code handler. Async, with a read-only
/// [`GraphHandle`] for graph-backed behaviours. Returns a `Result` so a
/// handler can fail gracefully; a *panic* or a timeout is also contained by
/// the dispatcher (see [`Dispatcher::dispatch`]).
#[async_trait::async_trait]
pub trait WorkflowHandler: Send + Sync {
    /// Run the workflow for one matched event.
    async fn run(
        &self,
        event: &AgentEvent,
        graph: &dyn GraphHandle,
    ) -> Result<HandlerOutcome, HandlerError>;
}

/// Maps a behaviour manifest's `handler` id to its code handler.
pub type HandlerRegistry = BTreeMap<String, Box<dyn WorkflowHandler>>;

/// What the live executor did with a proven decision, carried on
/// [`DispatchOutcome::Decided`] so a write failure is surfaced rather than
/// erased. Only ever set when the daemon opted into executing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutionResult {
    /// The authorised write was performed; the variant records whether the edge
    /// was created or already present.
    Written(crate::executor::WriteOutcome),
    /// Execution was refused or definitely did not write (a stale proof, an
    /// audit outage, a pre-send write error). The reason is for logging and
    /// recovery; the attempt, if it reached the write, was audited beforehand.
    Failed(String),
    /// The write timed out after the request may already have been sent, so it
    /// is **unknown** whether the daemon committed the relation. NOT a failure:
    /// reporting it as one would lose the authoritative created/exists signal for
    /// a write that did commit. It is pre-audited, and the executor's
    /// re-validation reconciles it on the next run (a committed edge fails the
    /// `Not(EdgeExists)` proof, so it is neither double-written nor lost).
    Indeterminate(String),
}

/// The outcome of dispatching one matched behaviour for one event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DispatchOutcome {
    /// The gate decided on a proposed action (and audited it).
    Decided {
        /// The behaviour that ran.
        behaviour: String,
        /// The action the gate decided on, carried so a surface can show it
        /// to the user (the summary is for display; the audit subject stays
        /// content-free).
        action: ProposedAction,
        /// The gate's decision.
        decision: ActionDecision,
        /// The faithful reason for that decision (D2), from the gate's logic.
        reason: DecisionReason,
        /// The audit ledger index of the recorded decision.
        audit_index: u64,
        /// What the action would do and how to undo it (B1 compensation), for a
        /// registered action; `None` for an unregistered tool. Surfaced so the
        /// decision's concrete effect and its undo are visible (logged today).
        plan: Option<ExecutionPlan>,
        /// The dry-run executor's plan for this decision: the concrete relation
        /// write it would perform plus its strict-create condition. `Some` only
        /// for a proven `PreviewThenExecute` whose action the executor can plan
        /// (an unproven `RequireConfirmation`, the manual `Propose`, or an
        /// unplannable action yields `None`). This is a non-authoritative record
        /// for the activity/preview surface: a live executor must NOT write from
        /// it but re-run the full trusted precondition proof atomically at write
        /// time (obligation 2), since the report does not carry the point-in-time
        /// preconditions (e.g. `PathUnderField`) that justified the lift.
        dry_run: Option<DryRunReport>,
        /// What the live executor did with this decision, when the daemon opted
        /// into executing. `None` in suggest-mode (no executor) or for a
        /// non-executing decision; `Some(Written)` when the authorised write was
        /// performed; `Some(Failed)` when execution was refused or failed (a
        /// stale proof, an audit outage, a write error) so the failure is not
        /// erased but surfaced for logging and recovery. The act, when attempted,
        /// is audited before the write regardless of this outcome.
        executed: Option<ExecutionResult>,
    },
    /// The handler reached a terminal condition with no action.
    Terminal {
        /// The behaviour that ran.
        behaviour: String,
        /// The terminal outcome name.
        outcome: String,
    },
    /// The gate refused the action (e.g. out of scope, or audit down).
    Refused {
        /// The behaviour that ran.
        behaviour: String,
        /// Why it was refused.
        reason: String,
    },
    /// The handler returned an error or panicked; isolated so the rest of
    /// the dispatch continues.
    Failed {
        /// The behaviour whose handler failed.
        behaviour: String,
        /// Why it failed.
        reason: String,
    },
    /// The behaviour matched but was not run (not a workflow, or no handler).
    Skipped {
        /// The behaviour that matched.
        behaviour: String,
        /// Why it was skipped.
        reason: String,
    },
    /// The behaviour matched but its dispatch was coalesced into a recent
    /// identical one (gap G1 burst dedup), so it did not run again this window.
    Coalesced {
        /// The behaviour whose dispatch was coalesced.
        behaviour: String,
    },
    /// An agent behaviour was not run because the event's external content was
    /// blocked by the injection classifier (S17): the content never reaches
    /// the model. Only agent behaviours are screened (they alone feed content
    /// to an LLM); the reason is content-free.
    Blocked {
        /// The behaviour that would have run.
        behaviour: String,
        /// Why it was blocked (content-free).
        reason: String,
    },
}

/// Per-behaviour burst coalescing (gap G1). Suppresses a repeat dispatch of a
/// behaviour for an identical event within a short window, so a burst (e.g.
/// `file.opened` x100 for one path in a second) fires the behaviour once, not
/// once per event. Distinct from S15 (the knowledge daemon's per-identity query
/// rate limiter): this is the agent's own dispatch dedup, keyed on (behaviour,
/// event), and it drops the duplicate dispatch rather than throttling a caller.
///
/// A dispatch is keyed by a fixed-size digest of (behaviour, event type, decoded
/// fields, external-content origin), hashed with a per-process random seed. The
/// digest keeps the map's per-entry memory bounded regardless of how large a
/// producer-controlled field is (a long `path` cannot bloat it) and still
/// coalesces such events rather than letting them bypass the dedup; the random
/// seed makes a crafted colliding key (to drop a distinct event) infeasible,
/// and a chance collision at the map's bounded size is astronomically unlikely
/// and would at worst suppress one suggestion. The external-content origin is
/// part of the key because a local and an external event with otherwise-
/// identical fields are not the same dispatch (the gate forces confirmation for
/// an external trigger and audits it on its own), so they must not coalesce.
struct Coalescer {
    window: Duration,
    /// Per-process random seed for the key digest, so a producer cannot craft a
    /// colliding key without knowing it.
    hasher: std::collections::hash_map::RandomState,
    /// Key digest to the time that (behaviour, event) was last admitted.
    seen: HashMap<u64, SystemTime>,
}

impl Coalescer {
    fn new(window: Duration) -> Self {
        Self {
            window,
            hasher: std::collections::hash_map::RandomState::new(),
            seen: HashMap::new(),
        }
    }

    /// The fixed-size digest a dispatch is coalesced on.
    fn digest(
        &self,
        behaviour: &str,
        event_type: &str,
        fields: &BTreeMap<String, String>,
        external_content: bool,
    ) -> u64 {
        use std::hash::{BuildHasher as _, Hash as _, Hasher as _};
        let mut h = self.hasher.build_hasher();
        behaviour.hash(&mut h);
        event_type.hash(&mut h);
        // A BTreeMap hashes in sorted key order, so the same fields always
        // digest identically.
        fields.hash(&mut h);
        external_content.hash(&mut h);
        h.finish()
    }

    /// Decide whether to dispatch the keyed event at `now`, recording the time
    /// when it admits. Returns `true` (dispatch) when the key is new or its last
    /// dispatch is older than the window; `false` (coalesce) when a dispatch
    /// happened within the window. The window is measured from the first
    /// dispatch of a burst, not extended by each coalesced duplicate, so a
    /// sustained stream fires once per window rather than being suppressed
    /// forever.
    fn admit(
        &mut self,
        behaviour: &str,
        event_type: &str,
        fields: &BTreeMap<String, String>,
        external_content: bool,
        now: SystemTime,
    ) -> bool {
        let key = self.digest(behaviour, event_type, fields, external_content);
        let window = self.window;
        // Bound cost and memory. The common case (a small map) does no scan:
        // expiry is lazy, per key, on access below. Only when the map has grown
        // to the cap do we prune stale entries in one pass; if a genuine storm
        // of distinct, still-fresh events keeps it at the cap, clear it
        // entirely. Clearing only forgets recent entries, so at worst a few
        // duplicates slip through afterwards (over-dispatch, never a dropped
        // distinct event), while per-event cost stays amortised O(1) and the
        // map stays bounded under a noisy or hostile producer.
        if self.seen.len() >= MAX_COALESCE_ENTRIES {
            self.seen.retain(|_, last| {
                now.duration_since(*last)
                    .map(|elapsed| elapsed < window)
                    .unwrap_or(false)
            });
            if self.seen.len() >= MAX_COALESCE_ENTRIES {
                self.seen.clear();
            }
        }
        match self.seen.entry(key) {
            std::collections::hash_map::Entry::Occupied(mut slot) => {
                // Within the window: coalesce, without refreshing (the window is
                // measured from the first dispatch of a burst, not extended by
                // each duplicate). Expired, or future-stamped after a backwards
                // clock move: treat as stale, refresh to `now` and admit, rather
                // than suppressing the event past the window.
                let within = now
                    .duration_since(*slot.get())
                    .map(|elapsed| elapsed < window)
                    .unwrap_or(false);
                if within {
                    false
                } else {
                    slot.insert(now);
                    true
                }
            }
            std::collections::hash_map::Entry::Vacant(slot) => {
                slot.insert(now);
                true
            }
        }
    }
}

/// The dispatch engine over a set of loaded behaviours, their handlers, the
/// graph handle they read through, the action gate, and (for `kind: agent`
/// behaviours) the provider that drives the bounded loop plus the clock the
/// loop measures its wall-clock budget against.
pub struct Dispatcher<'a> {
    behaviours: &'a [LoadedBehaviour],
    handlers: &'a HandlerRegistry,
    graph: &'a dyn GraphHandle,
    /// The agent's configured global read tier; a behaviour declaring more
    /// is refused before its handler runs.
    read_tier: AccessTier,
    gate: Gate<'a>,
    /// The live action executor, present only when the daemon opts into
    /// executing (default off: suggest-mode, nothing is written). When set, a
    /// proven `PreviewThenExecute` workflow decision is executed after the gate
    /// records it: the executor re-validates against the current graph and
    /// performs the authorised, audited write. `None` leaves the loop in
    /// suggest-mode, where the decision is surfaced but never acted on.
    executor: Option<crate::executor::LiveExecutor<'a>>,
    /// The LLM provider the bounded agent loop drives. `None` when no
    /// provider is configured, in which case `kind: agent` behaviours are
    /// skipped (workflow behaviours never need one).
    provider: Option<&'a dyn AIProvider>,
    /// The clock the agent loop measures its wall-clock budget against
    /// (a seam so the budget is deterministic under test).
    clock: &'a dyn Clock,
    /// How the agent loop keeps its working memory inside the model's
    /// context window. Defaults to a conservative fixed buffer; the daemon
    /// (and the provider, once wired) can override it via
    /// [`Dispatcher::with_compaction`].
    compaction: CompactionPolicy,
    /// Per-behaviour burst coalescing (gap G1). Behind a mutex because
    /// `dispatch` takes `&self` yet must record each admitted dispatch; the
    /// lock is held only for the synchronous admit decision, never across an
    /// await. Reset when the dispatcher is rebuilt (a config reload or provider
    /// recovery), which is acceptable for a burst-dedup optimisation.
    coalescer: std::sync::Mutex<Coalescer>,
    /// How external content is screened (S17) before it can reach the model in
    /// an agent loop. See [`ScreeningMode`]: `Off` flows under the gate's
    /// mandatory-confirmation containment, `FailClosed` blocks external-content
    /// agent loops (a configured-but-unavailable classifier), `On` screens.
    screening: ScreeningMode,
    /// Single-flight guard for classifier scoring (S17). The owned permit is
    /// held by the blocking scorer task until it actually finishes, not just
    /// until its timeout, so a wedged score (one that outlives [`SCREEN_TIMEOUT`])
    /// keeps the permit and every later external event fails closed (blocks)
    /// rather than spawning another blocking task. This bounds outstanding
    /// scorer tasks to one and stops a wedged classifier from exhausting the
    /// blocking pool. A fresh dispatcher (a config epoch) gets a fresh permit,
    /// acceptable since epochs are rare and a wedge is pathological.
    screen_gate: Arc<tokio::sync::Semaphore>,
}

/// How the dispatcher screens external content for prompt injection (S17).
///
/// The three states keep a configured-but-broken classifier from silently
/// degrading to no screening: a packaging error or config typo must fail
/// closed, while a deliberately unprovisioned classifier (the default build or
/// no `[classifier]` section) flows under the gate's always-on
/// confirm-on-external-trigger containment.
///
/// The classifier is held as an `Arc` (not a borrow) so scoring can run on a
/// bounded blocking task: ONNX inference is a blocking native call, and running
/// it inline would pin the dispatch poll so config revocation and shutdown
/// could not preempt it. The `Arc` is cheap to clone into that task.
#[derive(Clone)]
pub enum ScreeningMode {
    /// No classifier provisioned (default build, or no `[classifier]` config).
    /// External content is sanitised (S18-B) and flows; the gate's mandatory
    /// confirmation for externally-triggered actions is the containment.
    Off,
    /// A classifier was configured but could not be loaded (missing model,
    /// bad path, unreadable export, invalid thresholds). External content cannot
    /// be screened, so it is blocked from reaching the model: an intended screen
    /// that is broken fails closed rather than silently disabling S17.
    FailClosed,
    /// Screen with this classifier and threshold policy.
    On(Arc<dyn InjectionClassifier>, ClassifierPolicy),
}

impl<'a> Dispatcher<'a> {
    /// Build a dispatcher. The compaction policy defaults to a conservative
    /// fixed buffer; override it with [`Dispatcher::with_compaction`].
    pub fn new(
        behaviours: &'a [LoadedBehaviour],
        handlers: &'a HandlerRegistry,
        graph: &'a dyn GraphHandle,
        read_tier: AccessTier,
        gate: Gate<'a>,
        provider: Option<&'a dyn AIProvider>,
        clock: &'a dyn Clock,
    ) -> Self {
        Self {
            behaviours,
            handlers,
            graph,
            read_tier,
            gate,
            executor: None,
            provider,
            clock,
            compaction: CompactionPolicy::default(),
            coalescer: std::sync::Mutex::new(Coalescer::new(DEFAULT_COALESCE_WINDOW)),
            screening: ScreeningMode::Off,
            screen_gate: Arc::new(tokio::sync::Semaphore::new(1)),
        }
    }

    /// Opt into executing proven workflow decisions through the given live
    /// executor. Without this the dispatcher stays in suggest-mode: a decision
    /// is gated, audited, and surfaced, but nothing is written. The daemon sets
    /// this only when its config enables the executor.
    pub fn with_executor(mut self, executor: crate::executor::LiveExecutor<'a>) -> Self {
        self.executor = Some(executor);
        self
    }

    /// Override the context-compaction policy (e.g. with the real model's
    /// window once a provider is wired).
    pub fn with_compaction(mut self, policy: CompactionPolicy) -> Self {
        self.compaction = policy;
        self
    }

    /// Set the external-content screening mode (S17). [`ScreeningMode::Off`]
    /// (the default) flows external content under the gate's confirmation
    /// containment; [`ScreeningMode::FailClosed`] blocks it; [`ScreeningMode::On`]
    /// screens it with a classifier.
    pub fn with_screening_mode(mut self, mode: ScreeningMode) -> Self {
        self.screening = mode;
        self
    }

    /// Enable prompt-injection screening (S17) of external content with the
    /// given classifier and threshold policy. Convenience for
    /// `with_screening_mode(ScreeningMode::On(..))`.
    pub fn with_screening(
        self,
        classifier: Arc<dyn InjectionClassifier>,
        policy: ClassifierPolicy,
    ) -> Self {
        self.with_screening_mode(ScreeningMode::On(classifier, policy))
    }

    /// Use a shared, process-lived single-flight gate for classifier scoring
    /// (S17) instead of this dispatcher's own. The daemon rebuilds the
    /// dispatcher on every config reload and provider rearm; a wedged scorer
    /// task from a prior dispatcher keeps its permit, so passing the same gate
    /// into each rebuild keeps the "one outstanding scorer" bound across
    /// rebuilds (a fresh per-dispatcher gate would let repeated rearms spawn a
    /// new scorer each time and exhaust the blocking pool).
    pub fn with_screen_gate(mut self, gate: Arc<tokio::sync::Semaphore>) -> Self {
        self.screen_gate = gate;
        self
    }

    /// Override the per-behaviour coalescing window (gap G1). Mainly for tests,
    /// which pair it with a manual clock to exercise the window deterministically.
    pub fn with_coalesce_window(mut self, window: Duration) -> Self {
        self.coalescer = std::sync::Mutex::new(Coalescer::new(window));
        self
    }

    /// Dispatch one event: route it to every enabled matching behaviour and
    /// run each, returning the outcomes. A workflow behaviour yields one
    /// outcome; a `kind: agent` behaviour yields one per loop step plus a
    /// terminal.
    pub async fn dispatch(&self, event: &AgentEvent) -> Vec<DispatchOutcome> {
        let mut outcomes = Vec::new();
        // A trusted, consumer-side timestamp for coalescing: the dispatcher's
        // wall clock, taken once per event for all behaviours it matches.
        let now = self.clock.now();
        // S17 verdict for this event's external content, computed lazily inside
        // the admitted agent path (after matching, filters, coalescing, and the
        // kind check) so a workflow-only, filtered, or fully-coalesced external
        // event never runs the classifier, and cached so a burst of matched
        // agent behaviours screens once. The content is event-level.
        let mut external_verdict: Option<Verdict> = None;
        for lb in matching_behaviours(&event.event_type, &event.fields, self.behaviours) {
            let name = lb.behaviour.manifest.name.clone();
            // G1: coalesce a burst of identical events for this behaviour, so a
            // storm of one event fires it once per window, not once per event.
            // The window is measured on the dispatcher's wall clock (a trusted,
            // consumer-side time), NOT the bus envelope timestamp: producers do
            // not agree on that field's unit or epoch (the kernel layer sends
            // monotonic ns-since-boot, the compositor Unix micros) and it is
            // producer-controlled (spoofable), so it is neither comparable
            // across sources nor trustworthy for a window. The trade-off: the
            // daemon awaits each dispatch before receiving the next event, so an
            // identical burst queued behind a slow handler (a multi-step agent
            // loop) is processed more than a window apart and is not coalesced.
            // This degrades to no-dedup, which is safe (over-dispatch, never a
            // wrong drop), and the realistic coalesced case avoids it (fast
            // workflow handlers on high-frequency events; agent loops trigger on
            // field-less calendar/schedule events, which are not coalesced). A
            // backlog-proof form needs a trusted ingestion timestamp stamped
            // before dispatch (in the SDK event consumer) or ingestion decoupled
            // from dispatch, a deliberate follow-up. Coalesce only events with a
            // stable identity (non-empty decoded fields): an event with no
            // decoded fields carries no entity to key on, so distinct ones would
            // collide and a real one could be dropped, so those always dispatch.
            // The key is a bounded digest, so a large producer-controlled field
            // neither bloats the map nor bypasses dedup. Lock only for the
            // synchronous admit decision, released before any await below.
            let admitted = if event.fields.is_empty() {
                true
            } else {
                self.coalescer
                    .lock()
                    .expect("coalescer mutex poisoned")
                    .admit(
                        &name,
                        &event.event_type,
                        &event.fields,
                        event.external_content,
                        now,
                    )
            };
            if !admitted {
                outcomes.push(DispatchOutcome::Coalesced { behaviour: name });
                continue;
            }
            if lb.behaviour.manifest.kind == BehaviourKind::Agent {
                // Preflight every non-model eligibility check before screening:
                // no provider (a bus outage), an over-scoped read tier, or a
                // missing budget all skip the behaviour without reaching the
                // model, so the classifier must not run for them (else a degraded
                // provider or an over-scoped agent becomes a per-event classifier
                // CPU cost on events that go nowhere).
                if let Some(reason) =
                    agent_skip_reason(&lb.behaviour.manifest, self.read_tier, self.provider.is_some())
                {
                    outcomes.push(DispatchOutcome::Skipped { behaviour: name, reason });
                    continue;
                }
                // S17: blocked external content never reaches the model. The gate
                // (and its mandatory confirmation) is downstream of the model, so
                // blocking here, before the loop runs, is the only place that
                // stops poisoned content from being read at all. Reached only for
                // an eligible agent (provider present, read-scope ok, budget set),
                // computed once per event and cached across agent behaviours.
                if event.external_content {
                    let verdict = match external_verdict {
                        Some(v) => v,
                        None => {
                            let v = self.screen_external(event).await;
                            // Surface a Warn: the classifier finds the content
                            // suspicious but not blocking. Per the policy it
                            // passes to the model, and any action it triggers is
                            // already gated to confirmation (external_trigger), so
                            // this logs the signal rather than dropping it. A
                            // richer surface (audit entry, confirmation-UI flag)
                            // is a P9 follow-up once that surface exists.
                            if v == Verdict::Warn {
                                tracing::warn!(
                                    event = %event.event_type,
                                    "external content flagged suspicious (Warn) by the injection classifier; passed to the model, any resulting action still requires confirmation"
                                );
                            }
                            external_verdict = Some(v);
                            v
                        }
                    };
                    if verdict == Verdict::Block {
                        outcomes.push(DispatchOutcome::Blocked {
                            behaviour: name,
                            reason: "external content blocked before reaching the model"
                                .to_string(),
                        });
                        continue;
                    }
                }
                let provider = self
                    .provider
                    .expect("agent_skip_reason returns a reason when no provider");
                // Each agent loop gets its own wall-clock anchor, read here
                // immediately before it runs, so an earlier behaviour's runtime
                // never eats into this loop's budget.
                let start = self.clock.now();
                outcomes.extend(self.run_agent_loop(lb, event, provider, start).await);
                continue;
            }
            outcomes.push(self.dispatch_one(lb, event).await);
        }
        outcomes
    }

    /// Screen this event's external content (S17), returning the verdict the
    /// dispatcher acts on. `Off` is [`Verdict::Allow`] (external content flows
    /// under the gate's confirmation containment); `FailClosed` is
    /// [`Verdict::Block`] (a configured classifier that could not load); `On`
    /// screens the sanitised external text and returns the classifier's verdict,
    /// mapping a classifier error (`screen` fails closed), an oversized payload
    /// (bounding the work), a scoring timeout/panic, or a busy single-flight
    /// gate to [`Verdict::Block`]. Only called for an external-content event in
    /// the admitted agent path.
    ///
    /// Scoring runs on a blocking task bounded by [`SCREEN_TIMEOUT`]: ONNX
    /// inference is a blocking native call, so running it inline would pin the
    /// dispatch poll and stop config revocation / shutdown from preempting. The
    /// `await` here is that preemption point; a timeout or a panic in the
    /// blocking task fails closed (block).
    async fn screen_external(&self, event: &AgentEvent) -> Verdict {
        match &self.screening {
            ScreeningMode::Off => Verdict::Allow,
            ScreeningMode::FailClosed => Verdict::Block,
            ScreeningMode::On(classifier, policy) => {
                let text = external_screen_text(event);
                // Refuse (block) external content past the cap rather than run
                // unbounded inference windows on a hostile oversized payload.
                if text.len() > MAX_SCREEN_BYTES {
                    return Verdict::Block;
                }
                // Single-flight: take the one permit, held inside the blocking
                // task until it truly finishes. If it is unavailable a prior
                // score is still running (wedged past its timeout), so fail
                // closed rather than spawn another blocking task and risk
                // exhausting the pool.
                let Ok(permit) = Arc::clone(&self.screen_gate).try_acquire_owned() else {
                    return Verdict::Block;
                };
                let classifier = Arc::clone(classifier);
                let policy = *policy;
                let scored = tokio::time::timeout(
                    SCREEN_TIMEOUT,
                    tokio::task::spawn_blocking(move || {
                        // Hold the permit for the real duration of the native
                        // call, so a wedge keeps blocking later scorers.
                        let _permit = permit;
                        screen(&*classifier, &policy, &text)
                    }),
                )
                .await;
                // Timeout (Err) or a panicked blocking task (Ok(Err)): fail closed.
                match scored {
                    Ok(Ok(verdict)) => verdict,
                    _ => Verdict::Block,
                }
            }
        }
    }

    async fn dispatch_one(&self, lb: &LoadedBehaviour, event: &AgentEvent) -> DispatchOutcome {
        let m = &lb.behaviour.manifest;
        let behaviour = m.name.clone();

        // B1 runs workflow behaviours only; the bounded agent loop is B2.
        if m.kind != BehaviourKind::Workflow {
            // The bounded agent loop is not run by this engine yet.
            return DispatchOutcome::Skipped {
                behaviour,
                reason: "agent behaviours are not run by this engine yet".to_string(),
            };
        }
        // Read scope: a behaviour may not read more of the graph than the
        // agent is granted. Refused before the handler runs.
        if !reads_satisfied(m.reads, self.read_tier) {
            return DispatchOutcome::Skipped {
                behaviour,
                reason: format!(
                    "declared read scope {:?} exceeds the configured grant",
                    m.reads
                ),
            };
        }
        let Some(handler_id) = m.handler.as_deref() else {
            // A workflow without a handler is rejected at load; backstop.
            return DispatchOutcome::Skipped {
                behaviour,
                reason: "no handler declared".to_string(),
            };
        };
        let Some(handler) = self.handlers.get(handler_id) else {
            return DispatchOutcome::Skipped {
                behaviour,
                reason: format!("handler '{handler_id}' not registered"),
            };
        };

        // A Minimal-reads behaviour gets no graph access (a denying handle),
        // so its declared scope is enforced on the actual query, not just at
        // enablement. Finer per-behaviour sub-tier scoping (e.g. capping a
        // session-scoped behaviour under a full grant) needs a per-query
        // scope on the daemon request and is a documented follow-up.
        let denied = DeniedGraph;
        let graph: &dyn GraphHandle = if m.reads == ReadScope::Minimal {
            &denied
        } else {
            self.graph
        };

        // Run the handler under a timeout and panic isolation, so a runaway,
        // blocking, or panicking handler yields a Failed outcome and the
        // dispatch of the other behaviours continues.
        let guarded = std::panic::AssertUnwindSafe(handler.run(event, graph)).catch_unwind();
        let outcome = match tokio::time::timeout(HANDLER_TIMEOUT, guarded).await {
            Err(_elapsed) => {
                return DispatchOutcome::Failed {
                    behaviour,
                    reason: "handler timed out".to_string(),
                }
            }
            Ok(Err(_panic)) => {
                return DispatchOutcome::Failed {
                    behaviour,
                    reason: "handler panicked".to_string(),
                }
            }
            Ok(Ok(Err(e))) => {
                return DispatchOutcome::Failed {
                    behaviour,
                    reason: e.to_string(),
                }
            }
            Ok(Ok(Ok(outcome))) => outcome,
        };

        match outcome {
            HandlerOutcome::Terminal(outcome) => DispatchOutcome::Terminal { behaviour, outcome },
            HandlerOutcome::Propose(action) => {
                // Correlate the gate/audit entry to this event + behaviour.
                let correlation_id = format!("{}:{}", event.id, behaviour);
                let ctx = ActionContext {
                    app_id: AGENT_APP_ID,
                    // Trusted origin fact from the event (the decoder stamps
                    // it; unknown defaults to external/true). An externally-
                    // triggered action always confirms.
                    external_trigger: event.external_content,
                    correlation_id: &correlation_id,
                };
                match self
                    .gate
                    .decide_action(&behaviour, m.mode, &m.tools, &action, &ctx, graph)
                    .await
                {
                    Ok(receipt) => {
                        let tool_scope = m.tools.get(&action.tool).map(Vec::as_slice).unwrap_or(&[]);
                        let dry_run =
                            plan_dry_run(&behaviour, &action, receipt.decision, tool_scope);
                        // If the daemon opted into executing, act on a proven
                        // decision now (re-validate + audited write). Suggest-mode
                        // (no executor) skips this and only surfaces the decision.
                        let executed = self
                            .maybe_execute(
                                &behaviour,
                                &action,
                                receipt.decision,
                                tool_scope,
                                graph,
                                &ctx,
                                m.mode,
                            )
                            .await;
                        let plan = plan_for(&action.tool);
                        DispatchOutcome::Decided {
                            behaviour,
                            action,
                            decision: receipt.decision,
                            reason: receipt.reason,
                            audit_index: receipt.audit_index,
                            plan,
                            dry_run,
                            executed,
                        }
                    }
                    Err(e) => DispatchOutcome::Refused {
                        behaviour,
                        reason: e.to_string(),
                    },
                }
            }
        }
    }

    /// Execute a gated workflow decision when the daemon opted into executing.
    /// In suggest-mode (`executor` is `None`) this returns `None`. Otherwise the
    /// executor re-validates the proof against the current graph and performs the
    /// authorised, audited write; the result is returned so it is carried on the
    /// dispatch outcome (a write failure is surfaced, not erased) and logged.
    ///
    /// NB this executes a proven `PreviewThenExecute` **immediately**, with no
    /// preview or cancellation window: the decided silent-curator semantics for a
    /// safe reversible workflow curation action (per-file prompts are annoying
    /// and these workflows cost no tokens; the user inspects results via the pull
    /// activity view). See the `executor_live` config doc; the cancellation and
    /// proof-atomicity gaps there still gate flipping the flag in a deployment.
    async fn maybe_execute(
        &self,
        behaviour: &str,
        action: &ProposedAction,
        decision: ActionDecision,
        tool_scope: &[String],
        graph: &dyn GraphHandle,
        ctx: &ActionContext<'_>,
        ceiling: BaselineMode,
    ) -> Option<ExecutionResult> {
        let executor = self.executor.as_ref()?;
        match executor
            .execute(action, decision, tool_scope, graph, behaviour, ctx, ceiling)
            .await
        {
            Ok(Some(written)) => {
                tracing::info!(
                    behaviour = %behaviour,
                    write = ?written.write,
                    outcome = ?written.outcome,
                    "live executor performed the authorised write"
                );
                Some(ExecutionResult::Written(written.outcome))
            }
            // A non-executing decision (the executor planned nothing): nothing
            // to record beyond the decision itself.
            Ok(None) => None,
            // A timed-out or post-send-failed write may have committed, so it is
            // indeterminate, not a failure: mapping it to Failed would discard the
            // created/exists signal for a write that did persist. The next run's
            // re-validation reconciles it.
            Err(
                e @ (crate::executor::ExecError::WriteTimeout
                | crate::executor::ExecError::WriteIndeterminate(_)),
            ) => {
                tracing::warn!(
                    behaviour = %behaviour,
                    "live executor write outcome unknown (may have committed): {e}"
                );
                Some(ExecutionResult::Indeterminate(e.to_string()))
            }
            Err(e) => {
                tracing::warn!(
                    behaviour = %behaviour,
                    "live executor did not write: {e}"
                );
                Some(ExecutionResult::Failed(e.to_string()))
            }
        }
    }

    /// Run a `kind: agent` behaviour's bounded loop. Each step asks the
    /// provider for one move, parses it, and (for a proposed action) passes
    /// it through the gate exactly as a workflow proposal would be. The loop
    /// is bounded three ways by the manifest [`Budget`](crate::behaviour::Budget):
    /// step count, total tokens, and wall-clock; it also ends when the model
    /// stops on a declared terminal condition. There is no "until the model
    /// decides to quit" path. Suggest-mode only: every step is gated and
    /// audited, nothing is executed.
    ///
    /// Because nothing executes yet, the only feedback between steps is the
    /// gate verdict (carried in the transcript) — there is no ground-truth
    /// observation of an action's effect, so a multi-step run is bounded
    /// reasoning over the same state, not a true predict-act-observe loop.
    /// Real observation arrives with the world model and an action executor
    /// (later increments); the budget + declared terminals bound the run
    /// until then.
    ///
    /// Returns one outcome per gated step plus a terminal outcome naming why
    /// the loop ended (the declared terminal name, `budget_steps`,
    /// `budget_tokens`, `budget_wall_ms`, or `budget_context` when the prompt
    /// cannot be compacted under the model's context window), or a single
    /// `Failed` on a provider error or an audit outage.
    async fn run_agent_loop(
        &self,
        lb: &LoadedBehaviour,
        event: &AgentEvent,
        provider: &dyn AIProvider,
        // This loop's wall-clock anchor, read by `dispatch` immediately before
        // this call so the budget is per-behaviour (an earlier behaviour's
        // runtime does not count against it).
        start: SystemTime,
    ) -> Vec<DispatchOutcome> {
        let m = &lb.behaviour.manifest;
        let behaviour = m.name.clone();

        // Self-guard via the shared eligibility check (has_provider = true: this
        // function is only reached with a provider). `dispatch` already runs the
        // same check before screening, so this never fires from there; it keeps
        // the loop correct if called another way.
        if let Some(reason) = agent_skip_reason(m, self.read_tier, true) {
            return vec![DispatchOutcome::Skipped { behaviour, reason }];
        }
        // Guaranteed present by the eligibility check above.
        let budget = m
            .budget
            .as_ref()
            .expect("agent_skip_reason ensures a budget is declared");

        // The behaviour-scoped graph the gate's predict-before-act reads
        // through (a denying handle for a `reads: minimal` behaviour), so the
        // proof never reads more than the behaviour may.
        let denied = DeniedGraph;
        let graph: &dyn GraphHandle = if m.reads == ReadScope::Minimal {
            &denied
        } else {
            self.graph
        };

        let mut outcomes = Vec::new();
        let mut transcript: Vec<TranscriptEntry> = Vec::new();
        let mut tokens_spent: u32 = 0;
        // Repetition / no-progress guard: the last step's [`ProgressKey`] and how
        // many times it has repeated. The loop neither executes nor observes
        // between steps, so an identical key means the model is stuck (a refused
        // action re-proposed, or an exact duplicate accepted suggestion) and is
        // burning the budget for nothing.
        let mut last_progress_key: Option<ProgressKey> = None;
        let mut repeated_no_progress: u32 = 0;
        // The model's input window, from the wired provider, so the
        // context-window guard tracks the real backend rather than a guess.
        let window = provider.context_window();

        for step in 0..budget.max_steps {
            // Monotonic-safe: if the wall clock moved backwards, treat the
            // budget as exhausted rather than resetting elapsed to zero (which
            // would hand each step a fresh near-full timeout). A proper
            // monotonic `Instant` seam is the cleaner long-term form.
            let elapsed_ms = match self.clock.now().duration_since(start) {
                Ok(d) => d.as_millis() as u64,
                Err(_) => u64::MAX,
            };
            if elapsed_ms >= budget.max_wall_ms {
                outcomes.push(DispatchOutcome::Terminal {
                    behaviour,
                    outcome: "budget_wall_ms".to_string(),
                });
                return outcomes;
            }
            if tokens_spent >= budget.max_tokens {
                outcomes.push(DispatchOutcome::Terminal {
                    behaviour,
                    outcome: "budget_tokens".to_string(),
                });
                return outcomes;
            }

            // Keep the working memory inside the model's context window before
            // building this step's prompt: a deterministic, model-free prune
            // (collapse redundant correction feedback) then tighten (drop the
            // rationale prose of older proposals, keeping every tool, decision,
            // and refusal verbatim). If it still will not fit, terminate closed
            // rather than send an over-window prompt or drop a load-bearing
            // fact. This makes no model call, so it spends no budget here.
            if let CompactionOutcome::OverWindow =
                compact_for_window(&lb.behaviour, event, &mut transcript, window, &self.compaction)
            {
                outcomes.push(DispatchOutcome::Terminal {
                    behaviour,
                    outcome: "budget_context".to_string(),
                });
                return outcomes;
            }

            let prompt = build_agent_prompt(&lb.behaviour, event, &transcript);
            let prompt_len = prompt.len();
            // Refuse to spend on a call whose input alone would already
            // exceed the budget: enforce the token bound *before* the call,
            // not only after, so one oversized prompt (a large skill body or
            // event payload) cannot blow past max_tokens in a single step.
            let input_estimate = estimate_tokens(None, prompt_len);
            // Output allowance = what the budget leaves after this call's
            // input. Refuse the call when there is no room for any output, so
            // one oversized prompt cannot use up the whole budget on input and
            // the advisory cap below is the genuine remaining headroom.
            let output_allowance = budget
                .max_tokens
                .saturating_sub(tokens_spent)
                .saturating_sub(input_estimate);
            if output_allowance == 0 {
                outcomes.push(DispatchOutcome::Terminal {
                    behaviour,
                    outcome: "budget_tokens".to_string(),
                });
                return outcomes;
            }
            let request = CompletionRequest {
                prompt,
                // Advisory output cap: the smaller of the remaining run budget
                // and the context window's room after input (the input measured
                // with the conservative window bound), so a provider that
                // honours `extras` keeps input+output within both budget and
                // window. The post-call accounting still enforces the budget
                // locally; hard enforcement in the adapters/proxy is a
                // provider-contract follow-up, not this increment.
                extras: serde_json::json!({
                    "max_tokens": output_window_cap(window, output_allowance, window_token_estimate(prompt_len))
                }),
            };
            // Bound the call by the wall-clock budget that remains, so a
            // stalled provider cannot hang the loop (and the daemon) past
            // max_wall_ms. `elapsed_ms < max_wall_ms` here, so this is >= 1.
            let remaining_ms = budget.max_wall_ms.saturating_sub(elapsed_ms).max(1);
            let resp = match tokio::time::timeout(
                Duration::from_millis(remaining_ms),
                provider.complete(request),
            )
            .await
            {
                Err(_elapsed) => {
                    outcomes.push(DispatchOutcome::Terminal {
                        behaviour,
                        outcome: "budget_wall_ms".to_string(),
                    });
                    return outcomes;
                }
                Ok(Err(e)) => {
                    outcomes.push(DispatchOutcome::Failed {
                        behaviour,
                        reason: format!("provider error: {e}"),
                    });
                    return outcomes;
                }
                Ok(Ok(resp)) => resp,
            };
            // Charge reported usage, falling back to a coarse length estimate
            // when a provider omits it, so the token budget always bounds the
            // run rather than being silently bypassed by `None` usage.
            tokens_spent = tokens_spent
                .saturating_add(estimate_tokens(resp.audit.input_tokens, prompt_len))
                .saturating_add(estimate_tokens(resp.audit.output_tokens, resp.text.len()));
            // Even if the input fit the budget, an over-budget *response* must
            // not drive an action: terminate before parsing or gating once the
            // total spend exceeds the budget.
            if tokens_spent > budget.max_tokens {
                outcomes.push(DispatchOutcome::Terminal {
                    behaviour,
                    outcome: "budget_tokens".to_string(),
                });
                return outcomes;
            }

            let step_action = match parse_agent_step(&resp.text) {
                Ok(s) => s,
                Err(e) => {
                    // Feed the parse failure back so the model can correct on
                    // the next step; the step budget bounds repeated failures.
                    transcript.push(TranscriptEntry::Nag {
                        step,
                        detail: format!(
                            "your response was not a valid step ({e}); reply with exactly one JSON step"
                        ),
                    });
                    continue;
                }
            };

            match step_action {
                AgentStep::Stop { terminal, .. } => {
                    // The model may only stop on a condition the behaviour
                    // declared; an unknown (or injected) terminal is rejected
                    // and fed back for correction rather than ending the loop.
                    if !m.terminal.contains_key(&terminal) {
                        transcript.push(TranscriptEntry::Nag {
                            step,
                            detail: format!(
                                "\"{terminal}\" is not a declared stop condition; stop only with one of: {}",
                                m.terminal.keys().cloned().collect::<Vec<_>>().join(", ")
                            ),
                        });
                        continue;
                    }
                    // A stop ends the loop immediately, so the note is not fed
                    // back into any later step's prompt; the declared terminal
                    // name below is what the surfacing keys off.
                    // Emit the bare declared terminal name (as a workflow
                    // handler would), so the surfacing disposition can key off
                    // it later.
                    outcomes.push(DispatchOutcome::Terminal {
                        behaviour,
                        outcome: terminal,
                    });
                    return outcomes;
                }
                AgentStep::Propose { tool, summary } => {
                    // One distinct correlation id per step, so each step's
                    // gate/audit entry is a separate, ordered ledger record.
                    let correlation_id = format!("{}:{}:step-{step}", event.id, behaviour);
                    let ctx = ActionContext {
                        app_id: AGENT_APP_ID,
                        external_trigger: event.external_content,
                        correlation_id: &correlation_id,
                    };
                    // The model does not yet propose structured operands, so a
                    // loop-proposed action carries none and can only be
                    // suggested, never proven for an execution-cap lift. Typed
                    // model operands land with the real agent behaviour.
                    let action = ProposedAction {
                        tool,
                        summary,
                        arguments: Default::default(),
                    };
                    // Capture the identity before `action` is moved into the
                    // outcome, so the no-progress guard can key on it after.
                    let tool_name = action.tool.clone();
                    let summary_text = action.summary.clone();
                    let progress_key = match self
                        .gate
                        .decide_action(&behaviour, m.mode, &m.tools, &action, &ctx, graph)
                        .await
                    {
                        Ok(receipt) => {
                            transcript.push(TranscriptEntry::Proposed {
                                step,
                                tool: action.tool.clone(),
                                summary: action.summary.clone(),
                                decision: format!("{:?}", receipt.decision),
                            });
                            let tool_scope =
                                m.tools.get(&action.tool).map(Vec::as_slice).unwrap_or(&[]);
                            let dry_run = plan_dry_run(
                                &behaviour,
                                &action,
                                receipt.decision,
                                tool_scope,
                            );
                            let plan = plan_for(&action.tool);
                            outcomes.push(DispatchOutcome::Decided {
                                behaviour: behaviour.clone(),
                                action,
                                decision: receipt.decision,
                                reason: receipt.reason,
                                audit_index: receipt.audit_index,
                                plan,
                                dry_run,
                                // The multi-step agent loop is not wired to the
                                // executor (separate design pass); it never
                                // executes here.
                                executed: None,
                            });
                            ProgressKey::Accepted(tool_name, summary_text)
                        }
                        Err(GateError::AuditUnavailable(reason)) => {
                            // The audit boundary is down: do not keep acting
                            // without a durable record. Stop the loop closed.
                            outcomes.push(DispatchOutcome::Failed {
                                behaviour,
                                reason: format!("audit unavailable: {reason}"),
                            });
                            return outcomes;
                        }
                        Err(e) => {
                            // Recoverable (e.g. a tool out of scope): record it
                            // and feed it back so the model can choose again.
                            transcript.push(TranscriptEntry::Refused {
                                step,
                                reason: e.to_string(),
                            });
                            outcomes.push(DispatchOutcome::Refused {
                                behaviour: behaviour.clone(),
                                reason: e.to_string(),
                            });
                            ProgressKey::Refused(tool_name)
                        }
                    };
                    // No-progress guard: the identical key repeated (a refused
                    // action re-proposed even reworded, or an exact duplicate
                    // accepted suggestion) is a stuck loop, so stop rather than
                    // burn the rest of the budget. The outcome above is recorded
                    // first; any different key resets the streak.
                    if last_progress_key.as_ref() == Some(&progress_key) {
                        repeated_no_progress += 1;
                    } else {
                        repeated_no_progress = 0;
                        last_progress_key = Some(progress_key);
                    }
                    if repeated_no_progress >= MAX_NO_PROGRESS_REPEATS {
                        outcomes.push(DispatchOutcome::Terminal {
                            behaviour,
                            outcome: "no_progress".to_string(),
                        });
                        return outcomes;
                    }
                }
            }
        }

        // The loop exhausted its step budget without the model stopping.
        outcomes.push(DispatchOutcome::Terminal {
            behaviour,
            outcome: "budget_steps".to_string(),
        });
        outcomes
    }

    /// Run the dispatch loop until the source is exhausted, logging each
    /// outcome. (Surfacing through the P9 shell surfaces lands later.)
    pub async fn run<S: TriggerSource>(&self, source: &mut S) {
        while let Some(event) = source.recv().await {
            for outcome in self.dispatch(&event).await {
                tracing::info!(?outcome, event = %event.event_type, "agent dispatch");
            }
        }
    }
}

/// Whether a behaviour's declared read scope is satisfied by the agent's
/// configured grant. Conservative: Minimal needs nothing, Full grants
/// everything, otherwise the tiers must match exactly. The access tiers are
/// non-nested label *lenses* (e.g. project-scoped grants `Project` but
/// time-scoped does not), so a precise superset check needs the schema and
/// belongs to the read/grounding layer; this conservative form never
/// over-grants (it may refuse a satisfiable combination, fail-safe). Exposed
/// so the daemon can compute the same dispatch eligibility before deciding
/// whether a behaviour is runnable or needs a provider wired.
pub fn reads_satisfied(needs: ReadScope, granted: AccessTier) -> bool {
    needs == ReadScope::Minimal || granted == AccessTier::Full || needs.tier() == granted
}

/// Why a `kind: agent` behaviour cannot reach the model on this dispatch, if
/// any. The single source of truth for agent eligibility: no provider (a bus
/// outage), a declared read scope the configured tier does not grant, or a
/// missing budget. `dispatch` calls it before screening so the classifier never
/// runs for an event that cannot reach the model, and `run_agent_loop` calls it
/// as its own guard, so the two never disagree. `None` means the behaviour is
/// eligible to run (provider present, read-scope satisfied, budget declared).
fn agent_skip_reason(
    m: &crate::behaviour::BehaviourManifest,
    read_tier: AccessTier,
    has_provider: bool,
) -> Option<String> {
    if !has_provider {
        return Some("no AI provider configured; agent behaviours cannot run".to_string());
    }
    if !reads_satisfied(m.reads, read_tier) {
        return Some(format!(
            "declared read scope {:?} exceeds the configured grant",
            m.reads
        ));
    }
    if m.budget.is_none() {
        return Some("agent behaviour declares no budget".to_string());
    }
    None
}

/// Tokens to charge for one side of a completion. A coarse length-based
/// estimate (~4 bytes/token) is the floor; the provider's reported count is
/// used only when it is at least the estimate. So a provider that omits
/// usage, or reports an implausibly low count, cannot bypass the token
/// budget — the estimate always applies as a lower bound.
fn estimate_tokens(reported: Option<u32>, text_len: usize) -> u32 {
    let estimate = (text_len / 4) as u32;
    reported.map_or(estimate, |r| r.max(estimate))
}

/// A deliberately conservative upper bound on the token count of `text_len`
/// bytes for the context-window guard: one byte per token. A token is at least
/// one byte in any tokenizer, so the real count never exceeds this, which
/// keeps the window check fail-closed even for token-dense input (where the
/// 4-bytes-per-token cost estimate would under-count). It over-counts ordinary
/// English by ~4x, so it errs toward compacting early; a model-accurate
/// tokenizer (a provider property) replaces it when the provider is wired and
/// reclaims the full window. Distinct from [`estimate_tokens`], which averages
/// for cost accounting rather than bounding for safety.
fn window_token_estimate(text_len: usize) -> u32 {
    u32::try_from(text_len).unwrap_or(u32::MAX)
}

/// The advisory output-token cap for a completion: the smaller of what the run
/// token budget leaves (`budget_allowance`) and what the model's context
/// `window` leaves after this call's input (`window - input_window_estimate`,
/// the input measured with the conservative window bound). Bounding by the
/// window, not just the budget, keeps a large manifest token budget from
/// requesting more output than the window can hold once the input is counted.
fn output_window_cap(window: u32, budget_allowance: u32, input_window_estimate: u32) -> u32 {
    budget_allowance.min(window.saturating_sub(input_window_estimate))
}

/// The outcome of a compaction pass: either this step's prompt fits the
/// context window, or it cannot and the loop must terminate closed. Compaction
/// makes no model call, so it has no provider/timeout/budget failure modes of
/// its own.
#[derive(Debug, PartialEq, Eq)]
enum CompactionOutcome {
    /// The step prompt fits the window (it always did, or compaction brought it
    /// under the threshold).
    Proceed,
    /// Pruning and tightening could not get the prompt under the window;
    /// terminate closed (`budget_context`) rather than send it.
    OverWindow,
}

/// Keep the loop's working memory inside the model's context `window` before a
/// step's prompt is built, deterministically and with no model call: estimate
/// the prompt with the conservative window bound, and if it is over the
/// window, prune redundant correction feedback, then tighten older proposals
/// (dropping rationale prose while keeping every tool, decision, and refusal
/// verbatim). If it still will not fit, report `OverWindow` so the caller
/// closes the loop rather than sending an over-window prompt or silently
/// dropping a load-bearing fact. `window` is the wired model's input window
/// (from the provider), so the bound tracks the real backend.
fn compact_for_window(
    behaviour: &Behaviour,
    event: &AgentEvent,
    transcript: &mut Vec<TranscriptEntry>,
    window: u32,
    policy: &CompactionPolicy,
) -> CompactionOutcome {
    let estimate = |t: &[TranscriptEntry]| {
        window_token_estimate(build_agent_prompt(behaviour, event, t).len())
    };
    if !policy.over(window, estimate(transcript)) {
        return CompactionOutcome::Proceed;
    }
    compaction::prune(transcript);
    if !policy.over(window, estimate(transcript)) {
        return CompactionOutcome::Proceed;
    }
    compaction::tighten(transcript, policy.keep_recent);
    if !policy.over(window, estimate(transcript)) {
        return CompactionOutcome::Proceed;
    }
    CompactionOutcome::OverWindow
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, VecDeque};
    use std::path::PathBuf;

    use audit_proto::MockAuditSink;
    use lunaris_ai_core::capability::{AccessTier, ActionPermissions, Capability};
    use lunaris_ai_core::provider::{CompletionResponse, ProviderAudit, ProviderError};

    use crate::behaviour::parse;
    use crate::loader::{DisableReason, LoadedBehaviour, Provenance, Status};
    use crate::seams::{GraphError, ManualClock, NullObserver, SystemClock};

    /// A fixed clock for the workflow tests, which do not exercise the
    /// wall-clock budget (the agent-loop tests use an advancing clock).
    const TEST_CLOCK: SystemClock = SystemClock;

    const AUTO_TAG: &str = r#"---
name: auto-tag-by-project
description: Tag a newly opened file with the project it belongs to.
kind: workflow
handler: auto_tag_by_project
reads: project
trigger:
  type: event
  event: file.opened
  filter: "path not_startswith ~/.cache"
tools:
  graph.write: [Project, FILE_PART_OF]
---
"#;

    /// Like `AUTO_TAG` but `mode: supervised`, so a proven decision lifts to
    /// `PreviewThenExecute` (the executing path the live executor acts on)
    /// instead of being capped to `Propose` by a Suggest ceiling.
    const LIVE_AUTO_TAG: &str = r#"---
name: auto-tag-by-project
description: Tag a newly opened file with the project it belongs to.
kind: workflow
mode: supervised
handler: auto_tag_by_project
reads: project
trigger:
  type: event
  event: file.opened
  filter: "path not_startswith ~/.cache"
tools:
  graph.write: [Project, FILE_PART_OF]
---
"#;

    /// A workflow on an event type the source decodes to no fields, to check
    /// that field-less events are never coalesced.
    const CALENDAR: &str = r#"---
name: meeting-prep-test
description: A test behaviour on an event type that decodes to no fields.
kind: workflow
handler: auto_tag_by_project
reads: minimal
trigger:
  type: event
  event: calendar.event.upcoming
tools:
  graph.write: [Project, FILE_PART_OF]
---
"#;

    fn loaded(skill: &str, status: Status) -> LoadedBehaviour {
        LoadedBehaviour {
            behaviour: parse(skill).expect("valid fixture"),
            provenance: Provenance::BuiltIn,
            dir: PathBuf::from("/test"),
            status,
        }
    }

    fn event(path: &str) -> AgentEvent {
        AgentEvent {
            id: "e1".to_string(),
            event_type: "file.opened".to_string(),
            fields: [("path".to_string(), path.to_string())].into_iter().collect(),
            external_content: false,
        }
    }

    /// An event with no decoded payload fields (the shape the source produces
    /// for an event type it does not yet decode, e.g. `calendar.event.upcoming`).
    fn fieldless_event(event_type: &str) -> AgentEvent {
        AgentEvent {
            id: "c1".to_string(),
            event_type: event_type.to_string(),
            fields: BTreeMap::new(),
            external_content: false,
        }
    }

    /// A graph that returns nothing — handlers under test do not query it.
    struct EmptyGraph;
    #[async_trait::async_trait]
    impl GraphHandle for EmptyGraph {
        async fn query(
            &self,
            _cypher: &str,
        ) -> Result<Vec<HashMap<String, serde_json::Value>>, GraphError> {
            Ok(Vec::new())
        }
    }

    struct StubPropose;
    #[async_trait::async_trait]
    impl WorkflowHandler for StubPropose {
        async fn run(
            &self,
            _event: &AgentEvent,
            _graph: &dyn GraphHandle,
        ) -> Result<HandlerOutcome, HandlerError> {
            Ok(HandlerOutcome::Propose(ProposedAction {
                tool: "graph.write".to_string(),
                summary: "tag the opened file".to_string(),
                arguments: Default::default(),
            }))
        }
    }

    struct StubTerminal;
    #[async_trait::async_trait]
    impl WorkflowHandler for StubTerminal {
        async fn run(
            &self,
            _event: &AgentEvent,
            _graph: &dyn GraphHandle,
        ) -> Result<HandlerOutcome, HandlerError> {
            Ok(HandlerOutcome::Terminal("no_matching_project".to_string()))
        }
    }

    struct StubPanic;
    #[async_trait::async_trait]
    impl WorkflowHandler for StubPanic {
        async fn run(
            &self,
            _event: &AgentEvent,
            _graph: &dyn GraphHandle,
        ) -> Result<HandlerOutcome, HandlerError> {
            panic!("handler blew up");
        }
    }

    struct VecSource(VecDeque<AgentEvent>);
    impl TriggerSource for VecSource {
        async fn recv(&mut self) -> Option<AgentEvent> {
            self.0.pop_front()
        }
    }

    fn registry(handler: Box<dyn WorkflowHandler>) -> HandlerRegistry {
        [("auto_tag_by_project".to_string(), handler)].into_iter().collect()
    }

    fn gate<'a>(audit: &'a MockAuditSink, obs: &'a NullObserver, cap: &'a Capability) -> Gate<'a> {
        // The system path/mount resolvers the gate's predict-before-act step
        // reads through. These dispatch tests propose actions with no operands,
        // so a prediction is never `Valid` and the conservative cap stands; the
        // resolvers are never meaningfully read. `static` zero-cost stand-ins
        // keep them `'static` so the gate can borrow them for any test lifetime.
        // (The graph is passed per call to `decide_action` by the dispatcher.)
        use crate::slice::{FsPathResolver, StaticMountPolicy};
        static FS: FsPathResolver = FsPathResolver;
        static MOUNTS: StaticMountPolicy = StaticMountPolicy::empty();
        Gate::new(cap, audit, obs, &FS, &MOUNTS)
    }

    fn fields_of(path: &str) -> BTreeMap<String, String> {
        [("path".to_string(), path.to_string())].into_iter().collect()
    }

    #[test]
    fn coalescer_admits_first_then_coalesces_within_window_then_readmits_after() {
        let t0 = SystemTime::UNIX_EPOCH + Duration::from_secs(1000);
        let mut c = Coalescer::new(Duration::from_secs(1));
        let foo = fields_of("~/foo.rs");
        // First sighting of (behaviour, event): admitted.
        assert!(c.admit("b", "file.opened", &foo, false, t0));
        // Same key within the window: coalesced.
        assert!(!c.admit("b", "file.opened", &foo, false, t0 + Duration::from_millis(500)));
        // A different behaviour, same event: independent, admitted.
        assert!(c.admit("other", "file.opened", &foo, false, t0 + Duration::from_millis(500)));
        // A different entity (path) for the same behaviour: not coalesced.
        assert!(c.admit("b", "file.opened", &fields_of("~/bar.rs"), false, t0 + Duration::from_millis(600)));
        // The window is measured from the first admit (t0), not extended by the
        // coalesced duplicate, so once it elapses the same key admits again.
        assert!(c.admit("b", "file.opened", &foo, false, t0 + Duration::from_millis(1001)));
    }

    #[test]
    fn coalescer_window_is_measured_from_the_first_admit_not_each_duplicate() {
        let t0 = SystemTime::UNIX_EPOCH + Duration::from_secs(1000);
        let mut c = Coalescer::new(Duration::from_secs(1));
        let foo = fields_of("~/foo.rs");
        assert!(c.admit("b", "file.opened", &foo, false, t0));
        // A duplicate near the end of the window does not push the window out.
        assert!(!c.admit("b", "file.opened", &foo, false, t0 + Duration::from_millis(900)));
        // Just past one window from the first admit: re-admitted, despite the
        // recent duplicate.
        assert!(c.admit("b", "file.opened", &foo, false, t0 + Duration::from_millis(1001)));
    }

    #[test]
    fn coalescer_treats_a_backwards_clock_entry_as_stale_and_readmits() {
        let t_high = SystemTime::UNIX_EPOCH + Duration::from_secs(1000);
        let t_low = SystemTime::UNIX_EPOCH + Duration::from_secs(100); // clock rolled back
        let mut c = Coalescer::new(Duration::from_secs(1));
        let foo = fields_of("~/foo.rs");
        assert!(c.admit("b", "file.opened", &foo, false, t_high));
        // After a rollback the prior entry is in the future relative to `now`.
        // It must be treated as stale so the same event re-admits, rather than
        // being suppressed until wall time catches back up to t_high.
        assert!(c.admit("b", "file.opened", &foo, false, t_low));
    }

    #[test]
    fn coalescer_map_stays_bounded_under_a_distinct_event_storm() {
        // Many distinct events within one window are not coalesced (each is a
        // new key), but the tracking map must not grow without bound.
        let t0 = SystemTime::UNIX_EPOCH + Duration::from_secs(1000);
        let mut c = Coalescer::new(Duration::from_secs(60)); // long window: nothing expires
        for i in 0..(MAX_COALESCE_ENTRIES * 2) {
            // Each distinct path is a new key, so every admit dispatches.
            assert!(c.admit("b", "file.opened", &fields_of(&format!("~/f{i}.rs")), false, t0));
        }
        assert!(
            c.seen.len() <= MAX_COALESCE_ENTRIES,
            "coalescer map must stay bounded under a distinct-event storm: {}",
            c.seen.len()
        );
    }

    #[tokio::test]
    async fn events_with_no_decoded_fields_are_never_coalesced() {
        // calendar.event.upcoming decodes to empty fields today, so two distinct
        // upcoming meetings share a behaviour+type+fields key. They must both
        // dispatch, never be coalesced into one (which would drop a real prep).
        let behaviours = [loaded(CALENDAR, Status::Enabled)];
        let handlers = registry(Box::new(StubPropose));
        let cap = Capability::new(AccessTier::Full, ActionPermissions::suggest_only());
        let audit = MockAuditSink::accepting();
        let obs = NullObserver;
        let graph = EmptyGraph;
        let d = Dispatcher::new(
            &behaviours,
            &handlers,
            &graph,
            AccessTier::Full,
            gate(&audit, &obs, &cap),
            None,
            &TEST_CLOCK,
        )
        .with_coalesce_window(Duration::from_secs(1));

        // Same occurrence time, identical (empty) fields: both must still dispatch.
        let first = d.dispatch(&fieldless_event("calendar.event.upcoming")).await;
        let second = d.dispatch(&fieldless_event("calendar.event.upcoming")).await;
        assert!(matches!(first.as_slice(), [DispatchOutcome::Decided { .. }]));
        assert!(
            matches!(second.as_slice(), [DispatchOutcome::Decided { .. }]),
            "an event with no decoded fields must not be coalesced: {second:?}"
        );
    }

    #[tokio::test]
    async fn dispatch_coalesces_a_burst_then_readmits_after_the_window() {
        let behaviours = [loaded(AUTO_TAG, Status::Enabled)];
        let handlers = registry(Box::new(StubPropose));
        let cap = Capability::new(AccessTier::Full, ActionPermissions::suggest_only());
        let audit = MockAuditSink::accepting();
        let obs = NullObserver;
        let graph = EmptyGraph;
        // Coalescing keys on the dispatcher's wall clock; drive it with a manual
        // clock. This is a workflow behaviour, so the loop clock is never read.
        let clock = ManualClock::new(SystemTime::UNIX_EPOCH + Duration::from_secs(1000));
        let d = Dispatcher::new(
            &behaviours,
            &handlers,
            &graph,
            AccessTier::Full,
            gate(&audit, &obs, &cap),
            None,
            &clock,
        )
        .with_coalesce_window(Duration::from_secs(1));

        // First identical event: dispatched.
        let first = d.dispatch(&event("~/Repositories/foo.rs")).await;
        assert!(matches!(first.as_slice(), [DispatchOutcome::Decided { .. }]));

        // Same event again within the window (clock not advanced): coalesced,
        // handler not re-run.
        let second = d.dispatch(&event("~/Repositories/foo.rs")).await;
        assert_eq!(
            second,
            vec![DispatchOutcome::Coalesced {
                behaviour: "auto-tag-by-project".to_string()
            }]
        );

        // A different file (different entity) is not coalesced.
        let other = d.dispatch(&event("~/Repositories/bar.rs")).await;
        assert!(matches!(other.as_slice(), [DispatchOutcome::Decided { .. }]));

        // Once the window elapses, the same file admits again.
        clock.advance(Duration::from_millis(1001));
        let later = d.dispatch(&event("~/Repositories/foo.rs")).await;
        assert!(matches!(later.as_slice(), [DispatchOutcome::Decided { .. }]));
    }

    #[tokio::test]
    async fn a_long_path_event_is_coalesced_via_the_digest() {
        // A field larger than any byte cap is still coalesced: the key is a
        // fixed-size digest, so a long producer-controlled path neither bloats
        // the map nor lets a duplicate storm bypass the dedup.
        let behaviours = [loaded(AUTO_TAG, Status::Enabled)];
        let handlers = registry(Box::new(StubPropose));
        let cap = Capability::new(AccessTier::Full, ActionPermissions::suggest_only());
        let audit = MockAuditSink::accepting();
        let obs = NullObserver;
        let graph = EmptyGraph;
        let clock = ManualClock::new(SystemTime::UNIX_EPOCH + Duration::from_secs(1000));
        let d = Dispatcher::new(
            &behaviours,
            &handlers,
            &graph,
            AccessTier::Full,
            gate(&audit, &obs, &cap),
            None,
            &clock,
        )
        .with_coalesce_window(Duration::from_secs(1));

        // A very long path, outside ~/.cache so the filter passes.
        let big = format!("~/Repositories/{}", "x".repeat(4096));
        let first = d.dispatch(&event(&big)).await;
        // Identical event within the window: coalesced via the digest.
        let second = d.dispatch(&event(&big)).await;
        assert!(matches!(first.as_slice(), [DispatchOutcome::Decided { .. }]));
        assert_eq!(
            second,
            vec![DispatchOutcome::Coalesced {
                behaviour: "auto-tag-by-project".to_string()
            }],
            "a long-path event must still be coalesced via the digest: {second:?}"
        );
    }

    #[tokio::test]
    async fn dispatches_a_matching_workflow_through_the_gate() {
        let behaviours = [loaded(AUTO_TAG, Status::Enabled)];
        let handlers = registry(Box::new(StubPropose));
        let cap = Capability::new(AccessTier::Full, ActionPermissions::suggest_only());
        let audit = MockAuditSink::accepting();
        let obs = NullObserver;
        let graph = EmptyGraph;

        let dispatcher =
            Dispatcher::new(&behaviours, &handlers, &graph, AccessTier::Full, gate(&audit, &obs, &cap), None, &TEST_CLOCK);
        let outcomes = dispatcher.dispatch(&event("~/Repositories/foo.rs")).await;

        assert_eq!(outcomes.len(), 1);
        assert_eq!(
            outcomes[0],
            DispatchOutcome::Decided {
                behaviour: "auto-tag-by-project".to_string(),
                action: ProposedAction {
                    tool: "graph.write".to_string(),
                    summary: "tag the opened file".to_string(),
                    arguments: Default::default(),
                },
                decision: ActionDecision::Propose,
                reason: DecisionReason::mode(),
                audit_index: 0,
                plan: plan_for("graph.write"),
                dry_run: None,
                executed: None,
            }
        );
        let recorded = audit.recorded().await;
        assert_eq!(recorded[0].structural.subject, "agent.auto-tag-by-project");
        assert_eq!(recorded[0].call_chain_id.as_deref(), Some("e1:auto-tag-by-project"));
    }

    #[tokio::test]
    async fn filtered_or_disabled_dispatches_to_nothing() {
        let handlers = registry(Box::new(StubPropose));
        let cap = Capability::new(AccessTier::Full, ActionPermissions::suggest_only());
        let audit = MockAuditSink::accepting();
        let obs = NullObserver;
        let graph = EmptyGraph;

        let enabled = [loaded(AUTO_TAG, Status::Enabled)];
        let d = Dispatcher::new(&enabled, &handlers, &graph, AccessTier::Full, gate(&audit, &obs, &cap), None, &TEST_CLOCK);
        // ~/.cache is excluded by the filter.
        assert!(d.dispatch(&event("~/.cache/x")).await.is_empty());

        let disabled = [loaded(AUTO_TAG, Status::Disabled(DisableReason::NotEnabledInSettings))];
        let d2 = Dispatcher::new(&disabled, &handlers, &graph, AccessTier::Full, gate(&audit, &obs, &cap), None, &TEST_CLOCK);
        assert!(d2.dispatch(&event("~/foo.rs")).await.is_empty());
    }

    #[tokio::test]
    async fn a_read_scope_above_the_grant_is_skipped() {
        // auto-tag declares reads: project; under a session-only grant it
        // must not run.
        let behaviours = [loaded(AUTO_TAG, Status::Enabled)];
        let handlers = registry(Box::new(StubPropose));
        let cap = Capability::new(AccessTier::SessionScoped, ActionPermissions::suggest_only());
        let audit = MockAuditSink::accepting();
        let obs = NullObserver;
        let graph = EmptyGraph;
        let dispatcher = Dispatcher::new(
            &behaviours,
            &handlers,
            &graph,
            AccessTier::SessionScoped,
            gate(&audit, &obs, &cap),
            None,
            &TEST_CLOCK,
        );
        let outcomes = dispatcher.dispatch(&event("~/foo.rs")).await;
        assert!(matches!(outcomes[0], DispatchOutcome::Skipped { .. }));
        assert_eq!(audit.count().await, 0);
    }

    #[tokio::test]
    async fn terminal_handler_records_a_terminal_outcome() {
        let behaviours = [loaded(AUTO_TAG, Status::Enabled)];
        let handlers = registry(Box::new(StubTerminal));
        let cap = Capability::new(AccessTier::Full, ActionPermissions::suggest_only());
        let audit = MockAuditSink::accepting();
        let obs = NullObserver;
        let graph = EmptyGraph;
        let dispatcher =
            Dispatcher::new(&behaviours, &handlers, &graph, AccessTier::Full, gate(&audit, &obs, &cap), None, &TEST_CLOCK);

        let outcomes = dispatcher.dispatch(&event("~/foo.rs")).await;
        assert_eq!(
            outcomes[0],
            DispatchOutcome::Terminal {
                behaviour: "auto-tag-by-project".to_string(),
                outcome: "no_matching_project".to_string(),
            }
        );
        assert_eq!(audit.count().await, 0);
    }

    #[tokio::test]
    async fn unregistered_handler_is_skipped_not_run() {
        let behaviours = [loaded(AUTO_TAG, Status::Enabled)];
        let handlers: HandlerRegistry = BTreeMap::new();
        let cap = Capability::new(AccessTier::Full, ActionPermissions::suggest_only());
        let audit = MockAuditSink::accepting();
        let obs = NullObserver;
        let graph = EmptyGraph;
        let dispatcher =
            Dispatcher::new(&behaviours, &handlers, &graph, AccessTier::Full, gate(&audit, &obs, &cap), None, &TEST_CLOCK);

        let outcomes = dispatcher.dispatch(&event("~/foo.rs")).await;
        assert!(matches!(outcomes[0], DispatchOutcome::Skipped { .. }));
        assert_eq!(audit.count().await, 0);
    }

    #[tokio::test]
    async fn run_loop_drains_the_source() {
        let behaviours = [loaded(AUTO_TAG, Status::Enabled)];
        let handlers = registry(Box::new(StubPropose));
        let cap = Capability::new(AccessTier::Full, ActionPermissions::suggest_only());
        let audit = MockAuditSink::accepting();
        let obs = NullObserver;
        let graph = EmptyGraph;
        let dispatcher =
            Dispatcher::new(&behaviours, &handlers, &graph, AccessTier::Full, gate(&audit, &obs, &cap), None, &TEST_CLOCK);

        let mut source = VecSource(VecDeque::from([event("~/a.rs"), event("~/b.rs")]));
        dispatcher.run(&mut source).await;
        assert_eq!(audit.count().await, 2);
    }

    #[tokio::test]
    async fn external_content_event_is_gated_to_confirmation() {
        let behaviours = [loaded(AUTO_TAG, Status::Enabled)];
        let handlers = registry(Box::new(StubPropose));
        let cap = Capability::new(AccessTier::Full, ActionPermissions::suggest_only());
        let audit = MockAuditSink::accepting();
        let obs = NullObserver;
        let graph = EmptyGraph;
        let dispatcher =
            Dispatcher::new(&behaviours, &handlers, &graph, AccessTier::Full, gate(&audit, &obs, &cap), None, &TEST_CLOCK);

        let mut ev = event("~/foo.rs");
        ev.external_content = true;
        let outcomes = dispatcher.dispatch(&ev).await;
        assert_eq!(
            outcomes[0],
            DispatchOutcome::Decided {
                behaviour: "auto-tag-by-project".to_string(),
                action: ProposedAction {
                    tool: "graph.write".to_string(),
                    summary: "tag the opened file".to_string(),
                    arguments: Default::default(),
                },
                decision: ActionDecision::RequireConfirmation,
                reason: DecisionReason::external(),
                audit_index: 0,
                plan: plan_for("graph.write"),
                dry_run: None,
                executed: None,
            }
        );
    }

    #[tokio::test]
    async fn a_failing_or_panicking_handler_is_isolated() {
        let behaviours = [loaded(AUTO_TAG, Status::Enabled)];
        let handlers = registry(Box::new(StubPanic));
        let cap = Capability::new(AccessTier::Full, ActionPermissions::suggest_only());
        let audit = MockAuditSink::accepting();
        let obs = NullObserver;
        let graph = EmptyGraph;
        let dispatcher =
            Dispatcher::new(&behaviours, &handlers, &graph, AccessTier::Full, gate(&audit, &obs, &cap), None, &TEST_CLOCK);

        let outcomes = dispatcher.dispatch(&event("~/foo.rs")).await;
        assert!(matches!(outcomes[0], DispatchOutcome::Failed { .. }));
        assert_eq!(audit.count().await, 0);
    }

    const MINIMAL_PROBE: &str = r#"---
name: probe-graph
description: A minimal-reads probe behaviour.
kind: workflow
handler: auto_tag_by_project
reads: minimal
trigger:
  type: event
  event: file.opened
---
"#;

    // A handler that probes the graph and reports whether the read worked.
    struct StubQuery;
    #[async_trait::async_trait]
    impl WorkflowHandler for StubQuery {
        async fn run(
            &self,
            _event: &AgentEvent,
            graph: &dyn GraphHandle,
        ) -> Result<HandlerOutcome, HandlerError> {
            let outcome = match graph.query("MATCH (p:Project) RETURN p").await {
                Ok(_) => "queried",
                Err(_) => "denied",
            };
            Ok(HandlerOutcome::Terminal(outcome.to_string()))
        }
    }

    #[tokio::test]
    async fn minimal_reads_behaviour_is_denied_graph_access() {
        let handlers = registry(Box::new(StubQuery));
        let cap = Capability::new(AccessTier::Full, ActionPermissions::suggest_only());
        let audit = MockAuditSink::accepting();
        let obs = NullObserver;
        let graph = EmptyGraph; // would answer Ok(empty) if the handler reached it

        // A minimal-reads behaviour gets a denying handle: its query fails.
        let minimal = [loaded(MINIMAL_PROBE, Status::Enabled)];
        let d = Dispatcher::new(&minimal, &handlers, &graph, AccessTier::Full, gate(&audit, &obs, &cap), None, &TEST_CLOCK);
        assert_eq!(
            d.dispatch(&event("~/foo.rs")).await[0],
            DispatchOutcome::Terminal {
                behaviour: "probe-graph".to_string(),
                outcome: "denied".to_string(),
            }
        );

        // A project-reads behaviour reaches the real graph (here EmptyGraph).
        let project = [loaded(AUTO_TAG, Status::Enabled)];
        let d2 = Dispatcher::new(&project, &handlers, &graph, AccessTier::Full, gate(&audit, &obs, &cap), None, &TEST_CLOCK);
        assert_eq!(
            d2.dispatch(&event("~/foo.rs")).await[0],
            DispatchOutcome::Terminal {
                behaviour: "auto-tag-by-project".to_string(),
                outcome: "queried".to_string(),
            }
        );
    }

    // --- Agent loop (kind: agent) ---

    fn agent_skill(max_steps: u32, max_tokens: u32, max_wall_ms: u64) -> String {
        format!(
            "---\nname: demo-agent\ndescription: do a couple of things\nkind: agent\n\
             trigger:\n  type: event\n  event: file.opened\nreads: minimal\n\
             tools:\n  graph.write: []\nbudget:\n  max_steps: {max_steps}\n  \
             max_tokens: {max_tokens}\n  max_wall_ms: {max_wall_ms}\nterminal:\n  \
             done: silent\n---\nbody\n"
        )
    }

    /// A provider that replays scripted responses in order, billing a fixed
    /// token count per call so the token budget is deterministic.
    struct MockProvider {
        responses: std::sync::Mutex<VecDeque<Result<String, ProviderError>>>,
        tokens_per_call: u32,
        report_usage: bool,
        context_window: u32,
    }
    impl MockProvider {
        fn new(responses: Vec<Result<String, ProviderError>>) -> Self {
            Self {
                responses: std::sync::Mutex::new(responses.into()),
                tokens_per_call: 20,
                report_usage: true,
                context_window: 8_192,
            }
        }
        /// A provider that omits token usage (`None`), to check the loop's
        /// length-estimate fallback still bounds the token budget.
        fn without_usage(responses: Vec<Result<String, ProviderError>>) -> Self {
            Self {
                report_usage: false,
                ..Self::new(responses)
            }
        }
        /// Bill `n` tokens per call (split input/output), so a test can drive
        /// post-call accumulation to the budget independent of prompt size.
        fn with_tokens_per_call(mut self, n: u32) -> Self {
            self.tokens_per_call = n;
            self
        }
        /// Report a specific context window, to drive the compaction guard.
        fn with_context_window(mut self, w: u32) -> Self {
            self.context_window = w;
            self
        }
    }
    #[async_trait::async_trait]
    impl AIProvider for MockProvider {
        async fn complete(
            &self,
            _req: CompletionRequest,
        ) -> Result<CompletionResponse, ProviderError> {
            let next = self.responses.lock().unwrap().pop_front();
            let (input_tokens, output_tokens) = if self.report_usage {
                (Some(self.tokens_per_call / 2), Some(self.tokens_per_call / 2))
            } else {
                (None, None)
            };
            match next {
                Some(Ok(text)) => Ok(CompletionResponse {
                    text,
                    audit: ProviderAudit {
                        provider_name: "mock".to_string(),
                        model: "mock".to_string(),
                        input_tokens,
                        output_tokens,
                    },
                }),
                Some(Err(e)) => Err(e),
                None => Err(ProviderError::Internal("mock script exhausted".to_string())),
            }
        }
        async fn available(&self) -> bool {
            true
        }
        fn name(&self) -> &str {
            "mock"
        }
        fn context_window(&self) -> u32 {
            self.context_window
        }
    }

    /// A provider whose call never resolves, to check the wall-clock budget
    /// bounds a stalled provider rather than hanging the loop.
    struct StallProvider;
    #[async_trait::async_trait]
    impl AIProvider for StallProvider {
        async fn complete(
            &self,
            _req: CompletionRequest,
        ) -> Result<CompletionResponse, ProviderError> {
            std::future::pending().await
        }
        async fn available(&self) -> bool {
            true
        }
        fn name(&self) -> &str {
            "stall"
        }
    }

    /// A clock that advances by a fixed delta on every `now()` call, so the
    /// wall-clock budget can be exhausted deterministically.
    struct AdvancingClock {
        now: std::sync::Mutex<std::time::SystemTime>,
        delta: Duration,
    }
    impl AdvancingClock {
        fn new(delta_ms: u64) -> Self {
            Self {
                now: std::sync::Mutex::new(std::time::UNIX_EPOCH),
                delta: Duration::from_millis(delta_ms),
            }
        }
    }
    impl Clock for AdvancingClock {
        fn now(&self) -> std::time::SystemTime {
            let mut t = self.now.lock().unwrap();
            let cur = *t;
            *t += self.delta;
            cur
        }
    }

    fn propose(tool: &str) -> Result<String, ProviderError> {
        Ok(format!(
            "{{\"action\":\"propose\",\"tool\":\"{tool}\",\"summary\":\"do {tool}\"}}"
        ))
    }
    fn propose_summary(tool: &str, summary: &str) -> Result<String, ProviderError> {
        Ok(format!(
            "{{\"action\":\"propose\",\"tool\":\"{tool}\",\"summary\":\"{summary}\"}}"
        ))
    }
    fn stop() -> Result<String, ProviderError> {
        Ok("{\"action\":\"stop\",\"terminal\":\"done\",\"note\":\"finished\"}".to_string())
    }

    #[tokio::test]
    async fn agent_loop_runs_steps_through_the_gate_until_stop() {
        let behaviours = [loaded(&agent_skill(5, 1_000_000, 60_000), Status::Enabled)];
        let handlers = registry(Box::new(StubPropose));
        let cap = Capability::new(AccessTier::Full, ActionPermissions::suggest_only());
        let audit = MockAuditSink::accepting();
        let obs = NullObserver;
        let graph = EmptyGraph;
        let provider = MockProvider::new(vec![propose("graph.write"), propose("graph.write"), stop()]);
        let d = Dispatcher::new(
            &behaviours,
            &handlers,
            &graph,
            AccessTier::Full,
            gate(&audit, &obs, &cap),
            Some(&provider),
            &TEST_CLOCK,
        );

        let outcomes = d.dispatch(&event("~/foo.rs")).await;
        assert_eq!(outcomes.len(), 3);
        assert!(matches!(outcomes[0], DispatchOutcome::Decided { .. }));
        assert!(matches!(outcomes[1], DispatchOutcome::Decided { .. }));
        assert!(matches!(
            &outcomes[2],
            DispatchOutcome::Terminal { outcome, .. } if outcome == "done"
        ));

        // Each gated step is a distinct, ordered ledger record.
        let recorded = audit.recorded().await;
        assert_eq!(recorded.len(), 2);
        assert_eq!(recorded[0].call_chain_id.as_deref(), Some("e1:demo-agent:step-0"));
        assert_eq!(recorded[1].call_chain_id.as_deref(), Some("e1:demo-agent:step-1"));
    }

    // --- S17 injection screening of external content ---

    use lunaris_ai_classifier::{ClassifierError, InjectionScore};

    /// A classifier that returns a fixed injection probability for any input.
    struct FixedClassifier(f32);
    impl InjectionClassifier for FixedClassifier {
        fn score(&self, _text: &str) -> Result<InjectionScore, ClassifierError> {
            Ok(InjectionScore::new(self.0))
        }
    }

    /// A classifier that always errors, to check `screen` fails closed (Block).
    struct FailingClassifier;
    impl InjectionClassifier for FailingClassifier {
        fn score(&self, _text: &str) -> Result<InjectionScore, ClassifierError> {
            Err(ClassifierError::Unavailable("test".to_string()))
        }
    }

    /// A classifier that panics, to check the blocking-task join error fails
    /// closed (the same arm a scoring timeout takes).
    struct PanickingClassifier;
    impl InjectionClassifier for PanickingClassifier {
        fn score(&self, _text: &str) -> Result<InjectionScore, ClassifierError> {
            panic!("classifier panic");
        }
    }

    fn external_event(path: &str) -> AgentEvent {
        AgentEvent {
            external_content: true,
            ..event(path)
        }
    }

    #[tokio::test]
    async fn external_content_scored_as_injection_blocks_the_agent_loop() {
        let behaviours = [loaded(&agent_skill(5, 1_000_000, 60_000), Status::Enabled)];
        let handlers = registry(Box::new(StubPropose));
        let cap = Capability::new(AccessTier::Full, ActionPermissions::suggest_only());
        let audit = MockAuditSink::accepting();
        let obs = NullObserver;
        let graph = EmptyGraph;
        // The provider would step the loop if reached; it must not be.
        let provider = MockProvider::new(vec![propose("graph.write"), stop()]);
        let blocking: Arc<dyn InjectionClassifier> = Arc::new(FixedClassifier(0.99));
        let d = Dispatcher::new(
            &behaviours,
            &handlers,
            &graph,
            AccessTier::Full,
            gate(&audit, &obs, &cap),
            Some(&provider),
            &TEST_CLOCK,
        )
        .with_screening(Arc::clone(&blocking), ClassifierPolicy::default());

        let outcomes = d.dispatch(&external_event("~/evil.rs")).await;
        assert_eq!(outcomes.len(), 1);
        assert!(matches!(outcomes[0], DispatchOutcome::Blocked { .. }));
        // The model was never called, so no step was gated or audited.
        assert_eq!(audit.count().await, 0);
    }

    #[tokio::test]
    async fn benign_external_content_runs_the_loop() {
        let behaviours = [loaded(&agent_skill(5, 1_000_000, 60_000), Status::Enabled)];
        let handlers = registry(Box::new(StubPropose));
        let cap = Capability::new(AccessTier::Full, ActionPermissions::suggest_only());
        let audit = MockAuditSink::accepting();
        let obs = NullObserver;
        let graph = EmptyGraph;
        let provider = MockProvider::new(vec![propose("graph.write"), stop()]);
        let benign: Arc<dyn InjectionClassifier> = Arc::new(FixedClassifier(0.01));
        let d = Dispatcher::new(
            &behaviours,
            &handlers,
            &graph,
            AccessTier::Full,
            gate(&audit, &obs, &cap),
            Some(&provider),
            &TEST_CLOCK,
        )
        .with_screening(Arc::clone(&benign), ClassifierPolicy::default());

        let outcomes = d.dispatch(&external_event("~/notes.rs")).await;
        // Allowed content runs the loop (a decided step then a terminal stop).
        assert!(outcomes.iter().any(|o| matches!(o, DispatchOutcome::Decided { .. })));
        assert!(!outcomes.iter().any(|o| matches!(o, DispatchOutcome::Blocked { .. })));
    }

    #[tokio::test]
    async fn warn_scored_external_content_passes_to_the_model() {
        // A score between warn_at (0.5) and block_at (0.9) is Warn: per the
        // classifier policy it is suspicious but passes (the signal is logged,
        // and any resulting action is gated to confirmation). It must not block.
        let behaviours = [loaded(&agent_skill(5, 1_000_000, 60_000), Status::Enabled)];
        let handlers = registry(Box::new(StubPropose));
        let cap = Capability::new(AccessTier::Full, ActionPermissions::suggest_only());
        let audit = MockAuditSink::accepting();
        let obs = NullObserver;
        let graph = EmptyGraph;
        let provider = MockProvider::new(vec![propose("graph.write"), stop()]);
        let warn: Arc<dyn InjectionClassifier> = Arc::new(FixedClassifier(0.7));
        let d = Dispatcher::new(
            &behaviours,
            &handlers,
            &graph,
            AccessTier::Full,
            gate(&audit, &obs, &cap),
            Some(&provider),
            &TEST_CLOCK,
        )
        .with_screening(warn, ClassifierPolicy::default());

        let outcomes = d.dispatch(&external_event("~/notes.rs")).await;
        assert!(outcomes.iter().any(|o| matches!(o, DispatchOutcome::Decided { .. })));
        assert!(!outcomes.iter().any(|o| matches!(o, DispatchOutcome::Blocked { .. })));
    }

    #[tokio::test]
    async fn a_failing_classifier_blocks_external_content_fail_closed() {
        let behaviours = [loaded(&agent_skill(5, 1_000_000, 60_000), Status::Enabled)];
        let handlers = registry(Box::new(StubPropose));
        let cap = Capability::new(AccessTier::Full, ActionPermissions::suggest_only());
        let audit = MockAuditSink::accepting();
        let obs = NullObserver;
        let graph = EmptyGraph;
        let provider = MockProvider::new(vec![propose("graph.write"), stop()]);
        let d = Dispatcher::new(
            &behaviours,
            &handlers,
            &graph,
            AccessTier::Full,
            gate(&audit, &obs, &cap),
            Some(&provider),
            &TEST_CLOCK,
        )
        .with_screening(Arc::new(FailingClassifier), ClassifierPolicy::default());

        let outcomes = d.dispatch(&external_event("~/x.rs")).await;
        assert!(matches!(outcomes[0], DispatchOutcome::Blocked { .. }));
        assert_eq!(audit.count().await, 0);
    }

    /// A classifier that records how many times it is asked to score, to prove
    /// screening does not run for events that cannot reach the model.
    struct CountingClassifier {
        calls: std::sync::atomic::AtomicUsize,
        score: f32,
    }
    impl InjectionClassifier for CountingClassifier {
        fn score(&self, _text: &str) -> Result<InjectionScore, ClassifierError> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(InjectionScore::new(self.score))
        }
    }

    #[tokio::test]
    async fn screening_does_not_run_when_no_provider_is_available() {
        // Provider outage: an external agent event is Skipped (no provider)
        // *before* screening, so the classifier is never consulted (a degraded
        // provider must not become a per-event classifier CPU cost).
        let behaviours = [loaded(&agent_skill(5, 1_000_000, 60_000), Status::Enabled)];
        let handlers = registry(Box::new(StubPropose));
        let cap = Capability::new(AccessTier::Full, ActionPermissions::suggest_only());
        let audit = MockAuditSink::accepting();
        let obs = NullObserver;
        let graph = EmptyGraph;
        let counting = Arc::new(CountingClassifier {
            calls: std::sync::atomic::AtomicUsize::new(0),
            score: 0.99, // would Block if ever consulted
        });
        let handle: Arc<dyn InjectionClassifier> = counting.clone();
        let d = Dispatcher::new(
            &behaviours,
            &handlers,
            &graph,
            AccessTier::Full,
            gate(&audit, &obs, &cap),
            None, // no provider
            &TEST_CLOCK,
        )
        .with_screening(handle, ClassifierPolicy::default());

        let outcomes = d.dispatch(&external_event("~/x.rs")).await;
        // Skipped (no provider), not Blocked: screening was gated behind the
        // eligibility preflight.
        assert!(matches!(outcomes[0], DispatchOutcome::Skipped { .. }));
        assert_eq!(counting.calls.load(std::sync::atomic::Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn fail_closed_mode_blocks_external_content_without_a_classifier() {
        // A configured-but-unavailable classifier (FailClosed): external-content
        // agent loops are blocked rather than silently unscreened.
        let behaviours = [loaded(&agent_skill(5, 1_000_000, 60_000), Status::Enabled)];
        let handlers = registry(Box::new(StubPropose));
        let cap = Capability::new(AccessTier::Full, ActionPermissions::suggest_only());
        let audit = MockAuditSink::accepting();
        let obs = NullObserver;
        let graph = EmptyGraph;
        let provider = MockProvider::new(vec![propose("graph.write"), stop()]);
        let d = Dispatcher::new(
            &behaviours,
            &handlers,
            &graph,
            AccessTier::Full,
            gate(&audit, &obs, &cap),
            Some(&provider),
            &TEST_CLOCK,
        )
        .with_screening_mode(ScreeningMode::FailClosed);

        // External content is blocked...
        let outcomes = d.dispatch(&external_event("~/x.rs")).await;
        assert!(matches!(outcomes[0], DispatchOutcome::Blocked { .. }));
        // ...but non-external content still runs (fail-closed gates external only).
        let outcomes = d.dispatch(&event("~/foo.rs")).await;
        assert!(outcomes.iter().any(|o| matches!(o, DispatchOutcome::Decided { .. })));
    }

    #[tokio::test]
    async fn oversized_external_content_is_blocked_without_running_the_classifier() {
        // A benign-scoring classifier would Allow, but an external payload past
        // the screen cap is blocked fail-closed (bounding inference work).
        let behaviours = [loaded(&agent_skill(5, 1_000_000, 60_000), Status::Enabled)];
        let handlers = registry(Box::new(StubPropose));
        let cap = Capability::new(AccessTier::Full, ActionPermissions::suggest_only());
        let audit = MockAuditSink::accepting();
        let obs = NullObserver;
        let graph = EmptyGraph;
        let provider = MockProvider::new(vec![propose("graph.write"), stop()]);
        let benign: Arc<dyn InjectionClassifier> = Arc::new(FixedClassifier(0.0));
        let d = Dispatcher::new(
            &behaviours,
            &handlers,
            &graph,
            AccessTier::Full,
            gate(&audit, &obs, &cap),
            Some(&provider),
            &TEST_CLOCK,
        )
        .with_screening(Arc::clone(&benign), ClassifierPolicy::default());

        let huge = "a".repeat(MAX_SCREEN_BYTES + 1);
        let outcomes = d.dispatch(&external_event(&huge)).await;
        assert!(matches!(outcomes[0], DispatchOutcome::Blocked { .. }));
    }

    #[tokio::test]
    async fn a_held_scorer_permit_blocks_further_external_content_without_a_new_scorer() {
        // A shared, process-lived gate with its single permit already held (a
        // scorer wedged in a prior dispatcher epoch). A dispatcher built with the
        // same gate must fail closed (Block) on external content, with a benign
        // classifier that would otherwise Allow, and without consulting it (no
        // second blocking scorer spawned). This is the across-rebuild bound.
        let behaviours = [loaded(&agent_skill(5, 1_000_000, 60_000), Status::Enabled)];
        let handlers = registry(Box::new(StubPropose));
        let cap = Capability::new(AccessTier::Full, ActionPermissions::suggest_only());
        let audit = MockAuditSink::accepting();
        let obs = NullObserver;
        let graph = EmptyGraph;
        let provider = MockProvider::new(vec![propose("graph.write"), stop()]);

        let screen_gate = Arc::new(tokio::sync::Semaphore::new(1));
        let _held = Arc::clone(&screen_gate).try_acquire_owned().unwrap(); // wedged scorer
        let counting = Arc::new(CountingClassifier {
            calls: std::sync::atomic::AtomicUsize::new(0),
            score: 0.0, // would Allow if ever consulted
        });
        let handle: Arc<dyn InjectionClassifier> = counting.clone();
        let d = Dispatcher::new(
            &behaviours,
            &handlers,
            &graph,
            AccessTier::Full,
            gate(&audit, &obs, &cap),
            Some(&provider),
            &TEST_CLOCK,
        )
        .with_screening(handle, ClassifierPolicy::default())
        .with_screen_gate(Arc::clone(&screen_gate));

        let outcomes = d.dispatch(&external_event("~/x.rs")).await;
        assert!(matches!(outcomes[0], DispatchOutcome::Blocked { .. }));
        // The classifier was never consulted: the gate blocked before scoring.
        assert_eq!(counting.calls.load(std::sync::atomic::Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn a_panicking_classifier_blocks_external_content_fail_closed() {
        // A panic in the blocking scoring task surfaces as a join error, which
        // takes the same fail-closed arm as a scoring timeout: block.
        let behaviours = [loaded(&agent_skill(5, 1_000_000, 60_000), Status::Enabled)];
        let handlers = registry(Box::new(StubPropose));
        let cap = Capability::new(AccessTier::Full, ActionPermissions::suggest_only());
        let audit = MockAuditSink::accepting();
        let obs = NullObserver;
        let graph = EmptyGraph;
        let provider = MockProvider::new(vec![propose("graph.write"), stop()]);
        let d = Dispatcher::new(
            &behaviours,
            &handlers,
            &graph,
            AccessTier::Full,
            gate(&audit, &obs, &cap),
            Some(&provider),
            &TEST_CLOCK,
        )
        .with_screening(Arc::new(PanickingClassifier), ClassifierPolicy::default());

        let outcomes = d.dispatch(&external_event("~/x.rs")).await;
        assert!(matches!(outcomes[0], DispatchOutcome::Blocked { .. }));
        assert_eq!(audit.count().await, 0);
    }

    #[tokio::test]
    async fn non_external_content_is_not_screened() {
        // A blocking classifier must not touch content that is not external:
        // internal/graph-originated triggers are not subject to the injection
        // screen.
        let behaviours = [loaded(&agent_skill(5, 1_000_000, 60_000), Status::Enabled)];
        let handlers = registry(Box::new(StubPropose));
        let cap = Capability::new(AccessTier::Full, ActionPermissions::suggest_only());
        let audit = MockAuditSink::accepting();
        let obs = NullObserver;
        let graph = EmptyGraph;
        let provider = MockProvider::new(vec![propose("graph.write"), stop()]);
        let blocking: Arc<dyn InjectionClassifier> = Arc::new(FixedClassifier(0.99));
        let d = Dispatcher::new(
            &behaviours,
            &handlers,
            &graph,
            AccessTier::Full,
            gate(&audit, &obs, &cap),
            Some(&provider),
            &TEST_CLOCK,
        )
        .with_screening(Arc::clone(&blocking), ClassifierPolicy::default());

        // event() is non-external; the loop runs despite the blocking classifier.
        let outcomes = d.dispatch(&event("~/foo.rs")).await;
        assert!(outcomes.iter().any(|o| matches!(o, DispatchOutcome::Decided { .. })));
        assert!(!outcomes.iter().any(|o| matches!(o, DispatchOutcome::Blocked { .. })));
    }

    #[tokio::test]
    async fn agent_loop_stops_on_repeatedly_refused_proposals() {
        // A high step budget, but the model keeps proposing a tool outside the
        // behaviour's declared scope (graph.write), so the gate refuses each one.
        // The no-progress guard must end the run well before the budget, on the
        // third refusal of the same tool.
        let behaviours = [loaded(&agent_skill(10, 1_000_000, 60_000), Status::Enabled)];
        let handlers = registry(Box::new(StubPropose));
        let cap = Capability::new(AccessTier::Full, ActionPermissions::suggest_only());
        let audit = MockAuditSink::accepting();
        let obs = NullObserver;
        let graph = EmptyGraph;
        let provider = MockProvider::new(vec![
            propose("fs.delete"), // out of scope -> refused
            propose("fs.delete"),
            propose("fs.delete"),
            propose("fs.delete"),
        ]);
        let d = Dispatcher::new(
            &behaviours,
            &handlers,
            &graph,
            AccessTier::Full,
            gate(&audit, &obs, &cap),
            Some(&provider),
            &TEST_CLOCK,
        );

        let outcomes = d.dispatch(&event("~/foo.rs")).await;
        assert!(outcomes.len() < 10, "stopped before the step budget");
        assert!(matches!(
            outcomes.last().unwrap(),
            DispatchOutcome::Terminal { outcome, .. } if outcome == "no_progress"
        ));
        // Each refusal is recorded before the run stops.
        assert_eq!(
            outcomes.iter().filter(|o| matches!(o, DispatchOutcome::Refused { .. })).count(),
            3
        );
    }

    #[tokio::test]
    async fn no_progress_guard_resists_reworded_refusals() {
        // The same out-of-scope tool refused every step but with a different
        // summary each time: the guard keys on the refused tool, not the
        // rationale, so re-wording cannot dodge it.
        let behaviours = [loaded(&agent_skill(10, 1_000_000, 60_000), Status::Enabled)];
        let handlers = registry(Box::new(StubPropose));
        let cap = Capability::new(AccessTier::Full, ActionPermissions::suggest_only());
        let audit = MockAuditSink::accepting();
        let obs = NullObserver;
        let graph = EmptyGraph;
        let provider = MockProvider::new(vec![
            propose_summary("fs.delete", "remove it"),
            propose_summary("fs.delete", "please remove the file"),
            propose_summary("fs.delete", "delete now"),
            propose_summary("fs.delete", "rm"),
        ]);
        let d = Dispatcher::new(
            &behaviours,
            &handlers,
            &graph,
            AccessTier::Full,
            gate(&audit, &obs, &cap),
            Some(&provider),
            &TEST_CLOCK,
        );

        let outcomes = d.dispatch(&event("~/foo.rs")).await;
        assert!(matches!(
            outcomes.last().unwrap(),
            DispatchOutcome::Terminal { outcome, .. } if outcome == "no_progress"
        ));
    }

    #[tokio::test]
    async fn identical_accepted_proposals_stop_for_no_progress() {
        // The model re-proposes the EXACT same in-scope action (same tool and
        // summary) every step. Nothing executes or is observed between steps, so
        // it is stuck; the guard stops it before the budget even though each
        // proposal is gated/accepted.
        let behaviours = [loaded(&agent_skill(10, 1_000_000, 60_000), Status::Enabled)];
        let handlers = registry(Box::new(StubPropose));
        let cap = Capability::new(AccessTier::Full, ActionPermissions::suggest_only());
        let audit = MockAuditSink::accepting();
        let obs = NullObserver;
        let graph = EmptyGraph;
        // `propose` uses a fixed summary, so these are exact duplicates.
        let provider = MockProvider::new(vec![
            propose("graph.write"),
            propose("graph.write"),
            propose("graph.write"),
            propose("graph.write"),
        ]);
        let d = Dispatcher::new(
            &behaviours,
            &handlers,
            &graph,
            AccessTier::Full,
            gate(&audit, &obs, &cap),
            Some(&provider),
            &TEST_CLOCK,
        );

        let outcomes = d.dispatch(&event("~/foo.rs")).await;
        assert!(outcomes.len() < 10, "stopped before the step budget");
        assert!(matches!(
            outcomes.last().unwrap(),
            DispatchOutcome::Terminal { outcome, .. } if outcome == "no_progress"
        ));
    }

    #[tokio::test]
    async fn repeated_same_tool_decisions_are_progress_not_no_progress() {
        // The model proposes the same IN-SCOPE tool several times (legitimately
        // different work, gated each time). A gate decision is progress, so the
        // no-progress guard must NOT trip; the run ends on the model's stop.
        let behaviours = [loaded(&agent_skill(10, 1_000_000, 60_000), Status::Enabled)];
        let handlers = registry(Box::new(StubPropose));
        let cap = Capability::new(AccessTier::Full, ActionPermissions::suggest_only());
        let audit = MockAuditSink::accepting();
        let obs = NullObserver;
        let graph = EmptyGraph;
        let provider = MockProvider::new(vec![
            propose_summary("graph.write", "tag file a"),
            propose_summary("graph.write", "tag file b"),
            propose_summary("graph.write", "tag file c"),
            stop(),
        ]);
        let d = Dispatcher::new(
            &behaviours,
            &handlers,
            &graph,
            AccessTier::Full,
            gate(&audit, &obs, &cap),
            Some(&provider),
            &TEST_CLOCK,
        );

        let outcomes = d.dispatch(&event("~/foo.rs")).await;
        assert!(
            !outcomes.iter().any(|o| matches!(o, DispatchOutcome::Terminal { outcome, .. } if outcome == "no_progress")),
            "legitimate same-tool work must not trip the no-progress guard"
        );
        assert_eq!(
            outcomes.iter().filter(|o| matches!(o, DispatchOutcome::Decided { .. })).count(),
            3
        );
        assert!(matches!(
            outcomes.last().unwrap(),
            DispatchOutcome::Terminal { outcome, .. } if outcome == "done"
        ));
    }

    #[tokio::test]
    async fn agent_loop_stops_at_the_step_budget() {
        let behaviours = [loaded(&agent_skill(2, 1_000_000, 60_000), Status::Enabled)];
        let handlers = registry(Box::new(StubPropose));
        let cap = Capability::new(AccessTier::Full, ActionPermissions::suggest_only());
        let audit = MockAuditSink::accepting();
        let obs = NullObserver;
        let graph = EmptyGraph;
        // Never stops on its own; the step budget must end it.
        let provider =
            MockProvider::new(vec![propose("graph.write"), propose("graph.write"), propose("graph.write")]);
        let d = Dispatcher::new(
            &behaviours,
            &handlers,
            &graph,
            AccessTier::Full,
            gate(&audit, &obs, &cap),
            Some(&provider),
            &TEST_CLOCK,
        );

        let outcomes = d.dispatch(&event("~/foo.rs")).await;
        assert_eq!(outcomes.len(), 3);
        assert!(matches!(outcomes[0], DispatchOutcome::Decided { .. }));
        assert!(matches!(outcomes[1], DispatchOutcome::Decided { .. }));
        assert!(matches!(
            &outcomes[2],
            DispatchOutcome::Terminal { outcome, .. } if outcome == "budget_steps"
        ));
    }

    #[test]
    fn token_estimate_floors_reported_usage_at_the_length_estimate() {
        // ~4 bytes/token; 80 bytes -> 20.
        assert_eq!(estimate_tokens(None, 80), 20); // no usage -> estimate
        assert_eq!(estimate_tokens(Some(5), 80), 20); // under-report -> floored
        assert_eq!(estimate_tokens(Some(100), 80), 100); // honest higher count kept
    }

    #[tokio::test]
    async fn agent_loop_refuses_a_call_that_would_exceed_the_token_budget() {
        // A budget of 1 token is below any prompt's input estimate, so the
        // loop must terminate before making the call. `without_usage` also
        // shows an unreported-usage provider cannot slip past this pre-check.
        let behaviours = [loaded(&agent_skill(5, 1, 60_000), Status::Enabled)];
        let handlers = registry(Box::new(StubPropose));
        let cap = Capability::new(AccessTier::Full, ActionPermissions::suggest_only());
        let audit = MockAuditSink::accepting();
        let obs = NullObserver;
        let graph = EmptyGraph;
        let provider = MockProvider::without_usage(vec![propose("graph.write")]);
        let d = Dispatcher::new(
            &behaviours,
            &handlers,
            &graph,
            AccessTier::Full,
            gate(&audit, &obs, &cap),
            Some(&provider),
            &TEST_CLOCK,
        );

        let outcomes = d.dispatch(&event("~/foo.rs")).await;
        assert_eq!(outcomes.len(), 1);
        assert!(matches!(
            &outcomes[0],
            DispatchOutcome::Terminal { outcome, .. } if outcome == "budget_tokens"
        ));
        // No gate decision was recorded: nothing was spent.
        assert!(audit.recorded().await.is_empty());
    }

    #[tokio::test]
    async fn agent_loop_token_budget_ends_a_multistep_run() {
        // 100k tokens per call dwarfs the prompt-size estimate, so post-call
        // accumulation (not the step budget) drives termination.
        let behaviours = [loaded(&agent_skill(10, 150_000, 60_000), Status::Enabled)];
        let handlers = registry(Box::new(StubPropose));
        let cap = Capability::new(AccessTier::Full, ActionPermissions::suggest_only());
        let audit = MockAuditSink::accepting();
        let obs = NullObserver;
        let graph = EmptyGraph;
        let provider = MockProvider::new(vec![
            propose("graph.write"),
            propose("graph.write"),
            propose("graph.write"),
            propose("graph.write"),
        ])
        .with_tokens_per_call(100_000);
        let d = Dispatcher::new(
            &behaviours,
            &handlers,
            &graph,
            AccessTier::Full,
            gate(&audit, &obs, &cap),
            Some(&provider),
            &TEST_CLOCK,
        );

        let outcomes = d.dispatch(&event("~/foo.rs")).await;
        // Two ~100k-token steps fit under 150k; the third is refused.
        assert!(outcomes.len() < 10, "the token budget, not the step budget, ended it");
        assert!(matches!(
            outcomes.last().unwrap(),
            DispatchOutcome::Terminal { outcome, .. } if outcome == "budget_tokens"
        ));
    }

    #[tokio::test]
    async fn agent_loop_stops_at_the_wall_clock_budget() {
        let behaviours = [loaded(&agent_skill(5, 1_000_000, 100), Status::Enabled)];
        let handlers = registry(Box::new(StubPropose));
        let cap = Capability::new(AccessTier::Full, ActionPermissions::suggest_only());
        let audit = MockAuditSink::accepting();
        let obs = NullObserver;
        let graph = EmptyGraph;
        let provider = MockProvider::new(vec![propose("graph.write"), propose("graph.write")]);
        let clock = AdvancingClock::new(60); // 60ms per now() reading
        let d = Dispatcher::new(
            &behaviours,
            &handlers,
            &graph,
            AccessTier::Full,
            gate(&audit, &obs, &cap),
            Some(&provider),
            &clock,
        );

        let outcomes = d.dispatch(&event("~/foo.rs")).await;
        assert_eq!(outcomes.len(), 2);
        assert!(matches!(outcomes[0], DispatchOutcome::Decided { .. }));
        assert!(matches!(
            &outcomes[1],
            DispatchOutcome::Terminal { outcome, .. } if outcome == "budget_wall_ms"
        ));
    }

    #[tokio::test]
    async fn agent_loop_reports_a_provider_error_as_failed() {
        let behaviours = [loaded(&agent_skill(5, 1_000_000, 60_000), Status::Enabled)];
        let handlers = registry(Box::new(StubPropose));
        let cap = Capability::new(AccessTier::Full, ActionPermissions::suggest_only());
        let audit = MockAuditSink::accepting();
        let obs = NullObserver;
        let graph = EmptyGraph;
        let provider = MockProvider::new(vec![Err(ProviderError::Unavailable("down".to_string()))]);
        let d = Dispatcher::new(
            &behaviours,
            &handlers,
            &graph,
            AccessTier::Full,
            gate(&audit, &obs, &cap),
            Some(&provider),
            &TEST_CLOCK,
        );

        let outcomes = d.dispatch(&event("~/foo.rs")).await;
        assert_eq!(outcomes.len(), 1);
        assert!(matches!(&outcomes[0], DispatchOutcome::Failed { .. }));
    }

    #[tokio::test]
    async fn agent_loop_refuses_an_out_of_scope_tool_then_continues() {
        let behaviours = [loaded(&agent_skill(5, 1_000_000, 60_000), Status::Enabled)];
        let handlers = registry(Box::new(StubPropose));
        let cap = Capability::new(AccessTier::Full, ActionPermissions::suggest_only());
        let audit = MockAuditSink::accepting();
        let obs = NullObserver;
        let graph = EmptyGraph;
        // fs.delete is not in the behaviour's declared tools.
        let provider = MockProvider::new(vec![propose("fs.delete"), stop()]);
        let d = Dispatcher::new(
            &behaviours,
            &handlers,
            &graph,
            AccessTier::Full,
            gate(&audit, &obs, &cap),
            Some(&provider),
            &TEST_CLOCK,
        );

        let outcomes = d.dispatch(&event("~/foo.rs")).await;
        assert_eq!(outcomes.len(), 2);
        assert!(matches!(&outcomes[0], DispatchOutcome::Refused { .. }));
        assert!(matches!(
            &outcomes[1],
            DispatchOutcome::Terminal { outcome, .. } if outcome == "done"
        ));
    }

    #[tokio::test]
    async fn agent_behaviour_is_skipped_without_a_provider() {
        let behaviours = [loaded(&agent_skill(5, 1_000_000, 60_000), Status::Enabled)];
        let handlers = registry(Box::new(StubPropose));
        let cap = Capability::new(AccessTier::Full, ActionPermissions::suggest_only());
        let audit = MockAuditSink::accepting();
        let obs = NullObserver;
        let graph = EmptyGraph;
        let d = Dispatcher::new(
            &behaviours,
            &handlers,
            &graph,
            AccessTier::Full,
            gate(&audit, &obs, &cap),
            None,
            &TEST_CLOCK,
        );

        let outcomes = d.dispatch(&event("~/foo.rs")).await;
        assert_eq!(outcomes.len(), 1);
        assert!(matches!(&outcomes[0], DispatchOutcome::Skipped { .. }));
    }

    #[tokio::test]
    async fn agent_loop_times_out_a_stalled_provider_at_the_wall_budget() {
        let behaviours = [loaded(&agent_skill(5, 1_000_000, 50), Status::Enabled)];
        let handlers = registry(Box::new(StubPropose));
        let cap = Capability::new(AccessTier::Full, ActionPermissions::suggest_only());
        let audit = MockAuditSink::accepting();
        let obs = NullObserver;
        let graph = EmptyGraph;
        let provider = StallProvider;
        let d = Dispatcher::new(
            &behaviours,
            &handlers,
            &graph,
            AccessTier::Full,
            gate(&audit, &obs, &cap),
            Some(&provider),
            &TEST_CLOCK,
        );

        let outcomes = d.dispatch(&event("~/foo.rs")).await;
        assert_eq!(outcomes.len(), 1);
        assert!(matches!(
            &outcomes[0],
            DispatchOutcome::Terminal { outcome, .. } if outcome == "budget_wall_ms"
        ));
    }

    #[tokio::test]
    async fn agent_loop_does_not_gate_an_over_budget_response() {
        // The input fits the budget, but the provider reports usage far over
        // it; the response must be discarded (no gated action), not used.
        let behaviours = [loaded(&agent_skill(5, 1_000_000, 60_000), Status::Enabled)];
        let handlers = registry(Box::new(StubPropose));
        let cap = Capability::new(AccessTier::Full, ActionPermissions::suggest_only());
        let audit = MockAuditSink::accepting();
        let obs = NullObserver;
        let graph = EmptyGraph;
        let provider =
            MockProvider::new(vec![propose("graph.write")]).with_tokens_per_call(3_000_000);
        let d = Dispatcher::new(
            &behaviours,
            &handlers,
            &graph,
            AccessTier::Full,
            gate(&audit, &obs, &cap),
            Some(&provider),
            &TEST_CLOCK,
        );

        let outcomes = d.dispatch(&event("~/foo.rs")).await;
        assert_eq!(outcomes.len(), 1);
        assert!(matches!(
            &outcomes[0],
            DispatchOutcome::Terminal { outcome, .. } if outcome == "budget_tokens"
        ));
        // The over-budget response never reached the gate.
        assert!(audit.recorded().await.is_empty());
    }

    #[tokio::test]
    async fn agent_loop_rejects_an_undeclared_stop_terminal() {
        let behaviours = [loaded(&agent_skill(5, 1_000_000, 60_000), Status::Enabled)];
        let handlers = registry(Box::new(StubPropose));
        let cap = Capability::new(AccessTier::Full, ActionPermissions::suggest_only());
        let audit = MockAuditSink::accepting();
        let obs = NullObserver;
        let graph = EmptyGraph;
        // An invented terminal is rejected and fed back; the declared one ends it.
        let made_up = Ok("{\"action\":\"stop\",\"terminal\":\"made_up\"}".to_string());
        let provider = MockProvider::new(vec![made_up, stop()]);
        let d = Dispatcher::new(
            &behaviours,
            &handlers,
            &graph,
            AccessTier::Full,
            gate(&audit, &obs, &cap),
            Some(&provider),
            &TEST_CLOCK,
        );

        let outcomes = d.dispatch(&event("~/foo.rs")).await;
        assert_eq!(outcomes.len(), 1);
        assert!(matches!(
            &outcomes[0],
            DispatchOutcome::Terminal { outcome, .. } if outcome == "done"
        ));
    }

    /// A clock that reports an earlier time on its second reading, to check
    /// the loop fails safe on backwards clock movement rather than resetting
    /// its elapsed budget.
    struct BackwardsClock {
        calls: std::sync::Mutex<u32>,
    }
    impl Clock for BackwardsClock {
        fn now(&self) -> std::time::SystemTime {
            let mut c = self.calls.lock().unwrap();
            *c += 1;
            // Call 1 is dispatch's coalescing clock read; call 2 is the loop's
            // per-behaviour start anchor; call 3 (the in-loop elapsed check)
            // jumps backwards to 50s, which the loop must treat as exhausted.
            if *c <= 2 {
                std::time::UNIX_EPOCH + Duration::from_secs(100)
            } else {
                std::time::UNIX_EPOCH + Duration::from_secs(50)
            }
        }
    }

    #[tokio::test]
    async fn agent_loop_fails_closed_when_audit_is_unavailable() {
        let behaviours = [loaded(&agent_skill(5, 1_000_000, 60_000), Status::Enabled)];
        let handlers = registry(Box::new(StubPropose));
        let cap = Capability::new(AccessTier::Full, ActionPermissions::suggest_only());
        let audit = MockAuditSink::failing();
        let obs = NullObserver;
        let graph = EmptyGraph;
        // Would propose repeatedly; the audit outage must stop the loop, not
        // be treated as recoverable model feedback.
        let provider = MockProvider::new(vec![propose("graph.write"), propose("graph.write")]);
        let d = Dispatcher::new(
            &behaviours,
            &handlers,
            &graph,
            AccessTier::Full,
            gate(&audit, &obs, &cap),
            Some(&provider),
            &TEST_CLOCK,
        );

        let outcomes = d.dispatch(&event("~/foo.rs")).await;
        assert_eq!(outcomes.len(), 1);
        assert!(matches!(
            &outcomes[0],
            DispatchOutcome::Failed { reason, .. } if reason.contains("audit")
        ));
    }

    #[tokio::test]
    async fn agent_loop_fails_safe_on_backwards_clock_movement() {
        let behaviours = [loaded(&agent_skill(5, 1_000_000, 60_000), Status::Enabled)];
        let handlers = registry(Box::new(StubPropose));
        let cap = Capability::new(AccessTier::Full, ActionPermissions::suggest_only());
        let audit = MockAuditSink::accepting();
        let obs = NullObserver;
        let graph = EmptyGraph;
        let provider = MockProvider::new(vec![propose("graph.write")]);
        let clock = BackwardsClock {
            calls: std::sync::Mutex::new(0),
        };
        let d = Dispatcher::new(
            &behaviours,
            &handlers,
            &graph,
            AccessTier::Full,
            gate(&audit, &obs, &cap),
            Some(&provider),
            &clock,
        );

        let outcomes = d.dispatch(&event("~/foo.rs")).await;
        assert_eq!(outcomes.len(), 1);
        assert!(matches!(
            &outcomes[0],
            DispatchOutcome::Terminal { outcome, .. } if outcome == "budget_wall_ms"
        ));
    }

    // --- Context compaction (deterministic, model-free) ---

    /// A demo agent behaviour with a budget generous enough that the token and
    /// wall bounds never interfere with the compaction logic under test.
    fn demo_behaviour() -> Behaviour {
        parse(&agent_skill(20, 100_000_000, 600_000)).expect("valid agent skill")
    }

    /// The step-prompt token estimate for a transcript, computed with the same
    /// conservative window bound the loop uses, so a test can place a window
    /// threshold precisely between two transcript states.
    fn prompt_estimate(b: &Behaviour, ev: &AgentEvent, t: &[TranscriptEntry]) -> u32 {
        window_token_estimate(build_agent_prompt(b, ev, t).len())
    }

    fn proposed(step: u32) -> TranscriptEntry {
        TranscriptEntry::Proposed {
            step,
            tool: "graph.write".to_string(),
            summary: format!("tag file {step} as part of the active project"),
            decision: "RequireConfirmation".to_string(),
        }
    }
    fn nag(step: u32) -> TranscriptEntry {
        TranscriptEntry::Nag {
            step,
            detail: "your response was not a valid step (no JSON object); reply with exactly one JSON step".to_string(),
        }
    }

    fn compact(
        b: &Behaviour,
        ev: &AgentEvent,
        t: &mut Vec<TranscriptEntry>,
        window: u32,
        p: &CompactionPolicy,
    ) -> CompactionOutcome {
        compact_for_window(b, ev, t, window, p)
    }

    #[test]
    fn compaction_leaves_an_under_threshold_transcript_untouched() {
        let b = demo_behaviour();
        let ev = event("~/foo.rs");
        let mut t = vec![proposed(0), proposed(1)];
        let before = t.clone();
        let window = 1_000_000; // far above the prompt
        let p = CompactionPolicy::default();
        assert_eq!(compact(&b, &ev, &mut t, window, &p), CompactionOutcome::Proceed);
        assert_eq!(t, before);
    }

    #[test]
    fn cheap_prune_alone_can_bring_it_under() {
        let b = demo_behaviour();
        let ev = event("~/foo.rs");
        let full: Vec<TranscriptEntry> = (0..6).map(nag).collect();
        let pruned = {
            let mut t = full.clone();
            compaction::prune(&mut t);
            t
        };
        let full_est = prompt_estimate(&b, &ev, &full);
        let pruned_est = prompt_estimate(&b, &ev, &pruned);
        assert!(pruned_est < full_est, "prune must shrink the prompt");
        let window = (pruned_est + full_est) / 2;
        let p = CompactionPolicy {
            headroom: 0,
            keep_recent: 4,
        };
        let mut t = full.clone();
        assert_eq!(compact(&b, &ev, &mut t, window, &p), CompactionOutcome::Proceed);
        assert!(t.len() < full.len()); // the nag run collapsed
    }

    #[test]
    fn tighten_brings_a_substantive_transcript_under_and_keeps_facts() {
        let b = demo_behaviour();
        let ev = event("~/foo.rs");
        let full: Vec<TranscriptEntry> = (0..8).map(proposed).collect();
        let tightened = {
            let mut t = full.clone();
            compaction::tighten(&mut t, 2);
            t
        };
        let full_est = prompt_estimate(&b, &ev, &full);
        let tight_est = prompt_estimate(&b, &ev, &tightened);
        assert!(tight_est < full_est);
        // A window the full transcript overflows but the tightened one fits;
        // prune is a no-op here (no nags), so tighten must do the work.
        let window = (tight_est + full_est) / 2;
        let p = CompactionPolicy {
            headroom: 0,
            keep_recent: 2,
        };
        let mut t = full.clone();
        assert_eq!(compact(&b, &ev, &mut t, window, &p), CompactionOutcome::Proceed);
        // The oldest proposal kept its tool and decision; only the rationale
        // prose was dropped.
        match &t[0] {
            TranscriptEntry::Proposed {
                summary,
                tool,
                decision,
                ..
            } => {
                assert!(summary.is_empty());
                assert_eq!(tool, "graph.write");
                assert_eq!(decision, "RequireConfirmation");
            }
            other => panic!("expected a tightened proposal, got {other:?}"),
        }
    }

    #[test]
    fn over_window_when_even_tightening_cannot_fit() {
        let b = demo_behaviour();
        let ev = event("~/foo.rs");
        let base = prompt_estimate(&b, &ev, &[]);
        let full: Vec<TranscriptEntry> = (0..8).map(proposed).collect();
        // A window below even the empty-transcript prompt: neither prune nor
        // tighten can help, so the loop must close over-window.
        let window = base / 2;
        let p = CompactionPolicy {
            headroom: 0,
            keep_recent: 2,
        };
        let mut t = full.clone();
        assert_eq!(compact(&b, &ev, &mut t, window, &p), CompactionOutcome::OverWindow);
    }

    #[test]
    fn over_window_when_event_alone_exceeds_the_window() {
        let b = demo_behaviour();
        let ev = event("~/foo.rs");
        let base = prompt_estimate(&b, &ev, &[]);
        let window = base / 2;
        let p = CompactionPolicy {
            headroom: 0,
            keep_recent: 4,
        };
        let mut t: Vec<TranscriptEntry> = Vec::new();
        assert_eq!(compact(&b, &ev, &mut t, window, &p), CompactionOutcome::OverWindow);
    }

    #[tokio::test]
    async fn agent_loop_terminates_budget_context_when_compaction_cannot_fit() {
        // End-to-end: a behaviour whose event alone overflows the provider's
        // window. The loop reads the window from the provider, runs compaction
        // (which cannot help), and closes with budget_context rather than
        // sending an over-window prompt.
        let behaviours = [loaded(&agent_skill(5, 1_000_000, 60_000), Status::Enabled)];
        let handlers = registry(Box::new(StubPropose));
        let cap = Capability::new(AccessTier::Full, ActionPermissions::suggest_only());
        let audit = MockAuditSink::accepting();
        let obs = NullObserver;
        let graph = EmptyGraph;
        // An 8-token window: even the empty-transcript prompt overflows it.
        let provider = MockProvider::new(vec![propose("graph.write")]).with_context_window(8); // must not be reached
        let d = Dispatcher::new(
            &behaviours,
            &handlers,
            &graph,
            AccessTier::Full,
            gate(&audit, &obs, &cap),
            Some(&provider),
            &TEST_CLOCK,
        );

        let outcomes = d.dispatch(&event(&"x".repeat(10_000))).await;
        assert_eq!(outcomes.len(), 1);
        assert!(matches!(
            &outcomes[0],
            DispatchOutcome::Terminal { outcome, .. } if outcome == "budget_context"
        ));
        // Compaction closed the loop before any step call.
        assert!(audit.recorded().await.is_empty());
    }

    #[test]
    fn output_window_cap_keeps_input_plus_output_within_the_window() {
        let window = 1000;
        // Window leaves 1000-600=400 after input; budget leaves 800 -> window-bound.
        assert_eq!(output_window_cap(window, 800, 600), 400);
        assert!(600 + output_window_cap(window, 800, 600) <= window);
        // Budget is the tighter bound here.
        assert_eq!(output_window_cap(window, 100, 600), 100);
        // Input alone fills the window: no output room.
        assert_eq!(output_window_cap(window, 800, 1200), 0);
    }

    // ---- Live executor wired into the dispatch loop ----

    use crate::executor::{LiveExecutor, RelationWrite, RelationWriter, WriteError, WriteOutcome};
    use crate::slice::{PathResolver, SliceError, StaticMountPolicy};

    /// A graph shaped like the proof for tagging `/proj/a.rs` to `p1`, unlinked
    /// so the prediction is valid (the same needles the gate/executor query).
    struct TagGraph;
    #[async_trait::async_trait]
    impl GraphHandle for TagGraph {
        async fn query(
            &self,
            cypher: &str,
        ) -> Result<Vec<HashMap<String, serde_json::Value>>, GraphError> {
            let row = |pairs: &[(&str, serde_json::Value)]| -> HashMap<String, serde_json::Value> {
                pairs.iter().map(|(k, v)| (k.to_string(), v.clone())).collect()
            };
            if cypher.contains("n:File {id: '/proj/a.rs'}") {
                Ok(vec![row(&[("id", "/proj/a.rs".into()), ("path", "/proj/a.rs".into())])])
            } else if cypher.contains("n:Project {id: 'p1'}") {
                Ok(vec![row(&[("id", "p1".into()), ("root_path", "/proj".into())])])
            } else if cypher.contains("count(*) AS cnt") {
                Ok(vec![row(&[("cnt", serde_json::Value::from(0_i64))])])
            } else {
                Ok(Vec::new())
            }
        }
    }

    /// Accepts an already-canonical absolute path as itself.
    struct IdResolver;
    impl PathResolver for IdResolver {
        fn resolve(&self, raw: &str) -> Result<String, SliceError> {
            if raw.starts_with('/') {
                Ok(raw.to_string())
            } else {
                Err(SliceError::PathResolve {
                    raw: raw.to_string(),
                    reason: "not absolute".to_string(),
                })
            }
        }
    }

    /// Proposes the provable FILE_PART_OF tag (operands matching `TagGraph`).
    struct ProvableTag;
    #[async_trait::async_trait]
    impl WorkflowHandler for ProvableTag {
        async fn run(
            &self,
            _event: &AgentEvent,
            _graph: &dyn GraphHandle,
        ) -> Result<HandlerOutcome, HandlerError> {
            Ok(HandlerOutcome::Propose(ProposedAction {
                tool: "graph.write".to_string(),
                summary: "tag /proj/a.rs as part of p1".to_string(),
                arguments: BTreeMap::from([
                    ("file".to_string(), "/proj/a.rs".to_string()),
                    ("project".to_string(), "p1".to_string()),
                ]),
            }))
        }
    }

    /// Records each write without performing I/O.
    #[derive(Default)]
    struct RecordingWriter(std::sync::Mutex<Vec<RelationWrite>>);
    #[async_trait::async_trait]
    impl RelationWriter for RecordingWriter {
        async fn write_relation(&self, write: &RelationWrite) -> Result<WriteOutcome, WriteError> {
            self.0.lock().unwrap().push(write.clone());
            Ok(WriteOutcome::Created)
        }
    }

    /// A capability that lifts a proven Ordinary action to `PreviewThenExecute`
    /// for the agent's own app id (the id the dispatcher decides under).
    fn live_cap() -> Capability {
        Capability::new(
            AccessTier::Full,
            ActionPermissions::new(BaselineMode::Suggest, [AGENT_APP_ID]),
        )
    }

    #[tokio::test]
    async fn an_opted_in_executor_performs_and_audits_a_proven_workflow_write() {
        let behaviours = [loaded(LIVE_AUTO_TAG, Status::Enabled)];
        let handlers = registry(Box::new(ProvableTag));
        let cap = live_cap();
        let audit = MockAuditSink::accepting();
        let obs = NullObserver;
        let graph = TagGraph;
        let resolver = IdResolver;
        let mounts = StaticMountPolicy::empty();
        let writer = RecordingWriter::default();

        let g = Gate::new(&cap, &audit, &obs, &resolver, &mounts);
        let executor = LiveExecutor::new(&cap, &resolver, &mounts, &writer, &audit);
        let d = Dispatcher::new(&behaviours, &handlers, &graph, AccessTier::Full, g, None, &TEST_CLOCK)
            .with_executor(executor);

        let outcomes = d.dispatch(&event("/proj/a.rs")).await;

        // The decision lifted to PreviewThenExecute and the execution result is
        // carried on the outcome (here a `Created` write), not erased.
        match outcomes.as_slice() {
            [DispatchOutcome::Decided { decision, executed, .. }] => {
                assert_eq!(*decision, ActionDecision::PreviewThenExecute);
                assert_eq!(
                    *executed,
                    Some(ExecutionResult::Written(WriteOutcome::Created)),
                    "the execution result is surfaced on the outcome"
                );
            }
            other => panic!("expected one Decided/PreviewThenExecute, got {other:?}"),
        }
        // ...and the opted-in executor performed the write.
        let written = writer.0.lock().unwrap();
        assert_eq!(written.len(), 1, "the executor wrote the relation");
        assert_eq!(written[0].relation_type, "FILE_PART_OF");
        assert_eq!(written[0].to_type, "system.Project");

        // Both the gate decision and the execution are on the ledger, correlated.
        let entries = audit.recorded().await;
        assert_eq!(entries.len(), 2, "decision + execution are both audited");
        let labels: Vec<&str> = entries.iter().map(|e| e.structural.outcome.as_str()).collect();
        // The gate records `<decision>:<reason>` (e.g. preview-then-execute:proven-reversible);
        // the executor records the fixed `execute`.
        assert!(
            labels.iter().any(|l| l.starts_with("preview-then-execute")),
            "the decision is audited, got {labels:?}"
        );
        assert!(labels.contains(&"execute"), "the execution is audited, got {labels:?}");
        assert!(
            entries
                .iter()
                .all(|e| e.call_chain_id.as_deref() == Some("e1:auto-tag-by-project")),
            "decision and execution share one correlation id"
        );
    }

    /// A writer that always fails, to prove a write failure is surfaced.
    struct FailingWriter;
    #[async_trait::async_trait]
    impl RelationWriter for FailingWriter {
        async fn write_relation(&self, _write: &RelationWrite) -> Result<WriteOutcome, WriteError> {
            Err(WriteError::Failed("daemon unreachable".to_string()))
        }
    }

    /// A writer that never returns, to exercise the write timeout end to end.
    struct HangingWriter;
    #[async_trait::async_trait]
    impl RelationWriter for HangingWriter {
        async fn write_relation(&self, _write: &RelationWrite) -> Result<WriteOutcome, WriteError> {
            std::future::pending().await
        }
    }

    #[tokio::test(start_paused = true)]
    async fn a_timed_out_live_write_is_indeterminate_not_failed() {
        let behaviours = [loaded(LIVE_AUTO_TAG, Status::Enabled)];
        let handlers = registry(Box::new(ProvableTag));
        let cap = live_cap();
        let audit = MockAuditSink::accepting();
        let obs = NullObserver;
        let graph = TagGraph;
        let resolver = IdResolver;
        let mounts = StaticMountPolicy::empty();
        let writer = HangingWriter;

        let g = Gate::new(&cap, &audit, &obs, &resolver, &mounts);
        let executor = LiveExecutor::new(&cap, &resolver, &mounts, &writer, &audit);
        let d = Dispatcher::new(&behaviours, &handlers, &graph, AccessTier::Full, g, None, &TEST_CLOCK)
            .with_executor(executor);

        // Paused time advances past the write timeout once the write is the only
        // pending work, so this resolves immediately.
        let outcomes = d.dispatch(&event("/proj/a.rs")).await;
        match outcomes.as_slice() {
            [DispatchOutcome::Decided { executed: Some(ExecutionResult::Indeterminate(_)), .. }] => {}
            other => panic!("a timed-out write must be Indeterminate, not Failed: {other:?}"),
        }
    }

    /// A writer whose request may have been sent before the connection failed:
    /// commit-unknown, so the outcome must be Indeterminate, not Failed.
    struct PostSendFailWriter;
    #[async_trait::async_trait]
    impl RelationWriter for PostSendFailWriter {
        async fn write_relation(&self, _write: &RelationWrite) -> Result<WriteOutcome, WriteError> {
            Err(WriteError::Indeterminate("connection dropped after send".to_string()))
        }
    }

    #[tokio::test]
    async fn a_post_send_write_failure_is_indeterminate_not_failed() {
        let behaviours = [loaded(LIVE_AUTO_TAG, Status::Enabled)];
        let handlers = registry(Box::new(ProvableTag));
        let cap = live_cap();
        let audit = MockAuditSink::accepting();
        let obs = NullObserver;
        let graph = TagGraph;
        let resolver = IdResolver;
        let mounts = StaticMountPolicy::empty();
        let writer = PostSendFailWriter;

        let g = Gate::new(&cap, &audit, &obs, &resolver, &mounts);
        let executor = LiveExecutor::new(&cap, &resolver, &mounts, &writer, &audit);
        let d = Dispatcher::new(&behaviours, &handlers, &graph, AccessTier::Full, g, None, &TEST_CLOCK)
            .with_executor(executor);

        let outcomes = d.dispatch(&event("/proj/a.rs")).await;
        match outcomes.as_slice() {
            [DispatchOutcome::Decided { executed: Some(ExecutionResult::Indeterminate(_)), .. }] => {}
            other => panic!("a post-send write failure must be Indeterminate: {other:?}"),
        }
    }

    #[tokio::test]
    async fn a_failed_live_write_is_surfaced_not_erased() {
        let behaviours = [loaded(LIVE_AUTO_TAG, Status::Enabled)];
        let handlers = registry(Box::new(ProvableTag));
        let cap = live_cap();
        let audit = MockAuditSink::accepting();
        let obs = NullObserver;
        let graph = TagGraph;
        let resolver = IdResolver;
        let mounts = StaticMountPolicy::empty();
        let writer = FailingWriter;

        let g = Gate::new(&cap, &audit, &obs, &resolver, &mounts);
        let executor = LiveExecutor::new(&cap, &resolver, &mounts, &writer, &audit);
        let d = Dispatcher::new(&behaviours, &handlers, &graph, AccessTier::Full, g, None, &TEST_CLOCK)
            .with_executor(executor);

        let outcomes = d.dispatch(&event("/proj/a.rs")).await;
        match outcomes.as_slice() {
            [DispatchOutcome::Decided { executed: Some(ExecutionResult::Failed(reason)), .. }] => {
                assert!(reason.contains("daemon unreachable"), "the failure reason is surfaced: {reason}");
            }
            other => panic!("expected a Decided carrying a Failed execution, got {other:?}"),
        }
        // The act was still audited before the write was attempted (S13): the
        // decision entry plus the pre-write execution entry are both present.
        assert_eq!(audit.recorded().await.len(), 2, "decision + attempted execution audited");
    }

    #[tokio::test]
    async fn suggest_mode_dispatch_writes_nothing() {
        // No executor attached (the default): the same proven decision is
        // surfaced and audited, but the graph is never written.
        let behaviours = [loaded(LIVE_AUTO_TAG, Status::Enabled)];
        let handlers = registry(Box::new(ProvableTag));
        let cap = live_cap();
        let audit = MockAuditSink::accepting();
        let obs = NullObserver;
        let graph = TagGraph;
        let resolver = IdResolver;
        let mounts = StaticMountPolicy::empty();

        let g = Gate::new(&cap, &audit, &obs, &resolver, &mounts);
        let d = Dispatcher::new(&behaviours, &handlers, &graph, AccessTier::Full, g, None, &TEST_CLOCK);
        let outcomes = d.dispatch(&event("/proj/a.rs")).await;

        assert!(matches!(outcomes.as_slice(), [DispatchOutcome::Decided { .. }]));
        let entries = audit.recorded().await;
        assert_eq!(entries.len(), 1, "suggest-mode audits only the decision, no execution");
        assert!(entries[0].structural.outcome.starts_with("preview-then-execute"));
    }
}
