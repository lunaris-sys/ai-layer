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

use std::collections::BTreeMap;

use lunaris_ai_core::capability::ActionDecision;

use crate::behaviour::BehaviourKind;
use crate::gate::{ActionContext, Gate, ProposedAction};
use crate::loader::LoadedBehaviour;
use crate::router::matching_behaviours;
use crate::seams::{AgentEvent, TriggerSource};

/// The trusted app id the agent acts as in B1. Proper per-app resolution
/// (from the tool binding / the behaviour identity) lands with the
/// production wiring; until then the agent acts as itself, and B1 caps
/// execution to confirmation regardless, so this never widens authority.
const AGENT_APP_ID: &str = "org.lunaris.agent";

/// What a workflow handler decides for a matched event.
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

/// A workflow behaviour's code handler. Synchronous in B1 (stub / no I/O);
/// when a handler needs to read the graph (the real `auto-tag-by-project`),
/// it gains async + a read-scoped `GraphHandle` seam that lands with it.
/// Returns a `Result` so a handler can fail gracefully; a *panic* is also
/// caught by the dispatcher (see [`Dispatcher::dispatch`]).
pub trait WorkflowHandler: Send + Sync {
    /// Run the workflow for one matched event.
    fn run(&self, event: &AgentEvent) -> Result<HandlerOutcome, HandlerError>;
}

/// Maps a behaviour manifest's `handler` id to its code handler.
pub type HandlerRegistry = BTreeMap<String, Box<dyn WorkflowHandler>>;

/// The outcome of dispatching one matched behaviour for one event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DispatchOutcome {
    /// The gate decided on a proposed action (and audited it).
    Decided {
        /// The behaviour that ran.
        behaviour: String,
        /// The gate's decision.
        decision: ActionDecision,
        /// The audit ledger index of the recorded decision.
        audit_index: u64,
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
}

/// The dispatch engine over a set of loaded behaviours, their handlers, and
/// the action gate.
pub struct Dispatcher<'a> {
    behaviours: &'a [LoadedBehaviour],
    handlers: &'a HandlerRegistry,
    gate: Gate<'a>,
}

impl<'a> Dispatcher<'a> {
    /// Build a dispatcher.
    pub fn new(
        behaviours: &'a [LoadedBehaviour],
        handlers: &'a HandlerRegistry,
        gate: Gate<'a>,
    ) -> Self {
        Self {
            behaviours,
            handlers,
            gate,
        }
    }

    /// Dispatch one event: route it to every enabled matching behaviour and
    /// run each, returning one outcome per matched behaviour.
    pub async fn dispatch(&self, event: &AgentEvent) -> Vec<DispatchOutcome> {
        let mut outcomes = Vec::new();
        for lb in matching_behaviours(&event.event_type, &event.fields, self.behaviours) {
            outcomes.push(self.dispatch_one(lb, event).await);
        }
        outcomes
    }

    async fn dispatch_one(&self, lb: &LoadedBehaviour, event: &AgentEvent) -> DispatchOutcome {
        let m = &lb.behaviour.manifest;
        let behaviour = m.name.clone();

        // B1 runs workflow behaviours only; the bounded agent loop is B2.
        if m.kind != BehaviourKind::Workflow {
            return DispatchOutcome::Skipped {
                behaviour,
                reason: "agent behaviours run in B2".to_string(),
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

        // Isolate handler failures *and* panics: a malformed event or a
        // buggy handler produces a Failed outcome and the dispatcher moves
        // on to the next behaviour, rather than stalling the whole loop.
        // (A handler that *blocks* is only bounded once handlers are async
        // and a per-handler timeout lands with the GraphHandle seam.)
        let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| handler.run(event)));
        let outcome = match run {
            Ok(Ok(outcome)) => outcome,
            Ok(Err(e)) => {
                return DispatchOutcome::Failed {
                    behaviour,
                    reason: e.to_string(),
                }
            }
            Err(_) => {
                return DispatchOutcome::Failed {
                    behaviour,
                    reason: "handler panicked".to_string(),
                }
            }
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
                    .decide_action(&behaviour, m.mode, &m.tools, &action, &ctx)
                    .await
                {
                    Ok(receipt) => DispatchOutcome::Decided {
                        behaviour,
                        decision: receipt.decision,
                        audit_index: receipt.audit_index,
                    },
                    Err(e) => DispatchOutcome::Refused {
                        behaviour,
                        reason: e.to_string(),
                    },
                }
            }
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use std::path::PathBuf;

    use audit_proto::MockAuditSink;
    use lunaris_ai_core::capability::{AccessTier, ActionPermissions, Capability};

    use crate::behaviour::parse;
    use crate::loader::{LoadedBehaviour, Provenance, Status};
    use crate::seams::NullObserver;

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

    struct StubPropose;
    impl WorkflowHandler for StubPropose {
        fn run(&self, _event: &AgentEvent) -> Result<HandlerOutcome, HandlerError> {
            Ok(HandlerOutcome::Propose(ProposedAction {
                tool: "graph.write".to_string(),
                summary: "tag the opened file".to_string(),
            }))
        }
    }

    struct StubTerminal;
    impl WorkflowHandler for StubTerminal {
        fn run(&self, _event: &AgentEvent) -> Result<HandlerOutcome, HandlerError> {
            Ok(HandlerOutcome::Terminal("no_matching_project".to_string()))
        }
    }

    struct StubPanic;
    impl WorkflowHandler for StubPanic {
        fn run(&self, _event: &AgentEvent) -> Result<HandlerOutcome, HandlerError> {
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

    fn suggest_gate<'a>(audit: &'a MockAuditSink, obs: &'a NullObserver, cap: &'a Capability) -> Gate<'a> {
        Gate::new(cap, audit, obs)
    }

    #[tokio::test]
    async fn dispatches_a_matching_workflow_through_the_gate() {
        let behaviours = [loaded(AUTO_TAG, Status::Enabled)];
        let handlers = registry(Box::new(StubPropose));
        let cap = Capability::new(AccessTier::Full, ActionPermissions::suggest_only());
        let audit = MockAuditSink::accepting();
        let obs = NullObserver;

        let dispatcher = Dispatcher::new(&behaviours, &handlers, suggest_gate(&audit, &obs, &cap));
        let outcomes = dispatcher.dispatch(&event("~/Repositories/foo.rs")).await;

        assert_eq!(outcomes.len(), 1);
        assert_eq!(
            outcomes[0],
            DispatchOutcome::Decided {
                behaviour: "auto-tag-by-project".to_string(),
                decision: ActionDecision::Propose,
                audit_index: 0,
            }
        );
        // The decision was audited (content-free, correlated).
        let recorded = audit.recorded().await;
        assert_eq!(recorded[0].structural.subject, "agent.auto-tag-by-project");
        assert_eq!(recorded[0].call_chain_id.as_deref(), Some("e1:auto-tag-by-project"));
    }

    #[tokio::test]
    async fn filtered_event_dispatches_to_nothing() {
        let behaviours = [loaded(AUTO_TAG, Status::Enabled)];
        let handlers = registry(Box::new(StubPropose));
        let cap = Capability::new(AccessTier::Full, ActionPermissions::suggest_only());
        let audit = MockAuditSink::accepting();
        let obs = NullObserver;
        let dispatcher = Dispatcher::new(&behaviours, &handlers, suggest_gate(&audit, &obs, &cap));

        // ~/.cache is excluded by the behaviour's filter, and a disabled
        // behaviour never matches.
        assert!(dispatcher.dispatch(&event("~/.cache/x")).await.is_empty());
        let disabled = [loaded(AUTO_TAG, Status::Disabled(crate::loader::DisableReason::NotEnabledInSettings))];
        let d2 = Dispatcher::new(&disabled, &handlers, suggest_gate(&audit, &obs, &cap));
        assert!(d2.dispatch(&event("~/foo.rs")).await.is_empty());
    }

    #[tokio::test]
    async fn terminal_handler_records_a_terminal_outcome() {
        let behaviours = [loaded(AUTO_TAG, Status::Enabled)];
        let handlers = registry(Box::new(StubTerminal));
        let cap = Capability::new(AccessTier::Full, ActionPermissions::suggest_only());
        let audit = MockAuditSink::accepting();
        let obs = NullObserver;
        let dispatcher = Dispatcher::new(&behaviours, &handlers, suggest_gate(&audit, &obs, &cap));

        let outcomes = dispatcher.dispatch(&event("~/foo.rs")).await;
        assert_eq!(
            outcomes[0],
            DispatchOutcome::Terminal {
                behaviour: "auto-tag-by-project".to_string(),
                outcome: "no_matching_project".to_string(),
            }
        );
        // A terminal (no action) is not audited as a gate decision.
        assert_eq!(audit.count().await, 0);
    }

    #[tokio::test]
    async fn unregistered_handler_is_skipped_not_run() {
        let behaviours = [loaded(AUTO_TAG, Status::Enabled)];
        let handlers: HandlerRegistry = BTreeMap::new(); // nothing registered
        let cap = Capability::new(AccessTier::Full, ActionPermissions::suggest_only());
        let audit = MockAuditSink::accepting();
        let obs = NullObserver;
        let dispatcher = Dispatcher::new(&behaviours, &handlers, suggest_gate(&audit, &obs, &cap));

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
        let dispatcher = Dispatcher::new(&behaviours, &handlers, suggest_gate(&audit, &obs, &cap));

        let mut source = VecSource(VecDeque::from([event("~/a.rs"), event("~/b.rs")]));
        dispatcher.run(&mut source).await;
        // Both events dispatched + audited.
        assert_eq!(audit.count().await, 2);
    }

    #[tokio::test]
    async fn external_content_event_is_gated_to_confirmation() {
        let behaviours = [loaded(AUTO_TAG, Status::Enabled)];
        let handlers = registry(Box::new(StubPropose));
        let cap = Capability::new(AccessTier::Full, ActionPermissions::suggest_only());
        let audit = MockAuditSink::accepting();
        let obs = NullObserver;
        let dispatcher = Dispatcher::new(&behaviours, &handlers, suggest_gate(&audit, &obs, &cap));

        let mut ev = event("~/foo.rs");
        ev.external_content = true; // a run triggered by external content
        let outcomes = dispatcher.dispatch(&ev).await;
        assert_eq!(
            outcomes[0],
            DispatchOutcome::Decided {
                behaviour: "auto-tag-by-project".to_string(),
                decision: ActionDecision::RequireConfirmation,
                audit_index: 0,
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
        let dispatcher = Dispatcher::new(&behaviours, &handlers, suggest_gate(&audit, &obs, &cap));

        let outcomes = dispatcher.dispatch(&event("~/foo.rs")).await;
        assert!(matches!(outcomes[0], DispatchOutcome::Failed { .. }));
        // A failed handler produced no gate decision to audit.
        assert_eq!(audit.count().await, 0);
    }
}
