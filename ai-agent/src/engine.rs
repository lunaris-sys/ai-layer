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
use std::time::Duration;

use futures::FutureExt as _;
use lunaris_ai_core::capability::{AccessTier, ActionDecision};

use crate::behaviour::{BehaviourKind, ReadScope};
use crate::gate::{ActionContext, Gate, ProposedAction};
use crate::loader::LoadedBehaviour;
use crate::router::matching_behaviours;
use crate::seams::{AgentEvent, DeniedGraph, GraphHandle, TriggerSource};

/// The trusted app id the agent acts as for now. Proper per-app resolution
/// (from the tool binding / the behaviour identity) lands later; until then
/// the agent acts as itself, and execution is capped to confirmation
/// regardless, so this never widens authority.
const AGENT_APP_ID: &str = "org.lunaris.agent";

/// Wall-clock bound on a single workflow handler run. A handler that blocks
/// or runs away is abandoned with a Failed outcome rather than stalling the
/// loop. (Agent-loop budgets, which are per-step, arrive separately.)
const HANDLER_TIMEOUT: Duration = Duration::from_secs(10);

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

/// The dispatch engine over a set of loaded behaviours, their handlers, the
/// graph handle they read through, and the action gate.
pub struct Dispatcher<'a> {
    behaviours: &'a [LoadedBehaviour],
    handlers: &'a HandlerRegistry,
    graph: &'a dyn GraphHandle,
    /// The agent's configured global read tier; a behaviour declaring more
    /// is refused before its handler runs.
    read_tier: AccessTier,
    gate: Gate<'a>,
}

impl<'a> Dispatcher<'a> {
    /// Build a dispatcher.
    pub fn new(
        behaviours: &'a [LoadedBehaviour],
        handlers: &'a HandlerRegistry,
        graph: &'a dyn GraphHandle,
        read_tier: AccessTier,
        gate: Gate<'a>,
    ) -> Self {
        Self {
            behaviours,
            handlers,
            graph,
            read_tier,
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

/// Whether a behaviour's declared read scope is satisfied by the agent's
/// configured grant. Conservative: Minimal needs nothing, Full grants
/// everything, otherwise the tiers must match exactly. The access tiers are
/// non-nested label *lenses* (e.g. project-scoped grants `Project` but
/// time-scoped does not), so a precise superset check needs the schema and
/// belongs to the read/grounding layer; this conservative form never
/// over-grants (it may refuse a satisfiable combination, fail-safe).
fn reads_satisfied(needs: ReadScope, granted: AccessTier) -> bool {
    needs == ReadScope::Minimal || granted == AccessTier::Full || needs.tier() == granted
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, VecDeque};
    use std::path::PathBuf;

    use audit_proto::MockAuditSink;
    use lunaris_ai_core::capability::{AccessTier, ActionPermissions, Capability};

    use crate::behaviour::parse;
    use crate::loader::{DisableReason, LoadedBehaviour, Provenance, Status};
    use crate::seams::{GraphError, NullObserver};

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
        Gate::new(cap, audit, obs)
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
            Dispatcher::new(&behaviours, &handlers, &graph, AccessTier::Full, gate(&audit, &obs, &cap));
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
        let d = Dispatcher::new(&enabled, &handlers, &graph, AccessTier::Full, gate(&audit, &obs, &cap));
        // ~/.cache is excluded by the filter.
        assert!(d.dispatch(&event("~/.cache/x")).await.is_empty());

        let disabled = [loaded(AUTO_TAG, Status::Disabled(DisableReason::NotEnabledInSettings))];
        let d2 = Dispatcher::new(&disabled, &handlers, &graph, AccessTier::Full, gate(&audit, &obs, &cap));
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
            Dispatcher::new(&behaviours, &handlers, &graph, AccessTier::Full, gate(&audit, &obs, &cap));

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
            Dispatcher::new(&behaviours, &handlers, &graph, AccessTier::Full, gate(&audit, &obs, &cap));

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
            Dispatcher::new(&behaviours, &handlers, &graph, AccessTier::Full, gate(&audit, &obs, &cap));

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
            Dispatcher::new(&behaviours, &handlers, &graph, AccessTier::Full, gate(&audit, &obs, &cap));

        let mut ev = event("~/foo.rs");
        ev.external_content = true;
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
        let graph = EmptyGraph;
        let dispatcher =
            Dispatcher::new(&behaviours, &handlers, &graph, AccessTier::Full, gate(&audit, &obs, &cap));

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
        let d = Dispatcher::new(&minimal, &handlers, &graph, AccessTier::Full, gate(&audit, &obs, &cap));
        assert_eq!(
            d.dispatch(&event("~/foo.rs")).await[0],
            DispatchOutcome::Terminal {
                behaviour: "probe-graph".to_string(),
                outcome: "denied".to_string(),
            }
        );

        // A project-reads behaviour reaches the real graph (here EmptyGraph).
        let project = [loaded(AUTO_TAG, Status::Enabled)];
        let d2 = Dispatcher::new(&project, &handlers, &graph, AccessTier::Full, gate(&audit, &obs, &cap));
        assert_eq!(
            d2.dispatch(&event("~/foo.rs")).await[0],
            DispatchOutcome::Terminal {
                behaviour: "auto-tag-by-project".to_string(),
                outcome: "queried".to_string(),
            }
        );
    }
}
