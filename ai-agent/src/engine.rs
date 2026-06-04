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
use lunaris_ai_core::provider::{AIProvider, CompletionRequest};

use crate::agentic::{build_agent_prompt, parse_agent_step, AgentStep};
use crate::behaviour::{Behaviour, BehaviourKind, ReadScope};
use crate::compaction::{self, CompactionPolicy, TranscriptEntry};
use crate::gate::{ActionContext, Gate, GateError, ProposedAction};
use crate::loader::LoadedBehaviour;
use crate::router::matching_behaviours;
use crate::seams::{AgentEvent, Clock, DeniedGraph, GraphHandle, TriggerSource};

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
        /// The action the gate decided on, carried so a surface can show it
        /// to the user (the summary is for display; the audit subject stays
        /// content-free).
        action: ProposedAction,
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
            provider,
            clock,
            compaction: CompactionPolicy::default(),
        }
    }

    /// Override the context-compaction policy (e.g. with the real model's
    /// window once a provider is wired).
    pub fn with_compaction(mut self, policy: CompactionPolicy) -> Self {
        self.compaction = policy;
        self
    }

    /// Dispatch one event: route it to every enabled matching behaviour and
    /// run each, returning the outcomes. A workflow behaviour yields one
    /// outcome; a `kind: agent` behaviour yields one per loop step plus a
    /// terminal.
    pub async fn dispatch(&self, event: &AgentEvent) -> Vec<DispatchOutcome> {
        let mut outcomes = Vec::new();
        for lb in matching_behaviours(&event.event_type, &event.fields, self.behaviours) {
            if lb.behaviour.manifest.kind == BehaviourKind::Agent {
                match self.provider {
                    Some(provider) => {
                        outcomes.extend(self.run_agent_loop(lb, event, provider).await)
                    }
                    None => outcomes.push(DispatchOutcome::Skipped {
                        behaviour: lb.behaviour.manifest.name.clone(),
                        reason: "no AI provider configured; agent behaviours cannot run"
                            .to_string(),
                    }),
                }
                continue;
            }
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
                    .decide_action(&behaviour, m.mode, &m.tools, &action, &ctx, graph)
                    .await
                {
                    Ok(receipt) => DispatchOutcome::Decided {
                        behaviour,
                        action,
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
    ) -> Vec<DispatchOutcome> {
        let m = &lb.behaviour.manifest;
        let behaviour = m.name.clone();

        if !reads_satisfied(m.reads, self.read_tier) {
            return vec![DispatchOutcome::Skipped {
                behaviour,
                reason: format!(
                    "declared read scope {:?} exceeds the configured grant",
                    m.reads
                ),
            }];
        }
        // An agent manifest is required to declare a budget (enforced at
        // load); refuse to run one without bounds rather than loop freely.
        let Some(budget) = m.budget.as_ref() else {
            return vec![DispatchOutcome::Skipped {
                behaviour,
                reason: "agent behaviour declares no budget".to_string(),
            }];
        };

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
        let start = self.clock.now();
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
                    match self
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
                            outcomes.push(DispatchOutcome::Decided {
                                behaviour: behaviour.clone(),
                                action,
                                decision: receipt.decision,
                                audit_index: receipt.audit_index,
                            });
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
                        }
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
/// over-grants (it may refuse a satisfiable combination, fail-safe).
fn reads_satisfied(needs: ReadScope, granted: AccessTier) -> bool {
    needs == ReadScope::Minimal || granted == AccessTier::Full || needs.tier() == granted
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
    use crate::seams::{GraphError, NullObserver, SystemClock};

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
            if *c == 1 {
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
}
