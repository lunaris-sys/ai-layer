//! The action gate: the single point every action a behaviour proposes
//! must pass before it is surfaced or executed.
//!
//! It composes things that already exist:
//! * **tool-scope enforcement** — the proposed tool must be in the
//!   behaviour's declared `tools` scope (the manifest map); an
//!   out-of-scope proposal is refused fail-closed, so a behaviour can only
//!   ever act through the tools it declared;
//! * the **capability decision** (S16) — combining the behaviour's
//!   requested mode *ceiling* with the trusted per-app grant
//!   ([`Capability::decide_for_behaviour`]), so an untrusted behaviour can
//!   only ever narrow authority;
//! * a **fail-closed audit-before-acting** write — the decision is
//!   recorded in the ledger *before* the action is surfaced/executed; if
//!   the ledger cannot record it the gate refuses (no un-audited AI
//!   activity, foundation §8.4.6/.7);
//! * the [`GateObserver`] seam — the read-only tap the audit/anomaly/
//!   inspection layers attach to.
//!
//! ## Trust boundary
//!
//! The action proposal is **untrusted** (it originates from a behaviour /
//! the model). It must never be able to classify its own risk or claim
//! its own provenance, so this gate accepts **neither** from the proposal:
//!
//! * **All authorization inputs come from the trusted [`ActionContext`]**,
//!   resolved by the dispatcher — never from the proposal. That includes
//!   the **target `app_id`** (which per-app grant applies: a proposal
//!   cannot name an autonomous app to get a laxer decision), the
//!   **`external_trigger`** flag (any externally-triggered action always
//!   confirms — prompt-injection containment), and the correlation id. The
//!   proposal carries only the tool name + a human summary.
//! * **High-impact classification** is done here, using the *same* shared
//!   classifier the MCP layer uses ([`AlwaysConfirm`], keyed on the tool
//!   name) — so a destructive tool (delete / send / install / exec / …)
//!   always resolves to `RequireConfirmation` regardless of the configured
//!   mode. The proposer supplies only the tool *name* (drawn from its
//!   declared `tools` scope), never a risk class, so it cannot label a
//!   delete as `Ordinary`. The MCP dispatch boundary re-classifies the
//!   *real* tool at execution time as defense-in-depth.
//!
//! ## Boundary with the world model (B2)
//!
//! Name-based classification catches the *clearly* destructive set
//! (delete / send / install / sudo / exec / config-write). It cannot judge
//! *argument-dependent* destructiveness — e.g. an `fs.move` is reversible
//! unless its destination is occupied, in which case it is an irreversible
//! overwrite (design-doc gap F4). That judgment belongs to the **world-model
//! action schema** (preconditions + effects), not a name heuristic.
//!
//! So before an executing decision (PreviewThenExecute / Proceed) is allowed
//! through, the gate runs a **predict-before-act** step: it resolves the
//! action's trusted, registry-resolved schema, builds a bounded graph slice
//! (through the behaviour-scoped graph handle, so the proof never reads more
//! than the behaviour may) for the invocation's operands, and asks the
//! world-model interpreter whether the preconditions hold and the effects
//! apply cleanly. A `Valid` prediction lifts the conservative cap; otherwise
//! (no registered rule, an unprovable invocation, or any failure or timeout)
//! the executing decision is downgraded to explicit confirmation, so nothing
//! auto-executes whose argument-level safety is unproven. Suggest/Propose is
//! unaffected (the user executes manually). The prediction comes only from the
//! trusted world model, never the proposal, and the lifted-or-capped decision
//! is what the audit records.
//!
//! The lift is deliberately conservative: a proven action is lifted only to a
//! **previewed execution (PreviewThenExecute), never silent autonomous
//! Proceed**. Two boundaries that safe auto-execution needs are not yet in
//! place: the proof is a point-in-time slice (the graph exposes no
//! snapshot/version), so an executor must atomically re-check the
//! preconditions at write time (gap A2), and the per-app grant consulted is
//! the agent's own (the acting app), a coarse model a finer per-target grant
//! will refine. The human-visible preview is the bridge until those land;
//! nothing executes today (there is no executor), so the lifted decision is
//! the authorization the executor will later honour, not an execution.
//!
//! ## Executor obligations (the contract a lifted decision carries)
//!
//! The design (ground truth) deliberately separates this gate's *lift* (the
//! authorization) from the executor's *enforcement*. Before acting on a lifted
//! `PreviewThenExecute`, the executor (a later increment) must:
//! 1. **Execute exactly the proven effect**, the schema effect for the proven
//!    operands (e.g. the `FILE_PART_OF` `AssertEdge`), never a free-form
//!    re-interpretation of the tool name (else a different mutation rides on
//!    the proof).
//! 2. **Atomically re-check the preconditions at write time** (gap A2): the
//!    proof is a point-in-time slice and the graph exposes no snapshot, so a
//!    just-read absence can go stale; the write must be conditional on the
//!    preconditions still holding, and idempotent.
//! 3. **Enforce the manifest's per-tool scope *values*** (e.g. `graph.write`
//!    restricted to certain projects) and **resolve the real per-target/app
//!    binding**, refining today's coarse agent-grant model.
//!
//! Until all three exist, the cap holds beyond preview and nothing
//! auto-executes; these are not this increment's to build.

use std::collections::BTreeMap;
use std::time::Duration;

use lunaris_ai_core::audit::{behaviour_action_event, AuditSink};
use lunaris_ai_core::capability::{ActionDecision, ActionKind, BaselineMode, Capability};
use lunaris_ai_core::mcp::{AlwaysConfirm, AlwaysConfirmReason};

use crate::registry;
use crate::seams::{GateObserver, GraphHandle};
use crate::slice::{build_slice_trusted, MountPolicy, PathResolver};
use crate::world::{self, EvalContext};

/// How long the predict-before-act proof may run before it is treated as
/// unproven (so the conservative cap stands). It reads the graph and the
/// filesystem; a stalled dependency must not park the gate.
///
/// This bounds the async work (the graph round trips). It cannot interrupt a
/// *blocking* filesystem call mid-syscall (the production path/mount resolvers
/// use `std::fs`), so a hung FUSE/NFS mount could still park the worker past
/// the deadline. Making the path/mount seams async (resolving on a blocking
/// pool) so the deadline bounds them too is a follow-up; the common stall (a
/// slow knowledge socket) is async and is bounded here today.
const PROOF_TIMEOUT: Duration = Duration::from_secs(5);

/// An action a behaviour proposes. Carries only what a proposer may
/// legitimately state — the tool/operation it wants to invoke, a
/// human-facing summary, and the operands (arguments) the invocation will
/// use. It deliberately carries **no authorization inputs**: not the target
/// app id, not a risk class, not an external-content flag. Every input that
/// steers the gate decision is trusted and arrives via [`ActionContext`],
/// never the proposal — an untrusted proposal must not be able to pick which
/// per-app grant applies, label its own risk, or claim non-external
/// provenance. The `summary` is for the proposal/preview UI and is never
/// audited (the audit subject is content-free).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProposedAction {
    /// The MCP tool / operation the behaviour wants to invoke. Classified
    /// by the shared always-confirm classifier and checked against the
    /// behaviour's declared `tools` scope; the *real* tool is re-classified
    /// at MCP dispatch.
    pub tool: String,
    /// Human-facing description for the proposal/preview surface.
    pub summary: String,
    /// The action's operands, as parameter-name to value (a node id or a
    /// path literal). These are **untrusted** — the proposer states them —
    /// so they prove nothing on their own; the predict-before-act step checks
    /// them against the action's trusted, registry-resolved schema and the
    /// real graph before any execution cap is lifted. Empty when the proposer
    /// states no operands (the action can then only be suggested, not proven).
    pub arguments: BTreeMap<String, String>,
}

/// The **trusted** context for a gate decision, resolved by the dispatcher
/// — never taken from the (untrusted) proposal.
#[derive(Debug, Clone, Copy)]
pub struct ActionContext<'a> {
    /// The application whose per-app grant applies. The dispatcher resolves
    /// it from the behaviour identity / the tool's binding; a proposal can
    /// never name an arbitrary app to pick a laxer grant.
    pub app_id: &'a str,
    /// Whether this run was triggered by external content (forces
    /// confirmation — prompt-injection containment). A run-context fact.
    pub external_trigger: bool,
    /// Per-action correlation id, carried into the audit ledger so this
    /// decision links to the subsequent execution/outcome entry.
    pub correlation_id: &'a str,
}

/// The gate's verdict for one proposed action, plus the ledger index of
/// the audit entry that recorded it. The executor attaches `audit_index`
/// to the subsequent execution/outcome record so the two link in the
/// ledger.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GateReceipt {
    /// What the gate decided.
    pub decision: ActionDecision,
    /// The audit ledger index of the recorded decision.
    pub audit_index: u64,
}

/// Why the gate refused to let an action proceed.
#[derive(Debug, thiserror::Error)]
pub enum GateError {
    /// The audit ledger could not record the decision. Fail-closed: the
    /// action must not be surfaced or executed.
    #[error("audit log unavailable, action refused: {0}")]
    AuditUnavailable(String),
    /// The proposed tool is not in the behaviour's declared `tools` scope.
    /// A behaviour may only ever act through the tools it declared, so an
    /// out-of-scope proposal (a compromised or buggy behaviour) is refused.
    #[error("tool '{tool}' is not in the behaviour's declared scope")]
    ToolOutOfScope {
        /// The out-of-scope tool the proposal named.
        tool: String,
    },
}

/// The action gate, holding the long-lived collaborators: the capability, the
/// audit sink, the observer seam, and the system path/mount resolvers the
/// predict-before-act step reads through. The graph is *not* held here: it is
/// passed per call to [`Gate::decide_action`] as the behaviour-scoped handle
/// the dispatcher chose (a denying handle for a `reads: minimal` behaviour),
/// so the proof can never read more of the graph than the behaviour may.
pub struct Gate<'a> {
    capability: &'a Capability,
    audit: &'a dyn AuditSink,
    observer: &'a dyn GateObserver,
    paths: &'a dyn PathResolver,
    mounts: &'a dyn MountPolicy,
}

impl<'a> Gate<'a> {
    /// Build a gate over its collaborators.
    pub fn new(
        capability: &'a Capability,
        audit: &'a dyn AuditSink,
        observer: &'a dyn GateObserver,
        paths: &'a dyn PathResolver,
        mounts: &'a dyn MountPolicy,
    ) -> Self {
        Self {
            capability,
            audit,
            observer,
            paths,
            mounts,
        }
    }

    /// Decide the gate for one proposed action: resolve the capability
    /// decision, record it in the audit ledger fail-closed, notify the
    /// observer, and return a [`GateReceipt`] for the caller to act on.
    ///
    /// `behaviour_name` must be a validated kebab-case behaviour name (it
    /// becomes the content-free audit subject). `external_trigger` and
    /// `correlation_id` are supplied by the trusted dispatcher, never the
    /// proposal (see the trust-boundary note on this module).
    pub async fn decide_action(
        &self,
        behaviour_name: &str,
        ceiling: BaselineMode,
        tools: &BTreeMap<String, Vec<String>>,
        action: &ProposedAction,
        ctx: &ActionContext<'_>,
        graph: &dyn GraphHandle,
    ) -> Result<GateReceipt, GateError> {
        // Tool-scope enforcement: a behaviour may only act through a tool
        // it declared. An out-of-scope proposal is refused fail-closed and
        // still audited (a scope violation is AI activity worth recording).
        // NB: only the tool *name* is enforced here; the scope-list *values*
        // (e.g. `fs.move: [~/Downloads]`) need structured action arguments
        // to verify and are enforced by the B2 world-model/executor layer.
        if !tools.contains_key(&action.tool) {
            self.audit
                .submit(behaviour_action_event(
                    behaviour_name,
                    "refused-out-of-scope",
                    ctx.correlation_id,
                ))
                .await
                .map_err(|e| GateError::AuditUnavailable(e.to_string()))?;
            return Err(GateError::ToolOutOfScope {
                tool: action.tool.clone(),
            });
        }

        // Reversibility (Foundation B1) grounds the gate's high-impact logic:
        // "reversible" was assumed but never defined, leaving it circular. An
        // action is reversible iff its registry-resolved schema has a derivable
        // compensation. `None` = unregistered (unmodelled, not a *declared*
        // irreversibility), `Some(false)` = registered but irreversible,
        // `Some(true)` = reversible. A static property of the schema's effects.
        let schema_reversible = registry::lookup(&action.tool).map(|t| t.is_reversible());

        // Classify the proposed tool with the shared always-confirm classifier
        // (the same one MCP dispatch uses) — never a risk class taken from the
        // proposal. A tool already high-impact by name keeps that specific
        // class; otherwise a registered-but-irreversible schema escalates to
        // `Irreversible` (also high-impact), so an irreversible action always
        // requires confirmation in EVERY mode, not only the executing one. An
        // unregistered tool stays Ordinary and is held back instead by the lift
        // below, which needs a proof it cannot get. Combine with the mode
        // (ceiling ∧ grant) and the external-trigger override.
        let base_kind = action_kind_for_tool(&action.tool);
        let kind = if !base_kind.always_requires_confirmation() && schema_reversible == Some(false) {
            ActionKind::Irreversible
        } else {
            base_kind
        };
        let decision =
            self.capability
                .decide_for_behaviour(ctx.app_id, kind, ctx.external_trigger, ceiling);

        // Predict-before-act. An executing decision (PreviewThenExecute /
        // Proceed) is only authorised autonomously if the world model proves
        // *this* invocation safe: its trusted, registry-resolved schema holds
        // against the action's operands and a bounded graph slice. A `Valid`
        // prediction lifts the conservative cap to the capability's real
        // decision; no rule, an unprovable invocation, or any failure keeps
        // the cap (downgrade to explicit confirmation). Suggest/Propose is
        // unaffected (there the user executes manually). The proof runs only
        // for an executing decision (the cap would not change the others), and
        // the lifted decision is what the audit below records.
        // The proof reads the graph and the filesystem, so bound it: a stalled
        // knowledge socket or a slow path lookup must fail closed (unproven,
        // so the cap stands) rather than park the gate and stall later
        // dispatch. A timeout is treated exactly like an unprovable action.
        let proven = if matches!(
            decision,
            ActionDecision::PreviewThenExecute | ActionDecision::Proceed
        ) {
            tokio::time::timeout(PROOF_TIMEOUT, self.prove_action(action, kind, ctx, ceiling, graph))
                .await
                .unwrap_or(false)
        } else {
            false
        };

        let decision = match decision {
            // A proven, reversible executing action is lifted, but only to a
            // *previewed* execution, never silent autonomous `Proceed`. Two
            // boundaries are not yet in place that full auto-execution would
            // need, so the human-visible preview is the bridge: (1) the proof is
            // a point-in-time slice (the graph has no snapshot/version, gap A2),
            // so the executor that eventually acts on a lifted decision must
            // re-check the preconditions atomically at write time; (2) the
            // per-app grant consulted is the *agent's own* (the acting app),
            // the current coarse model, so a finer per-target/per-behaviour
            // grant is future work. Capping at preview keeps a human in the
            // loop until those land. Only a `Some(true)` reversible schema is
            // lifted: an irreversible one was already escalated to `Irreversible`
            // above (so it never reaches this arm), and an unregistered tool
            // (`None`) cannot be proven, so both stay confirmation. Nothing
            // executes today (there is no executor); this is the authorization
            // the executor will honour.
            ActionDecision::PreviewThenExecute | ActionDecision::Proceed => {
                if proven && schema_reversible == Some(true) {
                    ActionDecision::PreviewThenExecute
                } else {
                    ActionDecision::RequireConfirmation
                }
            }
            keep @ (ActionDecision::Propose | ActionDecision::RequireConfirmation) => keep,
        };

        // Audit-before-acting, fail-closed. The decision is committed to
        // the ledger before the caller is told what it is, so there is no
        // path on which the action is surfaced/executed without a record.
        let audit_index = self
            .audit
            .submit(behaviour_action_event(
                behaviour_name,
                decision_label(decision),
                ctx.correlation_id,
            ))
            .await
            .map_err(|e| GateError::AuditUnavailable(e.to_string()))?;

        self.observer.observed(&decision);
        Ok(GateReceipt {
            decision,
            audit_index,
        })
    }

    /// Whether the world model proves this invocation safe: its trusted,
    /// registry-resolved schema's preconditions hold and its effects apply
    /// cleanly over a bounded graph slice for the action's operands. Fails
    /// closed (returns `false`) on no registered rule, any slice-build failure
    /// (an unreachable graph, a malformed result, an unresolved path, an
    /// operand the schema does not name), or a prediction that is not `Valid`.
    /// A `false` here is not a refusal, it just means the conservative cap
    /// stands.
    ///
    /// The proof binds the trusted schema (resolved by the tool id) and the
    /// invocation's exact operands to a specific effect (e.g. the schema's
    /// `AssertEdge`). The executor that eventually acts on a lifted decision
    /// must execute *that proven effect* with *those operands*, not a free-form
    /// re-interpretation of the tool name, or a different mutation could ride
    /// on the proof. That obligation, with the atomic precondition re-check, is
    /// the executor's (it does not exist yet).
    async fn prove_action(
        &self,
        action: &ProposedAction,
        kind: ActionKind,
        ctx: &ActionContext<'_>,
        ceiling: BaselineMode,
        graph: &dyn GraphHandle,
    ) -> bool {
        // The schema must come from the trusted registry, never the proposal;
        // with no registered rule the action cannot be proven.
        let Some(trusted) = registry::lookup(&action.tool) else {
            return false;
        };
        // Build the bounded slice for this invocation's operands, reading
        // through the behaviour-scoped `graph` the caller passed (a denying
        // handle for a `reads: minimal` behaviour, so the proof cannot read
        // more than the behaviour may). `arguments` is the (untrusted) operand
        // set; the schema and the slice are what make a `Valid` prediction
        // trustworthy. Any build failure fails closed.
        let (state, bindings) = match build_slice_trusted(
            &trusted,
            &action.tool,
            &action.arguments,
            graph,
            self.paths,
            self.mounts,
        )
        .await
        {
            Ok(slice) => slice,
            Err(_) => return false,
        };
        let eval = EvalContext {
            capability: self.capability,
            action_id: &action.tool,
            app_id: ctx.app_id,
            action_kind: kind,
            external_trigger: ctx.external_trigger,
            ceiling,
        };
        world::predict(trusted.schema(), &bindings, &state, &eval).is_valid()
    }
}

/// Classify the proposed tool into a capability [`ActionKind`] using the
/// shared [`AlwaysConfirm`] classifier (the same patterns the MCP dispatch
/// layer uses). A tool not in the always-confirm set is [`ActionKind::Ordinary`];
/// every confirm-set reason maps to a high-impact kind whose
/// `always_requires_confirmation()` is true. Generic command execution has
/// no narrower kind — it maps to [`ActionKind::ElevatedPrivilege`], which
/// (correctly) forces confirmation.
fn action_kind_for_tool(tool: &str) -> ActionKind {
    match AlwaysConfirm::classify(tool) {
        None => ActionKind::Ordinary,
        Some(AlwaysConfirmReason::FileDeletion) => ActionKind::PermanentDelete,
        Some(AlwaysConfirmReason::ExternalMessage) => ActionKind::SendExternalMessage,
        Some(AlwaysConfirmReason::PackageChange) => ActionKind::PackageChange,
        Some(AlwaysConfirmReason::SystemConfigWrite) => ActionKind::SystemConfigChange,
        Some(AlwaysConfirmReason::ElevatedCommand | AlwaysConfirmReason::GenericExecution) => {
            ActionKind::ElevatedPrivilege
        }
    }
}

/// The coarse, content-free decision label recorded in the audit ledger.
fn decision_label(decision: ActionDecision) -> &'static str {
    match decision {
        ActionDecision::Propose => "propose",
        ActionDecision::PreviewThenExecute => "preview-then-execute",
        ActionDecision::Proceed => "proceed",
        ActionDecision::RequireConfirmation => "require-confirmation",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    use std::collections::HashMap;

    use audit_proto::MockAuditSink;
    use lunaris_ai_core::capability::{AccessTier, ActionPermissions};

    use crate::seams::{DeniedGraph, GraphError};
    use crate::slice::{FsPathResolver, SliceError, StaticMountPolicy};

    // A recording observer doubles as the GateObserver test double.
    #[derive(Default)]
    struct Recorder(Mutex<Vec<ActionDecision>>);
    impl GateObserver for Recorder {
        fn observed(&self, decision: &ActionDecision) {
            self.0.lock().unwrap().push(*decision);
        }
    }

    fn action() -> ProposedAction {
        ProposedAction {
            tool: "graph.write".to_string(),
            summary: "tag foo.rs as part of lunaris-sys".to_string(),
            arguments: BTreeMap::new(),
        }
    }

    fn irreversible_action() -> ProposedAction {
        ProposedAction {
            tool: "test.irreversible".to_string(),
            summary: "an action with no derivable compensation".to_string(),
            arguments: BTreeMap::new(),
        }
    }

    #[tokio::test]
    async fn an_irreversible_action_requires_confirmation_even_in_suggest_mode() {
        // Foundation B1: an action whose registry schema has no compensation is
        // irreversible -> high-impact -> always confirm, in EVERY mode. Under
        // Suggest the preliminary decision would be Propose; the schema-derived
        // Irreversible classification escalates it to RequireConfirmation.
        let cap = suggest_only();
        let audit = MockAuditSink::accepting();
        let obs = Recorder::default();
        let receipt = Gate::new(&cap, &audit, &obs, &FsPathResolver, &StaticMountPolicy::empty())
            .decide_action(
                "some-behaviour",
                BaselineMode::Suggest,
                &scope(&["test.irreversible"]),
                &irreversible_action(),
                &ctx(false, "run-irrev"),
                &DeniedGraph,
            )
            .await
            .expect("accepting sink");
        assert_eq!(receipt.decision, ActionDecision::RequireConfirmation);
    }

    /// A trusted action context targeting a fixed app.
    fn ctx<'a>(external: bool, correlation_id: &'a str) -> ActionContext<'a> {
        ActionContext {
            app_id: "org.lunaris.files",
            external_trigger: external,
            correlation_id,
        }
    }

    /// A declared tool scope containing exactly the given tool names.
    fn scope(names: &[&str]) -> BTreeMap<String, Vec<String>> {
        names.iter().map(|n| (n.to_string(), Vec::new())).collect()
    }

    fn suggest_only() -> Capability {
        Capability::new(AccessTier::Full, ActionPermissions::suggest_only())
    }

    #[tokio::test]
    async fn proposes_and_audits_a_suggest_behaviour() {
        let cap = suggest_only();
        let audit = MockAuditSink::accepting();
        let obs = Recorder::default();

        let receipt = Gate::new(&cap, &audit, &obs, &FsPathResolver, &StaticMountPolicy::empty())
            .decide_action(
                "auto-tag-by-project",
                BaselineMode::Suggest,
                &scope(&["graph.write"]),
                &action(),
                &ctx(false, "run-1"),
                &DeniedGraph,
            )
            .await
            .expect("accepting sink");

        assert_eq!(receipt.decision, ActionDecision::Propose);
        assert_eq!(receipt.audit_index, 0);
        // The decision was recorded, content-free + correlated, before returning.
        let recorded = audit.recorded().await;
        assert_eq!(recorded.len(), 1);
        assert_eq!(recorded[0].structural.subject, "agent.auto-tag-by-project");
        assert_eq!(recorded[0].structural.outcome, "propose");
        assert_eq!(recorded[0].call_chain_id.as_deref(), Some("run-1"));
        assert_eq!(obs.0.lock().unwrap().as_slice(), &[ActionDecision::Propose]);
    }

    #[tokio::test]
    async fn high_impact_tool_requires_confirmation_even_under_supervised() {
        // A supervised behaviour proposing a destructive tool must confirm,
        // not preview-then-execute: the tool name is classified by the
        // shared always-confirm classifier, never trusted from the proposal.
        let cap = Capability::new(
            AccessTier::Full,
            ActionPermissions::new(BaselineMode::Supervised, Vec::<String>::new()),
        );
        let audit = MockAuditSink::accepting();
        let obs = Recorder::default();
        for tool in ["delete_file", "send_email", "pkg_uninstall", "shell_exec", "sudo_thing"] {
            let act = ProposedAction {
                tool: tool.to_string(),
                summary: "x".to_string(),
                arguments: BTreeMap::new(),
            };
            let receipt = Gate::new(&cap, &audit, &obs, &FsPathResolver, &StaticMountPolicy::empty())
                .decide_action(
                    "tidy-downloads",
                    BaselineMode::Supervised,
                    &scope(&[tool]),
                    &act,
                    &ctx(false, "run-x"),
                    &DeniedGraph,
                )
                .await
                .unwrap();
            assert_eq!(
                receipt.decision,
                ActionDecision::RequireConfirmation,
                "destructive tool {tool} must require confirmation"
            );
        }
    }

    #[tokio::test]
    async fn external_trigger_forces_confirmation_regardless_of_mode() {
        // Even an autonomous-for-this-app behaviour must confirm when the
        // run was triggered by external content (a trusted dispatcher fact,
        // not a proposal claim).
        let cap = Capability::new(
            AccessTier::Full,
            ActionPermissions::new(BaselineMode::Suggest, ["org.lunaris.files"]),
        );
        let audit = MockAuditSink::accepting();
        let obs = Recorder::default();
        let receipt = Gate::new(&cap, &audit, &obs, &FsPathResolver, &StaticMountPolicy::empty())
            .decide_action(
                "tidy-downloads",
                BaselineMode::Supervised,
                &scope(&["graph.write"]),
                &action(),
                &ctx(true, "run-2"), // external trigger
                &DeniedGraph,
            )
            .await
            .unwrap();
        assert_eq!(receipt.decision, ActionDecision::RequireConfirmation);
        assert_eq!(
            audit.recorded().await[0].structural.outcome,
            "require-confirmation"
        );
    }

    #[tokio::test]
    async fn fails_closed_when_audit_is_unavailable() {
        let cap = suggest_only();
        let audit = MockAuditSink::failing();
        let obs = Recorder::default();

        let err = Gate::new(&cap, &audit, &obs, &FsPathResolver, &StaticMountPolicy::empty())
            .decide_action(
                "auto-tag-by-project",
                BaselineMode::Suggest,
                &scope(&["graph.write"]),
                &action(),
                &ctx(false, "run-3"),
                &DeniedGraph,
            )
            .await
            .expect_err("failing audit must refuse the action");
        assert!(matches!(err, GateError::AuditUnavailable(_)));
        // Fail-closed: the decision was never handed to the observer.
        assert!(obs.0.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn b1_caps_autonomous_execution_to_confirmation() {
        // The app is granted autonomy; a Supervised-ceiling ordinary action
        // would resolve to PreviewThenExecute by the capability model, but
        // B1 has no argument/world-model validation, so the gate caps any
        // executing decision to explicit confirmation.
        let cap = Capability::new(
            AccessTier::Full,
            ActionPermissions::new(BaselineMode::Suggest, ["org.lunaris.files"]),
        );
        let audit = MockAuditSink::accepting();
        let obs = Recorder::default();
        let receipt = Gate::new(&cap, &audit, &obs, &FsPathResolver, &StaticMountPolicy::empty())
            .decide_action(
                "auto-tag-by-project",
                BaselineMode::Supervised,
                &scope(&["graph.write"]),
                &action(),
                &ctx(false, "run-5"),
                &DeniedGraph,
            )
            .await
            .unwrap();
        assert_eq!(receipt.decision, ActionDecision::RequireConfirmation);
    }

    // --- predict-before-act: a Valid prediction lifts the conservative cap ---

    /// A graph returning canned rows when the query contains a needle.
    struct MockGraph(Vec<(&'static str, Vec<HashMap<String, serde_json::Value>>)>);

    #[async_trait::async_trait]
    impl GraphHandle for MockGraph {
        async fn query(
            &self,
            cypher: &str,
        ) -> Result<Vec<HashMap<String, serde_json::Value>>, GraphError> {
            for (needle, rows) in &self.0 {
                if cypher.contains(needle) {
                    return Ok(rows.clone());
                }
            }
            Ok(Vec::new())
        }
    }

    /// A resolver that accepts an already-canonical absolute path as itself.
    struct IdentityResolver;
    impl PathResolver for IdentityResolver {
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

    fn row(pairs: &[(&str, serde_json::Value)]) -> HashMap<String, serde_json::Value> {
        pairs.iter().map(|(k, v)| (k.to_string(), v.clone())).collect()
    }

    /// The graph for tagging `/proj/a.rs` (under `/proj`) to project `p1`,
    /// with the `FILE_PART_OF` edge present or not.
    fn tag_graph(linked: bool) -> MockGraph {
        MockGraph(vec![
            (
                "n:File {id: '/proj/a.rs'}",
                vec![row(&[("id", "/proj/a.rs".into()), ("path", "/proj/a.rs".into())])],
            ),
            (
                "n:Project {id: 'p1'}",
                vec![row(&[("id", "p1".into()), ("root_path", "/proj".into())])],
            ),
            (
                "count(*) AS cnt",
                vec![row(&[("cnt", serde_json::Value::from(i64::from(linked)))])],
            ),
        ])
    }

    fn graph_write_action() -> ProposedAction {
        ProposedAction {
            tool: "graph.write".to_string(),
            summary: "tag /proj/a.rs as part of p1".to_string(),
            arguments: BTreeMap::from([
                ("file".to_string(), "/proj/a.rs".to_string()),
                ("project".to_string(), "p1".to_string()),
            ]),
        }
    }

    /// The autonomy capability that resolves an ordinary Supervised-ceiling
    /// action to `PreviewThenExecute` (as in `b1_caps_...`).
    fn executing_cap() -> Capability {
        Capability::new(
            AccessTier::Full,
            ActionPermissions::new(BaselineMode::Suggest, ["org.lunaris.files"]),
        )
    }

    #[tokio::test]
    async fn a_valid_prediction_lifts_the_cap() {
        let cap = executing_cap();
        let audit = MockAuditSink::accepting();
        let obs = Recorder::default();
        // The file lies under the project root and is not yet linked.
        let graph = tag_graph(false);
        let receipt = Gate::new(&cap, &audit, &obs, &IdentityResolver, &StaticMountPolicy::empty())
            .decide_action(
                "auto-tag-by-project",
                BaselineMode::Supervised,
                &scope(&["graph.write"]),
                &graph_write_action(),
                &ctx(false, "run-lift"),
                &graph,
            )
            .await
            .unwrap();
        // The world model proved the link safe, so the capability's executing
        // decision stands instead of being capped to confirmation.
        assert_eq!(receipt.decision, ActionDecision::PreviewThenExecute);
        assert_eq!(obs.0.lock().unwrap().as_slice(), &[ActionDecision::PreviewThenExecute]);
    }

    #[tokio::test]
    async fn an_unprovable_action_keeps_the_cap() {
        let cap = executing_cap();
        let audit = MockAuditSink::accepting();
        let obs = Recorder::default();
        // The same action, but the edge already exists: `Not(EdgeExists)`
        // fails, so the prediction is not Valid and the cap stands.
        let graph = tag_graph(true);
        let receipt = Gate::new(&cap, &audit, &obs, &IdentityResolver, &StaticMountPolicy::empty())
            .decide_action(
                "auto-tag-by-project",
                BaselineMode::Supervised,
                &scope(&["graph.write"]),
                &graph_write_action(),
                &ctx(false, "run-cap"),
                &graph,
            )
            .await
            .unwrap();
        assert_eq!(receipt.decision, ActionDecision::RequireConfirmation);
    }

    #[tokio::test]
    async fn an_unregistered_tool_keeps_the_cap() {
        // A tool with no registry rule cannot be proven, so even an executing
        // capability decision is capped.
        let cap = executing_cap();
        let audit = MockAuditSink::accepting();
        let obs = Recorder::default();
        let action = ProposedAction {
            tool: "graph.query".to_string(),
            summary: "x".to_string(),
            arguments: BTreeMap::new(),
        };
        let receipt = Gate::new(&cap, &audit, &obs, &FsPathResolver, &StaticMountPolicy::empty())
            .decide_action(
                "auto-tag-by-project",
                BaselineMode::Supervised,
                &scope(&["graph.query"]),
                &action,
                &ctx(false, "run-unreg"),
                &DeniedGraph,
            )
            .await
            .unwrap();
        assert_eq!(receipt.decision, ActionDecision::RequireConfirmation);
    }

    #[tokio::test]
    async fn an_extra_operand_cannot_ride_on_the_proof() {
        // The action carries an operand the schema does not name. The proof
        // must not pass (the extra operand was never constrained), so the cap
        // stands rather than authorising an under-specified invocation.
        let cap = executing_cap();
        let audit = MockAuditSink::accepting();
        let obs = Recorder::default();
        let graph = tag_graph(false);
        let mut action = graph_write_action();
        action.arguments.insert("rogue".to_string(), "/etc/shadow".to_string());
        let receipt = Gate::new(&cap, &audit, &obs, &IdentityResolver, &StaticMountPolicy::empty())
            .decide_action(
                "auto-tag-by-project",
                BaselineMode::Supervised,
                &scope(&["graph.write"]),
                &action,
                &ctx(false, "run-extra"),
                &graph,
            )
            .await
            .unwrap();
        assert_eq!(receipt.decision, ActionDecision::RequireConfirmation);
    }

    #[tokio::test]
    async fn a_denied_graph_cannot_prove_so_the_cap_stands() {
        // The dispatcher hands a `reads: minimal` behaviour a denying graph
        // handle. The proof reads through that same handle, so even a
        // graph.write with real operands cannot be proven and the cap stands:
        // the proof path is not a read-scope side channel.
        let cap = executing_cap();
        let audit = MockAuditSink::accepting();
        let obs = Recorder::default();
        let receipt = Gate::new(&cap, &audit, &obs, &IdentityResolver, &StaticMountPolicy::empty())
            .decide_action(
                "auto-tag-by-project",
                BaselineMode::Supervised,
                &scope(&["graph.write"]),
                &graph_write_action(),
                &ctx(false, "run-denied"),
                &DeniedGraph,
            )
            .await
            .unwrap();
        assert_eq!(receipt.decision, ActionDecision::RequireConfirmation);
    }

    #[tokio::test]
    async fn refuses_a_tool_outside_the_declared_scope() {
        let cap = suggest_only();
        let audit = MockAuditSink::accepting();
        let obs = Recorder::default();
        // The behaviour declared only graph.query, but proposes graph.write.
        let err = Gate::new(&cap, &audit, &obs, &FsPathResolver, &StaticMountPolicy::empty())
            .decide_action(
                "auto-tag-by-project",
                BaselineMode::Suggest,
                &scope(&["graph.query"]),
                &action(),
                &ctx(false, "run-4"),
                &DeniedGraph,
            )
            .await
            .expect_err("an out-of-scope tool must be refused");
        assert!(matches!(err, GateError::ToolOutOfScope { .. }));
        // The refusal is still audited, and never handed to the observer.
        assert_eq!(
            audit.recorded().await[0].structural.outcome,
            "refused-out-of-scope"
        );
        assert!(obs.0.lock().unwrap().is_empty());
    }
}
