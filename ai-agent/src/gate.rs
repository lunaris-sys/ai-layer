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
//! overwrite (design-doc gap F4). That reversibility judgment belongs to
//! the **B2 world-model action schema** (preconditions + effects), not to
//! a name heuristic. Until it lands, this gate enforces a conservative
//! guard: it **caps any executing decision (PreviewThenExecute / Proceed)
//! to explicit confirmation**, so no action auto-executes while its
//! argument-level safety is unprovable. Suggest/Propose is unaffected (the
//! user executes manually). B2 replaces the blanket cap with per-action
//! argument + reversibility validation against the declared scope values.

use std::collections::BTreeMap;

use lunaris_ai_core::audit::{behaviour_action_event, AuditSink};
use lunaris_ai_core::capability::{ActionDecision, ActionKind, BaselineMode, Capability};
use lunaris_ai_core::mcp::{AlwaysConfirm, AlwaysConfirmReason};

use crate::seams::GateObserver;

/// An action a behaviour proposes. Carries only what a proposer may
/// legitimately state — the tool/operation it wants to invoke and a
/// human-facing summary. It deliberately carries **no authorization
/// inputs**: not the target app id, not a risk class, not an
/// external-content flag. Every input that steers the gate decision is
/// trusted and arrives via [`ActionContext`], never the proposal — an
/// untrusted proposal must not be able to pick which per-app grant
/// applies, label its own risk, or claim non-external provenance. The
/// `summary` is for the proposal/preview UI and is never audited (the
/// audit subject is content-free).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProposedAction {
    /// The MCP tool / operation the behaviour wants to invoke. Classified
    /// by the shared always-confirm classifier and checked against the
    /// behaviour's declared `tools` scope; the *real* tool is re-classified
    /// at MCP dispatch.
    pub tool: String,
    /// Human-facing description for the proposal/preview surface.
    pub summary: String,
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

/// The action gate, holding the long-lived collaborators (the capability,
/// the audit sink, and the observer seam). The engine constructs one and
/// calls [`Gate::decide_action`] per proposed action.
pub struct Gate<'a> {
    capability: &'a Capability,
    audit: &'a dyn AuditSink,
    observer: &'a dyn GateObserver,
}

impl<'a> Gate<'a> {
    /// Build a gate over its collaborators.
    pub fn new(
        capability: &'a Capability,
        audit: &'a dyn AuditSink,
        observer: &'a dyn GateObserver,
    ) -> Self {
        Self {
            capability,
            audit,
            observer,
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

        // Classify the proposed tool with the shared always-confirm
        // classifier (the same one MCP dispatch uses) — never a risk class
        // taken from the proposal. Combine with the mode (ceiling ∧ grant)
        // for the trusted target app and the external-trigger override.
        let kind = action_kind_for_tool(&action.tool);
        let decision =
            self.capability
                .decide_for_behaviour(ctx.app_id, kind, ctx.external_trigger, ceiling);

        // B1 conservative execution guard. Without structured action
        // arguments and the world-model action schema (B2), the gate cannot
        // prove a state-changing action is within its declared scope values
        // and reversible, so it must not authorise *autonomous* execution.
        // Any executing decision is capped to explicit confirmation;
        // Suggest/Propose is unaffected, because there the user executes
        // manually (the human is the check). B2 replaces this blanket cap
        // with per-action argument + reversibility validation.
        let decision = match decision {
            ActionDecision::PreviewThenExecute | ActionDecision::Proceed => {
                ActionDecision::RequireConfirmation
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

    use audit_proto::MockAuditSink;
    use lunaris_ai_core::capability::{AccessTier, ActionPermissions};

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
        }
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

        let receipt = Gate::new(&cap, &audit, &obs)
            .decide_action(
                "auto-tag-by-project",
                BaselineMode::Suggest,
                &scope(&["graph.write"]),
                &action(),
                &ctx(false, "run-1"),
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
            };
            let receipt = Gate::new(&cap, &audit, &obs)
                .decide_action(
                    "tidy-downloads",
                    BaselineMode::Supervised,
                    &scope(&[tool]),
                    &act,
                    &ctx(false, "run-x"),
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
        let receipt = Gate::new(&cap, &audit, &obs)
            .decide_action(
                "tidy-downloads",
                BaselineMode::Supervised,
                &scope(&["graph.write"]),
                &action(),
                &ctx(true, "run-2"), // external trigger
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

        let err = Gate::new(&cap, &audit, &obs)
            .decide_action(
                "auto-tag-by-project",
                BaselineMode::Suggest,
                &scope(&["graph.write"]),
                &action(),
                &ctx(false, "run-3"),
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
        let receipt = Gate::new(&cap, &audit, &obs)
            .decide_action(
                "auto-tag-by-project",
                BaselineMode::Supervised,
                &scope(&["graph.write"]),
                &action(),
                &ctx(false, "run-5"),
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
        let err = Gate::new(&cap, &audit, &obs)
            .decide_action(
                "auto-tag-by-project",
                BaselineMode::Suggest,
                &scope(&["graph.query"]),
                &action(),
                &ctx(false, "run-4"),
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
