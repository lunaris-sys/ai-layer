//! Dry-run action executor: turns a lifted gate decision into the concrete
//! graph write it authorises, and records that write WITHOUT performing any
//! I/O.
//!
//! This is the first half of the executor the gate's lift anticipates (see the
//! "Executor obligations" contract in [`crate::gate`]). It honours obligation 1
//! — **execute exactly the proven effect** — by deriving the write solely from
//! the trusted, registry-resolved schema for the invoked action: the single
//! `AssertEdge` effect gives the edge type and the endpoint binds, the schema's
//! `NodeExists` preconditions give those binds' node types, and only the
//! concrete node ids come from the (untrusted) invocation arguments. A schema
//! whose effect is anything other than one `AssertEdge` is refused, so a
//! different mutation can never ride on the proof.
//!
//! It is deliberately dry-run: it computes the planned write and returns it for
//! logging / the activity surface, but performs no write.
//!
//! ## What going live still needs (the strict-create gap)
//!
//! The `graph.write` rule is **strict-create**: its proof includes
//! `Not(EdgeExists)`, and its derived compensation (`RetractEdge`) is sound only
//! because the action is the one that created the edge. The os-sdk relation
//! client persists with the daemon's idempotent `MERGE`, which re-checks the
//! *endpoints* exist but **not** that the edge is absent. So a plain live wiring
//! would treat an edge created concurrently after the proof as a silent success
//! and leave a later compensation able to retract an edge this action did not
//! create. The dry-run report therefore carries
//! [`DryRunReport::conditional_on_absent_edge`]: the live executor must enforce
//! that absence atomically (a conditional create-or-conflict op), or the effect
//! must be re-modelled as an idempotent ensure-edge whose compensation only
//! undoes a create it actually performed. Resolving that, plus per-tool
//! scope-value enforcement (obligation 3), is the live executor increment;
//! nothing here executes.

use std::collections::BTreeMap;

use lunaris_ai_core::capability::ActionDecision;

use crate::gate::ProposedAction;
use crate::registry::{self, TrustedActionSchema};
use crate::world::{Effect, Predicate};

/// The concrete relation write a proven action would perform: the namespaced
/// endpoint entity types, their resolved node ids, and the edge type. The
/// shape the daemon's relation-write socket expects.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelationWrite {
    /// The source node's namespaced entity type (e.g. `system.File`).
    pub from_type: String,
    /// The source node's concrete id.
    pub from_id: String,
    /// The target node's namespaced entity type (e.g. `system.Project`).
    pub to_type: String,
    /// The target node's concrete id.
    pub to_id: String,
    /// The relation (edge) type to create.
    pub relation_type: String,
}

/// Why the executor could not turn a decision into a concrete write. Every
/// variant is fail-closed: the executor produces no write rather than guess.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ExecError {
    /// No registry rule backs the invoked action, so there is nothing to
    /// execute (the gate would not have lifted it either).
    #[error("no registry rule for action '{0}'")]
    NoRule(String),
    /// The action's schema is not a single `AssertEdge`, the only effect this
    /// executor performs. Refused rather than reinterpreted.
    #[error("action '{0}' has no single AssertEdge effect the executor can perform")]
    UnsupportedEffect(String),
    /// An endpoint bind has no `NodeExists` precondition, so its node type
    /// cannot be resolved from the trusted schema.
    #[error("bind '{0}' has no NodeExists precondition, so its node type is unknown")]
    UnknownBindLabel(String),
    /// An endpoint bind has no value in the invocation arguments, so its node
    /// id is unresolved.
    #[error("argument '{0}' is missing, so its node id is unresolved")]
    MissingArgument(String),
}

/// What a dry run would do, surfaced for logging / the activity view. Holds the
/// concrete write and never performs it.
///
/// This is a **non-authoritative record**, not an execution authority. It does
/// not carry the full proof: the gate's lift also rested on point-in-time
/// preconditions (for `graph.write`, `PathUnderField` proving the file lies
/// under the project root) that the report deliberately omits, because the proof
/// is a point-in-time slice with no graph snapshot (gap A2). A live executor
/// must therefore re-run the complete trusted precondition validation atomically
/// at write time (the gate's obligation 2) and never write straight from this
/// report; the report is for showing the user / activity log what a proven
/// decision would do.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DryRunReport {
    /// The relation the live executor would create.
    pub write: RelationWrite,
    /// Whether the proof required the edge to be **absent** (a strict-create
    /// `Not(EdgeExists)` precondition). When true, the live executor must create
    /// only if the edge is absent and treat a concurrently-created edge as a
    /// conflict, not a silent success — a bare idempotent `MERGE` would not
    /// honour the strict-create semantics or keep the derived compensation safe.
    pub conditional_on_absent_edge: bool,
}

/// Whether the schema proves the asserted edge is **absent** (a strict-create
/// precondition `Not(EdgeExists)` matching the single `AssertEdge` effect). The
/// live executor must enforce this atomically: create only if absent, else
/// conflict. A plain idempotent `MERGE` would silently treat a concurrently
/// created edge as success and make a later compensation (retract) unsafe.
fn create_is_conditional_on_absence(schema: &TrustedActionSchema) -> bool {
    let s = schema.schema();
    let Some(Effect::AssertEdge { from, edge, to }) = s.effects.first() else {
        return false;
    };
    s.preconditions.iter().any(|p| {
        if let Predicate::Not(inner) = p {
            if let Predicate::EdgeExists {
                from: f,
                edge: e,
                to: t,
            } = inner.as_ref()
            {
                return f == from && e == edge && t == to;
            }
        }
        false
    })
}

/// Map a bare world-model label (`File`) to the daemon's namespaced entity type
/// (`system.File`). The world model and graph node tables use bare labels; the
/// write socket's relation allowlist is namespaced, so the boundary is crossed
/// here, once.
fn namespaced(label: &str) -> String {
    format!("system.{label}")
}

/// Derive the concrete [`RelationWrite`] for a trusted action schema and the
/// invocation's (untrusted) arguments.
///
/// The edge type and the two endpoint binds come from the schema's single
/// `AssertEdge` effect; each bind's node type comes from a matching
/// `NodeExists` precondition in the same trusted schema (never the arguments);
/// only the node ids come from the arguments. Anything that cannot be resolved
/// fail-closes to an [`ExecError`].
pub(crate) fn plan_relation_write(
    schema: &TrustedActionSchema,
    arguments: &BTreeMap<String, String>,
) -> Result<RelationWrite, ExecError> {
    let s = schema.schema();

    // Obligation 1: execute EXACTLY the proven effect. The only effect this
    // executor performs is a single AssertEdge; anything else (a node mutation,
    // a field set, more than one effect) is refused, so no other mutation can
    // ride on the proof.
    let (from_bind, edge, to_bind) = match s.effects.as_slice() {
        [Effect::AssertEdge { from, edge, to }] => (from, edge, to),
        _ => return Err(ExecError::UnsupportedEffect(s.action.clone())),
    };

    // Each endpoint's node label comes from the trusted schema's NodeExists
    // preconditions, the authoritative source of the bind's type.
    let label_for = |bind: &str| -> Option<&str> {
        s.preconditions.iter().find_map(|p| match p {
            Predicate::NodeExists { bind: b, label } if b == bind => Some(label.as_str()),
            _ => None,
        })
    };
    let from_label =
        label_for(from_bind).ok_or_else(|| ExecError::UnknownBindLabel(from_bind.clone()))?;
    let to_label =
        label_for(to_bind).ok_or_else(|| ExecError::UnknownBindLabel(to_bind.clone()))?;

    // Only the concrete ids come from the (untrusted) operands.
    let from_id = arguments
        .get(from_bind)
        .ok_or_else(|| ExecError::MissingArgument(from_bind.clone()))?;
    let to_id = arguments
        .get(to_bind)
        .ok_or_else(|| ExecError::MissingArgument(to_bind.clone()))?;

    Ok(RelationWrite {
        from_type: namespaced(from_label),
        from_id: from_id.clone(),
        to_type: namespaced(to_label),
        to_id: to_id.clone(),
        relation_type: edge.clone(),
    })
}

/// Dry-run the executor for one gated action.
///
/// Returns a plan **only** for `PreviewThenExecute`, the single decision the
/// gate emits that carries a successful predict-before-act proof: that lift is
/// reached only when the world model proved this invocation safe (its operands
/// hold against the trusted schema and the real graph), so deriving a concrete
/// write from those operands is sound. Every other decision yields `Ok(None)`:
/// `RequireConfirmation` is the gate's unproven cap (an override, or a failed or
/// absent proof), so its operands may never have been validated and an
/// executable write derived from them could let an approval corrupt the graph (a
/// confirmation surface shows the proposal summary, not a write); `Propose` is
/// the manual Suggest flow; and `Proceed` is never emitted by the gate (a proven
/// autonomous decision is capped to `PreviewThenExecute`).
///
/// The schema is resolved independently from the trusted registry, keyed by the
/// same tool the gate proved (defence in depth: the executor binds to the
/// registry, not to a caller-passed schema). Performs **no I/O**: it records
/// what the live executor would write.
pub(crate) fn dry_run(
    action: &ProposedAction,
    decision: ActionDecision,
) -> Result<Option<DryRunReport>, ExecError> {
    // Only PreviewThenExecute carries a successful predict-before-act proof
    // (the gate lifts to it solely when the world model validated this
    // invocation's operands against the trusted schema and the real graph).
    // Every other final decision is non-executable here: RequireConfirmation is
    // the unproven cap, Propose is the manual flow, and Proceed is never emitted
    // by the gate.
    if decision != ActionDecision::PreviewThenExecute {
        return Ok(None);
    }
    let schema =
        registry::lookup(&action.tool).ok_or_else(|| ExecError::NoRule(action.tool.clone()))?;
    let write = plan_relation_write(&schema, &action.arguments)?;
    let conditional_on_absent_edge = create_is_conditional_on_absence(&schema);
    Ok(Some(DryRunReport {
        write,
        conditional_on_absent_edge,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph_write_action(args: &[(&str, &str)]) -> ProposedAction {
        ProposedAction {
            tool: "graph.write".to_string(),
            summary: "link file to project".to_string(),
            arguments: args
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        }
    }

    #[test]
    fn plans_the_file_part_of_write_from_the_trusted_schema() {
        let schema = registry::lookup("graph.write").unwrap();
        let args: BTreeMap<String, String> =
            [("file", "f1"), ("project", "p1")]
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect();
        let write = plan_relation_write(&schema, &args).unwrap();
        assert_eq!(
            write,
            RelationWrite {
                from_type: "system.File".to_string(),
                from_id: "f1".to_string(),
                to_type: "system.Project".to_string(),
                to_id: "p1".to_string(),
                relation_type: "FILE_PART_OF".to_string(),
            }
        );
    }

    #[test]
    fn a_missing_argument_fails_closed() {
        let schema = registry::lookup("graph.write").unwrap();
        // Only the `file` id is supplied; `project` is missing.
        let args: BTreeMap<String, String> =
            [("file", "f1")].iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
        assert_eq!(
            plan_relation_write(&schema, &args),
            Err(ExecError::MissingArgument("project".to_string()))
        );
    }

    #[test]
    fn dry_run_plans_only_for_the_proven_preview_decision() {
        let action = graph_write_action(&[("file", "f1"), ("project", "p1")]);

        // PreviewThenExecute is the only decision carrying a successful proof, so
        // it is the only one that produces an executable write. The built-in link
        // rule is strict-create, so the plan flags that the live executor must
        // check the edge is absent (not a bare MERGE).
        let report = dry_run(&action, ActionDecision::PreviewThenExecute)
            .unwrap()
            .unwrap();
        assert_eq!(report.write.relation_type, "FILE_PART_OF");
        assert!(
            report.conditional_on_absent_edge,
            "the FILE_PART_OF rule proves Not(EdgeExists), so the create is conditional"
        );

        // Every non-proven decision plans nothing: RequireConfirmation is the
        // unproven cap (operands unvalidated), Propose is manual, and Proceed is
        // not emitted by the gate. None must derive a write from the operands.
        assert_eq!(
            dry_run(&action, ActionDecision::RequireConfirmation).unwrap(),
            None,
            "an unproven confirmation must not yield an executable write"
        );
        assert_eq!(dry_run(&action, ActionDecision::Propose).unwrap(), None);
        assert_eq!(dry_run(&action, ActionDecision::Proceed).unwrap(), None);
    }

    #[test]
    fn dry_run_refuses_an_unregistered_tool() {
        let action = ProposedAction {
            tool: "fs.delete".to_string(),
            summary: "delete".to_string(),
            arguments: BTreeMap::new(),
        };
        assert_eq!(
            dry_run(&action, ActionDecision::PreviewThenExecute),
            Err(ExecError::NoRule("fs.delete".to_string()))
        );
    }
}
