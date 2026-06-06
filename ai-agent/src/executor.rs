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
//! undoes a create it actually performed. Resolving that absence check, plus
//! re-running the full trusted precondition proof atomically at write time
//! (obligation 2, which the live write inherently couples to), is the live
//! executor increment; nothing here executes.
//!
//! Per-tool scope-value enforcement (obligation 3) **is** done here: the planned
//! write is checked against the behaviour's declared `graph.write` scope, so it
//! can only ever target a relation/entity the behaviour was granted, not merely
//! one it holds the `graph.write` tool name for.

use std::collections::BTreeMap;
use std::time::Duration;

use async_trait::async_trait;
use lunaris_ai_core::capability::{ActionDecision, BaselineMode, Capability};

use crate::gate::{resolved_action_kind, ProposedAction};
use crate::registry::{self, TrustedActionSchema};
use crate::seams::GraphHandle;
use crate::slice::{build_slice_trusted, MountPolicy, PathResolver};
use crate::world::{self, Effect, EvalContext, Predicate};

/// Wall-clock bound on the live re-validation read (graph slice + path
/// resolution), mirroring the gate's proof timeout. A stalled dependency must
/// fail closed (the write is refused) rather than park the executor.
const REVALIDATION_TIMEOUT: Duration = Duration::from_secs(5);

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
    /// The planned write names a target entity or relation type the behaviour
    /// did not declare in its `graph.write` tool scope. The gate enforces the
    /// tool *name*; the executor enforces the scope *values* (obligation 3), so
    /// a behaviour cannot write a relation/target it was not granted.
    #[error("tool '{tool}' scope does not grant '{token}'")]
    ScopeViolation {
        /// The tool whose declared scope was exceeded.
        tool: String,
        /// The target label or relation type the scope did not name.
        token: String,
    },
    /// The predict-before-act proof no longer holds against the current graph:
    /// a precondition went stale between the gate decision and the write (a node
    /// removed, a path moved, the edge created concurrently). The write is
    /// refused fail-closed rather than acting on a stale proof.
    #[error("the action's proof no longer holds against the current graph")]
    ProofStale,
    /// The write itself failed at the graph boundary.
    #[error("write failed: {0}")]
    Write(String),
    /// The re-validation read (graph slice + path resolution) did not finish in
    /// time, so the proof could not be re-established. Fail-closed: a stalled
    /// knowledge socket or a slow path lookup must not park the executor.
    #[error("re-validation timed out before the write")]
    RevalidationTimeout,
}

/// A failure to persist a planned relation write.
#[derive(Debug, thiserror::Error)]
pub enum WriteError {
    /// The write could not be performed (daemon rejected it, or transport).
    #[error("relation write failed: {0}")]
    Failed(String),
}

/// Whether a write created the edge or found it already present. The daemon's
/// conditional create reports this atomically for a single attempt (and never
/// double-creates). It is NOT durable across an at-least-once retry: a create
/// whose response is lost and is retried reports `AlreadyExists` the second
/// time, so this alone does not make a compensator that survives retries safe.
/// Durable operation identity (an idempotency key) is the deferred follow-up;
/// the executor's pre-write re-validation is the interim guard (a retry whose
/// edge now exists fails its proof and writes nothing).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriteOutcome {
    /// This write created the edge.
    Created,
    /// The edge already existed; the write was an idempotent no-op.
    AlreadyExists,
}

/// The seam through which the live executor performs an authorised write. The
/// production impl wraps the os-sdk graph write client (the knowledge daemon's
/// write socket); tests inject a mock that records the write without I/O. Kept
/// separate from the read-only [`GraphHandle`] so the proof path can never write
/// and a writer is only ever reached after a re-validated proof.
#[async_trait]
pub trait RelationWriter: Send + Sync {
    /// Persist the relation, reporting whether it created the edge or found it
    /// already present. Idempotent at the daemon (a strict conditional create),
    /// so a transport retry re-confirms (`AlreadyExists`) rather than duplicates.
    async fn write_relation(&self, write: &RelationWrite) -> Result<WriteOutcome, WriteError>;
}

/// A write the live executor performed: the relation and whether it was created
/// or already present. Returned so a caller (and future compensation) acts only
/// on a real create.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutedWrite {
    /// The relation that was written.
    pub write: RelationWrite,
    /// Whether this call created the edge or found it present.
    pub outcome: WriteOutcome,
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

/// Enforce the behaviour's per-tool scope *values* on a planned write
/// (obligation 3). The gate already checked the tool name; here the concrete
/// target and relation must be ones the behaviour declared, so a behaviour
/// granted `graph.write: [Project, FILE_PART_OF]` cannot write some other
/// relation or target even though it holds `graph.write`.
///
/// An empty scope list means the tool is granted without a finer restriction
/// (the manifest convention), so it passes. Otherwise both the target entity's
/// bare label and the relation type must appear in the scope list.
fn enforce_tool_scope(write: &RelationWrite, tool: &str, scope: &[String]) -> Result<(), ExecError> {
    if scope.is_empty() {
        return Ok(());
    }
    let to_label = write
        .to_type
        .strip_prefix("system.")
        .unwrap_or(&write.to_type);
    for token in [to_label, write.relation_type.as_str()] {
        if !scope.iter().any(|s| s == token) {
            return Err(ExecError::ScopeViolation {
                tool: tool.to_string(),
                token: token.to_string(),
            });
        }
    }
    Ok(())
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
/// registry, not to a caller-passed schema). The planned write is also checked
/// against `tool_scope`, the behaviour's declared `graph.write` scope values
/// (obligation 3). Performs **no I/O**: it records what the live executor would
/// write.
pub(crate) fn dry_run(
    action: &ProposedAction,
    decision: ActionDecision,
    tool_scope: &[String],
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
    enforce_tool_scope(&write, &action.tool, tool_scope)?;
    let conditional_on_absent_edge = create_is_conditional_on_absence(&schema);
    Ok(Some(DryRunReport {
        write,
        conditional_on_absent_edge,
    }))
}

/// The live action executor: re-validates a proven decision against the current
/// graph (obligation 2) and performs the authorised write through a
/// [`RelationWriter`].
///
/// It holds the long-lived collaborators; [`execute`](LiveExecutor::execute)
/// takes the per-call decision and the behaviour-scoped graph handle, mirroring
/// how the gate is structured. The capability, path, and mount collaborators are
/// the SAME the gate proved with, so the re-validation classifies and resolves
/// identically.
pub struct LiveExecutor<'a> {
    capability: &'a Capability,
    paths: &'a dyn PathResolver,
    mounts: &'a dyn MountPolicy,
    writer: &'a dyn RelationWriter,
}

impl<'a> LiveExecutor<'a> {
    /// Build a live executor over its collaborators.
    pub fn new(
        capability: &'a Capability,
        paths: &'a dyn PathResolver,
        mounts: &'a dyn MountPolicy,
        writer: &'a dyn RelationWriter,
    ) -> Self {
        Self {
            capability,
            paths,
            mounts,
            writer,
        }
    }

    /// Execute a gated decision: derive the planned write (only for a proven
    /// `PreviewThenExecute`, with scope enforced), re-run the full trusted proof
    /// against the CURRENT graph, and write only if it still holds. Returns the
    /// write performed, `None` for a non-executable decision, or an
    /// [`ExecError`] (the proof went stale, the re-validation timed out, scope
    /// was exceeded, or the write failed). The graph handle is the
    /// behaviour-scoped one, so the re-validation reads no more than the
    /// behaviour may.
    ///
    /// The edge create itself is atomic and reports whether it created the edge:
    /// the daemon's conditional create is a single statement on its serial graph
    /// thread, so it cannot double-create and tells the writer `Created` vs
    /// `AlreadyExists` (so compensation only ever undoes a real create). What the
    /// re-validation **narrows but does not** make atomic is the rest of the
    /// proof: a `PathUnderField` fact (the file still lies under the project
    /// root) can change between this re-check and the write, since the daemon's
    /// create enforces only endpoint existence and edge absence, not the agent's
    /// path-prefix predicate. Fully closing that needs a graph snapshot/version
    /// the engine does not expose (gap A2); it is why nothing wires this live yet.
    pub async fn execute(
        &self,
        action: &ProposedAction,
        decision: ActionDecision,
        tool_scope: &[String],
        graph: &dyn GraphHandle,
        app_id: &str,
        external_trigger: bool,
        ceiling: BaselineMode,
    ) -> Result<Option<ExecutedWrite>, ExecError> {
        // Plan: only a proven PreviewThenExecute yields a write, and the planned
        // write is checked against the behaviour's declared scope (obligation 3).
        let Some(report) = dry_run(action, decision, tool_scope)? else {
            return Ok(None);
        };

        // Re-run the trusted proof against the live graph, right before the
        // write, so a precondition that went stale since the gate decision (a
        // node removed, the file moved out from under the project root, the edge
        // created concurrently) refuses the write. The schema and the slice are
        // rebuilt from the trusted registry, never the proposal. The read is
        // time-bounded (like the gate's proof): a stalled knowledge socket or a
        // slow path lookup fails closed rather than parking the executor.
        let trusted = registry::lookup(&action.tool)
            .ok_or_else(|| ExecError::NoRule(action.tool.clone()))?;
        let slice = tokio::time::timeout(
            REVALIDATION_TIMEOUT,
            build_slice_trusted(
                &trusted,
                &action.tool,
                &action.arguments,
                graph,
                self.paths,
                self.mounts,
            ),
        )
        .await
        .map_err(|_| ExecError::RevalidationTimeout)?;
        let (state, bindings) = slice.map_err(|_| ExecError::ProofStale)?;
        let eval = EvalContext {
            capability: self.capability,
            action_id: &action.tool,
            app_id,
            action_kind: resolved_action_kind(&action.tool),
            external_trigger,
            ceiling,
        };
        if !world::predict(trusted.schema(), &bindings, &state, &eval).is_valid() {
            return Err(ExecError::ProofStale);
        }

        // The proof still holds: perform the authorised write. The outcome
        // (created vs already-present) is carried back so only a real create is
        // ever compensated.
        let outcome = self
            .writer
            .write_relation(&report.write)
            .await
            .map_err(|e| ExecError::Write(e.to_string()))?;
        Ok(Some(ExecutedWrite {
            write: report.write,
            outcome,
        }))
    }
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

    /// The auto-tag behaviour's declared `graph.write` scope.
    fn auto_tag_scope() -> Vec<String> {
        vec!["Project".to_string(), "FILE_PART_OF".to_string()]
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
        let scope = auto_tag_scope();

        // PreviewThenExecute is the only decision carrying a successful proof, so
        // it is the only one that produces an executable write. The built-in link
        // rule is strict-create, so the plan flags that the live executor must
        // check the edge is absent (not a bare MERGE).
        let report = dry_run(&action, ActionDecision::PreviewThenExecute, &scope)
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
            dry_run(&action, ActionDecision::RequireConfirmation, &scope).unwrap(),
            None,
            "an unproven confirmation must not yield an executable write"
        );
        assert_eq!(dry_run(&action, ActionDecision::Propose, &scope).unwrap(), None);
        assert_eq!(dry_run(&action, ActionDecision::Proceed, &scope).unwrap(), None);
    }

    #[test]
    fn dry_run_refuses_an_unregistered_tool() {
        let action = ProposedAction {
            tool: "fs.delete".to_string(),
            summary: "delete".to_string(),
            arguments: BTreeMap::new(),
        };
        assert_eq!(
            dry_run(&action, ActionDecision::PreviewThenExecute, &[]),
            Err(ExecError::NoRule("fs.delete".to_string()))
        );
    }

    #[test]
    fn dry_run_enforces_the_declared_tool_scope() {
        let action = graph_write_action(&[("file", "f1"), ("project", "p1")]);

        // A scope that grants the relation but not the target entity is refused:
        // the executor enforces scope values (obligation 3), not just the name.
        let scope = vec!["FILE_PART_OF".to_string()];
        assert_eq!(
            dry_run(&action, ActionDecision::PreviewThenExecute, &scope),
            Err(ExecError::ScopeViolation {
                tool: "graph.write".to_string(),
                token: "Project".to_string(),
            })
        );

        // A scope missing the relation type is likewise refused.
        let scope = vec!["Project".to_string()];
        assert_eq!(
            dry_run(&action, ActionDecision::PreviewThenExecute, &scope),
            Err(ExecError::ScopeViolation {
                tool: "graph.write".to_string(),
                token: "FILE_PART_OF".to_string(),
            })
        );
    }

    #[test]
    fn an_empty_scope_grants_without_restriction() {
        let action = graph_write_action(&[("file", "f1"), ("project", "p1")]);
        // The manifest convention: an empty scope list grants the tool without a
        // finer restriction, so the write plans.
        let report = dry_run(&action, ActionDecision::PreviewThenExecute, &[])
            .unwrap()
            .unwrap();
        assert_eq!(report.write.relation_type, "FILE_PART_OF");
    }

    // ---- LiveExecutor: obligation-2 re-validation + write ----

    use crate::seams::GraphError;
    use crate::slice::{SliceError, StaticMountPolicy};
    use lunaris_ai_core::capability::{AccessTier, ActionPermissions};
    use std::collections::HashMap;
    use std::sync::Mutex;

    /// A graph returning canned rows when the query contains a needle (the same
    /// shape the gate's proof tests use).
    struct MockGraph(Vec<(&'static str, Vec<HashMap<String, serde_json::Value>>)>);

    #[async_trait]
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

    /// Accepts an already-canonical absolute path as itself.
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

    /// The graph for tagging `/proj/a.rs` (under `/proj`) to project `p1`, with
    /// the `FILE_PART_OF` edge present or not.
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

    fn tag_action() -> ProposedAction {
        graph_write_action(&[("file", "/proj/a.rs"), ("project", "p1")])
    }

    fn executing_cap() -> Capability {
        Capability::new(
            AccessTier::Full,
            ActionPermissions::new(BaselineMode::Suggest, ["org.lunaris.files"]),
        )
    }

    /// Records each write without performing I/O.
    #[derive(Default)]
    struct MockWriter(Mutex<Vec<RelationWrite>>);

    #[async_trait]
    impl RelationWriter for MockWriter {
        async fn write_relation(&self, write: &RelationWrite) -> Result<WriteOutcome, WriteError> {
            self.0.lock().unwrap().push(write.clone());
            Ok(WriteOutcome::Created)
        }
    }

    #[tokio::test]
    async fn live_executor_writes_a_revalidated_proof() {
        let cap = executing_cap();
        let writer = MockWriter::default();
        let graph = tag_graph(false); // file under root, not yet linked
        let (resolver, mounts) = (IdentityResolver, StaticMountPolicy::empty());
        let exec = LiveExecutor::new(&cap, &resolver, &mounts, &writer);
        let written = exec
            .execute(
                &tag_action(),
                ActionDecision::PreviewThenExecute,
                &auto_tag_scope(),
                &graph,
                "org.lunaris.files",
                false,
                BaselineMode::Supervised,
            )
            .await
            .unwrap()
            .unwrap();
        assert_eq!(written.write.relation_type, "FILE_PART_OF");
        assert_eq!(written.outcome, WriteOutcome::Created);
        let recorded = writer.0.lock().unwrap();
        assert_eq!(recorded.len(), 1, "exactly one write performed");
        assert_eq!(recorded[0].to_type, "system.Project");
        assert_eq!(recorded[0].from_id, "/proj/a.rs");
    }

    #[tokio::test]
    async fn live_executor_refuses_a_stale_proof() {
        let cap = executing_cap();
        let writer = MockWriter::default();
        // The edge already exists, so Not(EdgeExists) fails: the proof is stale.
        let graph = tag_graph(true);
        let (resolver, mounts) = (IdentityResolver, StaticMountPolicy::empty());
        let exec = LiveExecutor::new(&cap, &resolver, &mounts, &writer);
        let result = exec
            .execute(
                &tag_action(),
                ActionDecision::PreviewThenExecute,
                &auto_tag_scope(),
                &graph,
                "org.lunaris.files",
                false,
                BaselineMode::Supervised,
            )
            .await;
        assert!(matches!(result, Err(ExecError::ProofStale)));
        assert!(
            writer.0.lock().unwrap().is_empty(),
            "no write may happen on a stale proof"
        );
    }

    #[tokio::test]
    async fn live_executor_skips_a_non_executing_decision() {
        let cap = executing_cap();
        let writer = MockWriter::default();
        let graph = tag_graph(false);
        let (resolver, mounts) = (IdentityResolver, StaticMountPolicy::empty());
        let exec = LiveExecutor::new(&cap, &resolver, &mounts, &writer);
        let written = exec
            .execute(
                &tag_action(),
                ActionDecision::RequireConfirmation,
                &auto_tag_scope(),
                &graph,
                "org.lunaris.files",
                false,
                BaselineMode::Supervised,
            )
            .await
            .unwrap();
        assert_eq!(written, None);
        assert!(writer.0.lock().unwrap().is_empty());
    }
}
