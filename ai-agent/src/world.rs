//! The declarative world model: action schemas with preconditions and
//! effects in a fixed, closed predicate DSL, plus a deterministic
//! interpreter that evaluates them against a world state and the capability
//! layer.
//!
//! Binding idea: this is the one declarative rule layer, never generated
//! code. The model may *instantiate* these predicate kinds with parameters;
//! adding a new predicate KIND is a reviewed engine change, never something
//! the model emits. The interpreter is a pure function — given a grounded
//! action (its parameter bindings), a world state, and the capability, it
//! reports whether the preconditions hold and what the effects would make
//! true. So it is deterministic, auditable, and sandbox-safe.
//!
//! Scope of this increment: the types and the interpreter over a synthetic
//! in-memory [`WorldState`] (test fixtures). Populating the state from a
//! bounded Knowledge-Graph slice, and wiring predict-before-act into the
//! engine/gate (so a proven-safe action lifts the suggest-mode cap), are
//! later increments behind the same pure interpreter.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path};

use lunaris_ai_core::capability::{ActionDecision, ActionKind, BaselineMode, Capability};

/// A node in the world state: a labelled entity with string fields,
/// identified by an opaque id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Node {
    /// The entity's opaque id (what bindings point at).
    pub id: String,
    /// The Knowledge-Graph node label, e.g. `File` or `Project`.
    pub label: String,
    /// The entity's fields as strings (the read DSL returns string cells).
    pub fields: BTreeMap<String, String>,
}

impl Node {
    /// A node with a label and no fields.
    pub fn new(id: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            fields: BTreeMap::new(),
        }
    }

    /// Builder: set a field.
    pub fn with_field(mut self, field: impl Into<String>, value: impl Into<String>) -> Self {
        self.fields.insert(field.into(), value.into());
        self
    }
}

/// The state the interpreter evaluates against: labelled nodes, directed
/// typed edges between node ids, and the set of read-only path prefixes.
///
/// In this increment it is a synthetic in-memory set built by tests; a later
/// increment populates it from a bounded KG slice, behind the same shape.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WorldState {
    nodes: BTreeMap<String, Node>,
    edges: BTreeSet<(String, String, String)>,
    read_only_prefixes: BTreeSet<String>,
    /// Whether the read-only mount policy has actually been loaded into this
    /// state. Default `false`: until the policy is known, every path is
    /// treated as read-only (fail closed), so a truncated or unloaded slice
    /// cannot be mistaken for "no read-only mounts exist".
    read_only_loaded: bool,
}

impl WorldState {
    /// An empty state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: add a node.
    pub fn with_node(mut self, node: Node) -> Self {
        self.nodes.insert(node.id.clone(), node);
        self
    }

    /// Builder: add a directed, typed edge between two node ids.
    pub fn with_edge(
        mut self,
        from: impl Into<String>,
        edge: impl Into<String>,
        to: impl Into<String>,
    ) -> Self {
        self.edges.insert((from.into(), edge.into(), to.into()));
        self
    }

    /// Builder: mark a path prefix read-only. Loading a prefix also marks the
    /// read-only policy as known (so paths outside it are genuinely writable,
    /// not merely unjudged).
    pub fn with_read_only(mut self, prefix: impl Into<String>) -> Self {
        self.read_only_prefixes.insert(prefix.into());
        self.read_only_loaded = true;
        self
    }

    /// Builder: mark the read-only policy as loaded with no read-only mounts
    /// (so every canonical absolute path is writable). Without this, or
    /// [`Self::with_read_only`], the policy is unknown and every path is
    /// treated as read-only.
    pub fn with_empty_read_only_policy(mut self) -> Self {
        self.read_only_loaded = true;
        self
    }

    fn node(&self, id: &str) -> Option<&Node> {
        self.nodes.get(id)
    }

    fn has_edge(&self, from: &str, edge: &str, to: &str) -> bool {
        self.edges
            .contains(&(from.to_string(), edge.to_string(), to.to_string()))
    }

    fn is_read_only(&self, path: &str) -> bool {
        // If the read-only policy has not been loaded, the state cannot prove
        // any path writable: treat everything as read-only (fail closed) so a
        // truncated or unloaded slice never authorises a write.
        if !self.read_only_loaded {
            return true;
        }
        let p = Path::new(path);
        // This interpreter judges already-canonical absolute paths only.
        // Anything it cannot safely judge — a relative path, or one with a
        // `..` component that could escape a prefix — is treated as read-only
        // (fail-closed, since this guards a writable-path precondition).
        // Resolving `~`, `.`/`..`, and symlinks is the caller's job at the
        // trust boundary before a path enters the state.
        if !is_canonical_absolute(p) {
            return true;
        }
        // A read-only set is only trustworthy if every configured prefix is
        // itself canonical-absolute. A malformed prefix (relative, `..`,
        // empty) could silently fail to match a path it should protect, so if
        // any prefix is untrustworthy treat the path as read-only (fail
        // closed) rather than risk a false "writable".
        if self
            .read_only_prefixes
            .iter()
            .any(|prefix| !is_canonical_absolute(Path::new(prefix)))
        {
            return true;
        }
        // Component-aware containment: `/mnt/ro/x` is under `/mnt/ro`, but
        // `/mnt/rofoo` is not (a raw string prefix would wrongly match it).
        self.read_only_prefixes
            .iter()
            .any(|prefix| p.starts_with(Path::new(prefix)))
    }
}

/// Binding environment: a schema parameter name to a concrete value — a node
/// id for the node/edge predicates, or a literal (e.g. a filesystem path) for
/// [`Predicate::NotReadOnly`]. The proposed action supplies these; the
/// interpreter verifies them, it never searches for a binding.
pub type Bindings = BTreeMap<String, String>;

/// Comparison operators for [`Predicate::FieldCmp`]. Deliberately only exact
/// equality: raw substring/prefix matching is a footgun in a safety DSL (it
/// accepts sibling paths), so path containment must go through the
/// component-aware [`Predicate::PathUnder`], never a string prefix here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    /// Equal.
    Eq,
    /// Not equal.
    Ne,
}

/// A predicate in the closed v1 set. The model may instantiate these with
/// parameters; a new predicate *kind* is a reviewed engine change.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Predicate {
    /// The entity bound to `bind` exists with node label `label`.
    NodeExists {
        /// Binding name for the entity.
        bind: String,
        /// The required node label.
        label: String,
    },
    /// The `field` of the entity bound to `bind` compares to `value` by `op`.
    FieldCmp {
        /// Binding name for the entity.
        bind: String,
        /// The field to read.
        field: String,
        /// The comparison operator.
        op: CmpOp,
        /// The literal value to compare against.
        value: String,
    },
    /// A `edge`-typed edge exists from the entity bound to `from` to the one
    /// bound to `to`.
    EdgeExists {
        /// Binding name for the source entity.
        from: String,
        /// The edge type.
        edge: String,
        /// Binding name for the target entity.
        to: String,
    },
    /// The path in the `field` of the entity bound to `bind` lies within the
    /// directory `prefix`, by component-boundary containment (so `/a/proj`
    /// does not contain `/a/proj2`). Use this for path containment, never a
    /// raw string comparison via [`Predicate::FieldCmp`], which would accept
    /// a sibling path.
    PathUnder {
        /// Binding name for the entity holding the path.
        bind: String,
        /// The field holding the path.
        field: String,
        /// The directory the path must lie within (a canonical absolute path).
        prefix: String,
    },
    /// The path in `inner_field` of the entity bound to `inner` lies within the
    /// directory path in `outer_field` of the entity bound to `outer`, by
    /// component-boundary containment. Unlike [`Predicate::PathUnder`] (whose
    /// directory is a fixed literal), this relates two nodes' path fields, so
    /// it can express "this file is under that project's root". Both paths must
    /// be canonical absolute, resolved at the ingestion boundary.
    PathUnderField {
        /// Binding for the entity holding the inner (contained) path.
        inner: String,
        /// The field on `inner` holding the inner path.
        inner_field: String,
        /// Binding for the entity holding the outer (containing) directory.
        outer: String,
        /// The field on `outer` holding the outer directory path.
        outer_field: String,
    },
    /// The capability layer permits the acting application to *execute* this
    /// action without a mandatory confirmation gate. The action's risk class
    /// is the trusted one in [`EvalContext::action_kind`] (classified from the
    /// proposed tool by the dispatcher), never a value the schema supplies, so
    /// a schema cannot downgrade a high-impact action to an ordinary one.
    CapabilityAllows,
    /// The filesystem path bound to `bind` is writable (not under a read-only
    /// mount).
    NotReadOnly {
        /// Binding name for the path.
        bind: String,
    },
    /// Negation: the inner predicate does not hold. Covers absence forms an
    /// action needs (an edge that must not yet exist, a field that must be
    /// empty), which the positive predicates alone cannot express.
    Not(Box<Predicate>),
}

/// An effect: an assertion or retraction over the same node/edge/field
/// vocabulary the predicates read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Effect {
    /// Create the entity bound to `bind` with label `label`.
    AssertNode {
        /// Binding name for the entity.
        bind: String,
        /// The node label.
        label: String,
    },
    /// Remove the entity bound to `bind` (and any edges touching it).
    RetractNode {
        /// Binding name for the entity.
        bind: String,
    },
    /// Create an edge from the entity bound to `from` to the one bound to `to`.
    AssertEdge {
        /// Binding name for the source entity.
        from: String,
        /// The edge type.
        edge: String,
        /// Binding name for the target entity.
        to: String,
    },
    /// Remove the edge from the entity bound to `from` to the one bound to `to`.
    RetractEdge {
        /// Binding name for the source entity.
        from: String,
        /// The edge type.
        edge: String,
        /// Binding name for the target entity.
        to: String,
    },
    /// Set a field on the entity bound to `bind`.
    SetField {
        /// Binding name for the entity.
        bind: String,
        /// The field to set.
        field: String,
        /// The value to set it to.
        value: String,
    },
}

impl Effect {
    /// The compensation (inverse) of this effect, expressed in the same DSL, or
    /// `None` when the effect cannot be inverted from itself alone.
    ///
    /// This grounds the gate's "reversible" predicate (Foundation B1): an
    /// effect with a derivable inverse can be undone; one without is
    /// irreversible and must be treated as high-impact (always confirm). The
    /// derivation is deliberately conservative, never guessing a state it does
    /// not hold:
    ///
    /// - `AssertEdge` ⇄ `RetractEdge` (and back): the forward effect's
    ///   precondition guarantees the edge was absent (resp. present) before, so
    ///   removing (resp. re-adding) exactly that edge restores the prior state.
    /// - `AssertNode` → `RetractNode`: the forward effect created a bare node
    ///   that did not exist before, so retracting it restores its absence. Sound
    ///   only inside a [`compensation_of`] sequence, which undoes any edges the
    ///   same action added to the node *first* (reverse order).
    /// - `RetractNode` → `None`: re-creating the node would need its prior
    ///   label, fields, and every edge that touched it, none of which the effect
    ///   carries. Irreversible from the effect alone.
    /// - `SetField` → `None`: restoring needs the field's prior value, which the
    ///   effect does not carry. Irreversible from the effect alone (a schema
    ///   that captures the prior value can declare a `SetField` compensation
    ///   explicitly; auto-derivation must not invent one).
    pub(crate) fn inverse(&self) -> Option<Effect> {
        match self {
            Effect::AssertEdge { from, edge, to } => Some(Effect::RetractEdge {
                from: from.clone(),
                edge: edge.clone(),
                to: to.clone(),
            }),
            Effect::RetractEdge { from, edge, to } => Some(Effect::AssertEdge {
                from: from.clone(),
                edge: edge.clone(),
                to: to.clone(),
            }),
            Effect::AssertNode { bind, .. } => Some(Effect::RetractNode { bind: bind.clone() }),
            Effect::RetractNode { .. } | Effect::SetField { .. } => None,
        }
    }
}

/// The compensation of an effect sequence: each effect's inverse, applied in
/// reverse order, or `None` if any effect is not invertible (the whole sequence
/// is then irreversible, fail-closed). Reverse order matters: an action that
/// asserts a node and then an edge to it compensates by retracting the edge
/// first, then the node, so no [`Effect::RetractNode`] strips an edge a later
/// compensation step still depends on.
pub(crate) fn compensation_of(effects: &[Effect]) -> Option<Vec<Effect>> {
    effects.iter().rev().map(Effect::inverse).collect()
}

/// Where an action schema came from. This increment exercises only `Given`
/// (rules from the schema + capability layer, ground truth day one); the
/// `Learned` induction pipeline is a later increment, but the field exists
/// so a learned rule is never indistinguishable from a given one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Provenance {
    /// A ground-truth rule (from the schema, the capability layer, or a tool
    /// definition).
    Given,
    /// A rule induced from observed reality, trusted only after approval.
    Learned {
        /// Who approved the learned rule.
        approved_by: String,
    },
}

/// A declarative action model: the action it describes, what must hold
/// before it, what it changes, and where the rule came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActionSchema {
    /// The tool/operation id, matching the proposed action's tool.
    pub action: String,
    /// Preconditions that must all hold for the action to be valid now.
    pub preconditions: Vec<Predicate>,
    /// Effects the action would make true (applied to a copy when predicting).
    pub effects: Vec<Effect>,
    /// Where this schema came from.
    pub provenance: Provenance,
}

/// The trusted context the interpreter needs beyond the world state: the
/// acting application, whether the trigger carried external content, and the
/// behaviour's mode ceiling. These are dispatcher-resolved, never taken from
/// the (untrusted) proposal.
pub struct EvalContext<'a> {
    /// The capability layer the `CapabilityAllows` predicate consults.
    pub capability: &'a Capability,
    /// The trusted id of the action actually being invoked (the proposed
    /// tool), resolved by the dispatcher. `predict` refuses a schema whose
    /// `action` does not match this, so a benign schema can never be paired
    /// with a different invocation to authorise the wrong action.
    pub action_id: &'a str,
    /// The acting application id.
    pub app_id: &'a str,
    /// The proposed action's risk class, classified from the trusted tool
    /// name by the dispatcher — never taken from the schema, so a schema
    /// cannot understate an action's impact.
    pub action_kind: ActionKind,
    /// Whether the triggering event carried external content.
    pub external_trigger: bool,
    /// The behaviour's requested mode ceiling.
    pub ceiling: BaselineMode,
}

/// The deterministic prediction for a grounded action. A post-state is
/// carried only by [`Prediction::Valid`], so a caller can never read a
/// simulated state that does not actually reflect the action.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Prediction {
    /// Every precondition held and every effect applied cleanly. Carries the
    /// simulated post-state.
    Valid {
        /// The state after applying the effects to a copy of the input state.
        predicted_state: WorldState,
    },
    /// One or more preconditions did not hold (the failing ones, for replan
    /// feedback).
    PreconditionsFailed {
        /// The preconditions that did not evaluate to a known `true`.
        failed: Vec<Predicate>,
    },
    /// Preconditions held, but an effect could not be applied (an unbound
    /// parameter, a dangling edge, a missing or label-incompatible node, a
    /// retraction of an absent target). The action must not be authorised: a
    /// clean prediction has to reflect the action's real effects.
    EffectError {
        /// Why the effect could not be applied.
        reason: String,
    },
    /// The schema does not describe the action being predicted: its `action`
    /// does not match the trusted invocation. Refused before anything is
    /// evaluated, so a benign schema cannot prove a different action safe.
    SchemaMismatch {
        /// The action the schema describes.
        schema_action: String,
        /// The action actually being invoked.
        invocation: String,
    },
    /// The schema's provenance is not trusted (an unapproved learned rule), so
    /// it may not prove anything. A given rule, or a learned rule with an
    /// approver, is trusted; an unapproved learned rule is not.
    UntrustedSchema {
        /// Why the schema is not trusted.
        reason: String,
    },
}

impl Prediction {
    /// Whether the action is valid (preconditions held and effects applied).
    pub fn is_valid(&self) -> bool {
        matches!(self, Prediction::Valid { .. })
    }

    /// The simulated post-state, if the prediction is [`Prediction::Valid`].
    pub fn predicted_state(&self) -> Option<&WorldState> {
        match self {
            Prediction::Valid { predicted_state } => Some(predicted_state),
            _ => None,
        }
    }
}

/// Predict a grounded action: check its preconditions against `state`, and —
/// only if they all hold — simulate its effects on a copy. Pure and
/// deterministic; it touches nothing live.
///
/// Caller contract — what this pure interpreter assumes and CANNOT itself
/// enforce, so the gate-integration and slice-ingestion boundaries (later
/// increments) must guarantee it:
/// 1. `schema` is selected from a TRUSTED source (the given-rule registry /
///    the tool's own definition) by the proposed tool id — not supplied by
///    the model. The interpreter checks the schema's declared provenance and
///    that its `action` matches the invocation, but it cannot verify the
///    schema actually came from the registry; that selection is the gate's.
/// 2. `bindings` are derived from the invocation's TYPED arguments (the same
///    operands the executor will use), so the prediction proves the real
///    invocation, not a different operand set.
/// 3. path fields/values are already CANONICAL (absolute, symlink-resolved)
///    — resolved at the ingestion boundary, since a pure function must not
///    touch the filesystem. The interpreter does lexical component-boundary
///    containment and rejects non-absolute / `..` paths, but cannot resolve
///    symlinks.
///
/// A `Valid` prediction proves the safety of *that* invocation only.
pub fn predict(
    schema: &ActionSchema,
    bindings: &Bindings,
    state: &WorldState,
    ctx: &EvalContext,
) -> Prediction {
    // Bind the prediction to the action actually being invoked: a schema may
    // only prove the safety of its own action, never stand in for another.
    if schema.action != ctx.action_id {
        return Prediction::SchemaMismatch {
            schema_action: schema.action.clone(),
            invocation: ctx.action_id.to_string(),
        };
    }
    // Only a trusted-provenance schema may prove anything: a given rule, or a
    // learned rule that carries an approver. An unapproved learned rule is
    // refused (the design requires learned rules to be approved before use).
    if let Provenance::Learned { approved_by } = &schema.provenance {
        if approved_by.trim().is_empty() {
            return Prediction::UntrustedSchema {
                reason: "learned rule without an approver".to_string(),
            };
        }
    }
    // A precondition holds only when it is *known* true. An unknown result
    // (a missing binding, a fact the state cannot answer) is not a pass: it
    // is absence of evidence, which must never authorise an action.
    let failed: Vec<Predicate> = schema
        .preconditions
        .iter()
        .filter(|p| eval(p, bindings, state, ctx) != Some(true))
        .cloned()
        .collect();
    if !failed.is_empty() {
        return Prediction::PreconditionsFailed { failed };
    }
    // Preconditions hold; the prediction is valid only if every effect also
    // applies cleanly, so a `Valid` post-state always reflects the action.
    match apply_effects(&schema.effects, bindings, state) {
        Ok(predicted_state) => Prediction::Valid { predicted_state },
        Err(reason) => Prediction::EffectError { reason },
    }
}

/// Evaluate one predicate to a three-valued result: `Some(true)` /
/// `Some(false)` when the answer is known, `None` when it cannot be
/// determined (a referenced binding is absent, or a field the comparison
/// needs is not present). `None` propagates through negation, so an absence
/// precondition like `Not(EdgeExists)` never passes on missing evidence.
fn eval(p: &Predicate, b: &Bindings, s: &WorldState, ctx: &EvalContext) -> Option<bool> {
    match p {
        // The bound entity exists with the given label. Binding absent ->
        // unknown; binding present -> known true/false against the state.
        Predicate::NodeExists { bind, label } => {
            let id = b.get(bind)?;
            Some(s.node(id).is_some_and(|n| &n.label == label))
        }
        // Comparing a field requires the bound node and the field to be
        // present; if either is missing the comparison is unknown, not false
        // (so its negation does not pass on missing evidence).
        Predicate::FieldCmp {
            bind,
            field,
            op,
            value,
        } => {
            let id = b.get(bind)?;
            let actual = s.node(id)?.fields.get(field)?;
            Some(cmp(*op, actual, value))
        }
        // Both endpoints must be bound, and both endpoint *nodes* must be
        // present in the state, before edge presence/absence is known. If an
        // endpoint node is missing, the state does not authoritatively cover
        // this relationship, so the answer is unknown — and stays unknown
        // through negation, so `Not(EdgeExists)` cannot pass on an incomplete
        // slice whose endpoints were never established.
        Predicate::EdgeExists { from, edge, to } => {
            let f = b.get(from)?;
            let t = b.get(to)?;
            if s.node(f).is_none() || s.node(t).is_none() {
                return None;
            }
            Some(s.has_edge(f, edge, t))
        }
        // Component-aware path containment. Unknown (so the precondition
        // fails) when the binding/field is absent, or when the path is not a
        // canonical absolute path this interpreter can safely judge.
        Predicate::PathUnder { bind, field, prefix } => {
            let id = b.get(bind)?;
            let path = s.node(id)?.fields.get(field)?;
            path_under(path, prefix)
        }
        // Component-aware containment between two nodes' path fields. Unknown
        // (so the precondition fails) when any binding/field is absent, or when
        // either path is not a canonical absolute path.
        Predicate::PathUnderField {
            inner,
            inner_field,
            outer,
            outer_field,
        } => {
            let inner_path = s.node(b.get(inner)?)?.fields.get(inner_field)?;
            let outer_path = s.node(b.get(outer)?)?.fields.get(outer_field)?;
            path_under(inner_path, outer_path)
        }
        // The capability is always available, so this is always known.
        // "Allows" means the agent is authorised to *execute* the kind
        // (Supervised preview or Autonomous), not merely to suggest it:
        // `Propose` (Suggest mode, the user executes) and
        // `RequireConfirmation` (high-impact / external) are not execution
        // authority, so they do not satisfy the precondition.
        Predicate::CapabilityAllows => Some(matches!(
            ctx.capability.decide_for_behaviour(
                ctx.app_id,
                ctx.action_kind,
                ctx.external_trigger,
                ctx.ceiling
            ),
            ActionDecision::PreviewThenExecute | ActionDecision::Proceed
        )),
        // Binding absent -> unknown; present -> known against the prefix set.
        Predicate::NotReadOnly { bind } => Some(!s.is_read_only(b.get(bind)?)),
        // Negation of an unknown is unknown.
        Predicate::Not(inner) => eval(inner, b, s, ctx).map(|v| !v),
    }
}

fn cmp(op: CmpOp, lhs: &str, rhs: &str) -> bool {
    match op {
        CmpOp::Eq => lhs == rhs,
        CmpOp::Ne => lhs != rhs,
    }
}

/// Whether a path is one this interpreter can safely judge: absolute and
/// free of `..` components. Resolving `~`, `.`/`..`, and symlinks is the
/// caller's job at the trust boundary before a path enters the state.
fn is_canonical_absolute(p: &Path) -> bool {
    p.is_absolute() && !p.components().any(|c| matches!(c, Component::ParentDir))
}

/// Component-aware path containment. `None` (so the precondition fails
/// closed) when *either* the path or the prefix cannot be safely judged: a
/// non-canonical-absolute path could escape the prefix, and an empty,
/// relative, or `..`-bearing prefix is not a trustworthy scope boundary.
fn path_under(path: &str, prefix: &str) -> Option<bool> {
    let p = Path::new(path);
    let root = Path::new(prefix);
    if !is_canonical_absolute(p) || !is_canonical_absolute(root) {
        return None;
    }
    Some(p.starts_with(root))
}

/// Look up the entity id a binding points at, or fail: an effect that names
/// a parameter the action never bound is malformed, not a silent no-op.
fn bind_id<'a>(b: &'a Bindings, name: &str) -> Result<&'a String, String> {
    b.get(name)
        .ok_or_else(|| format!("effect references unbound parameter {name:?}"))
}

/// Apply effects to a copy of `state`, failing closed on any effect that
/// cannot be applied so a successful prediction always reflects the real
/// mutation: an unbound parameter, a node asserted with a label that
/// conflicts with an existing one, an edge whose endpoints are not present
/// (a dangling edge), or a field set on a node that is not there.
fn apply_effects(effects: &[Effect], b: &Bindings, state: &WorldState) -> Result<WorldState, String> {
    let mut s = state.clone();
    for effect in effects {
        match effect {
            Effect::AssertNode { bind, label } => {
                let id = bind_id(b, bind)?;
                // Strict create: an assertion must *create*, never no-op over a
                // pre-existing node. A no-op would make the derived inverse
                // (`RetractNode`) delete state the action did not create, an
                // unsound rollback (Foundation B1). Failing here also means a
                // schema that omits the matching absence precondition produces an
                // `EffectError` when the target exists, so it is never lifted.
                if let Some(node) = s.nodes.get(id) {
                    return Err(format!(
                        "AssertNode {id:?}: a node already exists (label {:?}); an assertion must create, so its inverse retracts only what it created",
                        node.label
                    ));
                }
                s.nodes
                    .insert(id.clone(), Node::new(id.clone(), label.clone()));
            }
            Effect::RetractNode { bind } => {
                let id = bind_id(b, bind)?;
                // A retraction must target something that exists, so the
                // prediction proves a real mutation rather than a vacuous
                // removal against stale or incomplete state.
                if !s.nodes.contains_key(id) {
                    return Err(format!("RetractNode {id:?}: the node is not present"));
                }
                s.nodes.remove(id);
                s.edges.retain(|(f, _, t)| f != id && t != id);
            }
            Effect::AssertEdge { from, edge, to } => {
                let f = bind_id(b, from)?;
                let t = bind_id(b, to)?;
                if !s.nodes.contains_key(f) || !s.nodes.contains_key(t) {
                    return Err(format!(
                        "AssertEdge {f:?}-{edge:?}-{t:?}: an endpoint node is missing (dangling edge)"
                    ));
                }
                // Strict create (see AssertNode): a no-op over a pre-existing
                // edge would make the derived inverse (`RetractEdge`) delete an
                // edge the action did not create, an unsound rollback. Failing
                // here keeps the inverse sound and fails a precondition-less
                // schema closed at predict time.
                let key = (f.clone(), edge.clone(), t.clone());
                if s.edges.contains(&key) {
                    return Err(format!(
                        "AssertEdge {f:?}-{edge:?}-{t:?}: the edge already exists; an assertion must create, so its inverse retracts only what it created"
                    ));
                }
                s.edges.insert(key);
            }
            Effect::RetractEdge { from, edge, to } => {
                let f = bind_id(b, from)?;
                let t = bind_id(b, to)?;
                let key = (f.clone(), edge.clone(), t.clone());
                if !s.edges.contains(&key) {
                    return Err(format!(
                        "RetractEdge {f:?}-{edge:?}-{t:?}: the edge is not present"
                    ));
                }
                s.edges.remove(&key);
            }
            Effect::SetField { bind, field, value } => {
                let id = bind_id(b, bind)?;
                let node = s
                    .nodes
                    .get_mut(id)
                    .ok_or_else(|| format!("SetField {id:?}.{field:?}: the node is missing"))?;
                node.fields.insert(field.clone(), value.clone());
            }
        }
    }
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lunaris_ai_core::capability::{AccessTier, ActionPermissions, Capability};

    fn bindings(pairs: &[(&str, &str)]) -> Bindings {
        pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
    }

    fn ctx<'a>(cap: &'a Capability, action_id: &'a str) -> EvalContext<'a> {
        EvalContext {
            capability: cap,
            action_id,
            app_id: "org.lunaris.agent",
            action_kind: ActionKind::Ordinary,
            external_trigger: false,
            ceiling: BaselineMode::Supervised,
        }
    }

    /// The auto-tag scenario: a file under a project root, no edge yet. Uses
    /// `PathUnder` (component-aware) for the under-root check, never a raw
    /// string prefix.
    fn tag_schema(root: &str) -> ActionSchema {
        ActionSchema {
            action: "graph.write".to_string(),
            preconditions: vec![
                Predicate::NodeExists {
                    bind: "file".to_string(),
                    label: "File".to_string(),
                },
                Predicate::NodeExists {
                    bind: "proj".to_string(),
                    label: "Project".to_string(),
                },
                Predicate::PathUnder {
                    bind: "file".to_string(),
                    field: "path".to_string(),
                    prefix: root.to_string(),
                },
                Predicate::Not(Box::new(Predicate::EdgeExists {
                    from: "file".to_string(),
                    edge: "FILE_PART_OF".to_string(),
                    to: "proj".to_string(),
                })),
            ],
            effects: vec![Effect::AssertEdge {
                from: "file".to_string(),
                edge: "FILE_PART_OF".to_string(),
                to: "proj".to_string(),
            }],
            provenance: Provenance::Given,
        }
    }

    fn tag_state() -> WorldState {
        WorldState::new()
            .with_node(Node::new("f1", "File").with_field("path", "/home/tim/repos/proj/foo.rs"))
            .with_node(Node::new("p1", "Project").with_field("root_path", "/home/tim/repos/proj"))
    }

    #[test]
    fn preconditions_hold_and_effect_is_simulated() {
        let cap = Capability::new(AccessTier::Full, ActionPermissions::suggest_only());
        let schema = tag_schema("/home/tim/repos/proj");
        let state = tag_state();
        let p = predict(&schema, &bindings(&[("file", "f1"), ("proj", "p1")]), &state, &ctx(&cap, "graph.write"));
        assert!(p.is_valid(), "expected Valid, got {p:?}");
        // The edge did not exist before and is present in the predicted state.
        assert!(!state.has_edge("f1", "FILE_PART_OF", "p1"));
        assert!(p.predicted_state().unwrap().has_edge("f1", "FILE_PART_OF", "p1"));
    }

    #[test]
    fn an_already_present_edge_fails_the_negation() {
        let cap = Capability::new(AccessTier::Full, ActionPermissions::suggest_only());
        let schema = tag_schema("/home/tim/repos/proj");
        let state = tag_state().with_edge("f1", "FILE_PART_OF", "p1");
        let p = predict(&schema, &bindings(&[("file", "f1"), ("proj", "p1")]), &state, &ctx(&cap, "graph.write"));
        // No post-state is exposed for a failed prediction.
        assert!(p.predicted_state().is_none());
        assert!(matches!(
            p,
            Prediction::PreconditionsFailed { failed } if matches!(failed.as_slice(), [Predicate::Not(_)])
        ));
    }

    #[test]
    fn a_sibling_path_does_not_satisfy_path_under() {
        // The classic raw-prefix bypass: /home/tim/repos/proj2 is NOT under
        // /home/tim/repos/proj. Component-aware containment rejects it.
        let cap = Capability::new(AccessTier::Full, ActionPermissions::suggest_only());
        let schema = tag_schema("/home/tim/repos/proj");
        let state = WorldState::new()
            .with_node(Node::new("f1", "File").with_field("path", "/home/tim/repos/proj2/foo.rs"))
            .with_node(Node::new("p1", "Project"));
        let p = predict(&schema, &bindings(&[("file", "f1"), ("proj", "p1")]), &state, &ctx(&cap, "graph.write"));
        assert!(matches!(
            p,
            Prediction::PreconditionsFailed { failed } if matches!(failed.as_slice(), [Predicate::PathUnder { .. }])
        ));
    }

    #[test]
    fn path_under_field_relates_two_node_paths() {
        let cap = Capability::new(AccessTier::Full, ActionPermissions::suggest_only());
        let schema = ActionSchema {
            action: "graph.write".to_string(),
            preconditions: vec![Predicate::PathUnderField {
                inner: "file".to_string(),
                inner_field: "path".to_string(),
                outer: "proj".to_string(),
                outer_field: "root_path".to_string(),
            }],
            effects: vec![],
            provenance: Provenance::Given,
        };
        // The file lies under the project root: the relation holds.
        let under = WorldState::new()
            .with_node(Node::new("f1", "File").with_field("path", "/home/tim/repos/proj/foo.rs"))
            .with_node(Node::new("p1", "Project").with_field("root_path", "/home/tim/repos/proj"));
        assert!(predict(
            &schema,
            &bindings(&[("file", "f1"), ("proj", "p1")]),
            &under,
            &ctx(&cap, "graph.write")
        )
        .is_valid());
        // A sibling project root does not contain it (component-aware).
        let sibling = WorldState::new()
            .with_node(Node::new("f1", "File").with_field("path", "/home/tim/repos/proj2/foo.rs"))
            .with_node(Node::new("p1", "Project").with_field("root_path", "/home/tim/repos/proj"));
        assert!(matches!(
            predict(
                &schema,
                &bindings(&[("file", "f1"), ("proj", "p1")]),
                &sibling,
                &ctx(&cap, "graph.write")
            ),
            Prediction::PreconditionsFailed { .. }
        ));
    }

    #[test]
    fn a_missing_binding_fails_closed() {
        let cap = Capability::new(AccessTier::Full, ActionPermissions::suggest_only());
        let schema = tag_schema("/home/tim/repos/proj");
        // `proj` is not bound -> NodeExists{proj} is unknown -> not Valid.
        let p = predict(&schema, &bindings(&[("file", "f1")]), &tag_state(), &ctx(&cap, "graph.write"));
        assert!(!p.is_valid());
        assert!(matches!(p, Prediction::PreconditionsFailed { .. }));
    }

    #[test]
    fn capability_allows_uses_the_trusted_kind_and_requires_execution_authority() {
        let state = WorldState::new();
        let schema = ActionSchema {
            action: "x".to_string(),
            preconditions: vec![Predicate::CapabilityAllows],
            effects: vec![],
            provenance: Provenance::Given,
        };
        // The kind comes from the trusted context, not the schema, so a schema
        // cannot understate impact: this test drives it via `ctx.action_kind`.
        let holds = |cap: &Capability, kind, external| {
            let mut c = ctx(cap, "x");
            c.action_kind = kind;
            c.external_trigger = external;
            predict(&schema, &bindings(&[]), &state, &c).is_valid()
        };

        // Supervised mode yields PreviewThenExecute for an ordinary kind:
        // the agent is authorised to execute, so the precondition holds.
        let supervised = Capability::new(
            AccessTier::Full,
            ActionPermissions::new(BaselineMode::Supervised, Vec::<String>::new()),
        );
        assert!(holds(&supervised, ActionKind::Ordinary, false));

        // Suggest-only yields Propose: the *user* executes, not the agent, so
        // it is NOT execution authority and must not satisfy the predicate.
        let suggest = Capability::new(AccessTier::Full, ActionPermissions::suggest_only());
        assert!(!holds(&suggest, ActionKind::Ordinary, false));

        // A high-impact kind always requires confirmation -> not allowed,
        // even though the schema itself carries no kind to lie about.
        assert!(!holds(&supervised, ActionKind::PermanentDelete, false));
        // External content forces confirmation even for an ordinary kind.
        assert!(!holds(&supervised, ActionKind::Ordinary, true));
    }

    #[test]
    fn path_under_rejects_a_malformed_prefix() {
        let cap = Capability::new(AccessTier::Full, ActionPermissions::suggest_only());
        let state = WorldState::new()
            .with_node(Node::new("f1", "File").with_field("path", "/home/tim/x/foo"));
        let schema = |prefix: &str| ActionSchema {
            action: "x".to_string(),
            preconditions: vec![Predicate::PathUnder {
                bind: "file".to_string(),
                field: "path".to_string(),
                prefix: prefix.to_string(),
            }],
            effects: vec![],
            provenance: Provenance::Given,
        };
        let holds = |prefix: &str| {
            predict(&schema(prefix), &bindings(&[("file", "f1")]), &state, &ctx(&cap, "x")).is_valid()
        };
        assert!(holds("/home/tim/x")); // a valid, canonical prefix
        assert!(!holds("")); // empty prefix is not a trustworthy boundary
        assert!(!holds("home/tim")); // relative prefix
        assert!(!holds("/home/tim/x/..")); // `..`-bearing prefix
    }

    #[test]
    fn not_read_only_uses_component_aware_containment_and_fails_closed() {
        let cap = Capability::new(AccessTier::Full, ActionPermissions::suggest_only());
        let state = WorldState::new().with_read_only("/mnt/ro");
        let schema = ActionSchema {
            action: "fs.write".to_string(),
            preconditions: vec![Predicate::NotReadOnly { bind: "dest".to_string() }],
            effects: vec![],
            provenance: Provenance::Given,
        };
        let writable = |path: &str| {
            predict(&schema, &bindings(&[("dest", path)]), &state, &ctx(&cap, "fs.write")).is_valid()
        };

        assert!(writable("/home/tim/x")); // outside the read-only mount (policy loaded)
        assert!(writable("/mnt/rofoo/x")); // sibling, not under /mnt/ro (raw prefix would wrongly match)
        assert!(!writable("/mnt/ro/x")); // under the read-only mount
        assert!(!writable("/mnt/ro/../escape")); // `..` cannot be judged -> read-only
        assert!(!writable("rel/path")); // relative -> read-only
        assert!(!writable("~/x")); // unexpanded `~` is not absolute -> read-only
    }

    #[test]
    fn an_unloaded_read_only_policy_fails_closed() {
        let cap = Capability::new(AccessTier::Full, ActionPermissions::suggest_only());
        let schema = ActionSchema {
            action: "fs.write".to_string(),
            preconditions: vec![Predicate::NotReadOnly { bind: "dest".to_string() }],
            effects: vec![],
            provenance: Provenance::Given,
        };
        // No read-only policy loaded: a path cannot be proven writable.
        let unknown = WorldState::new();
        assert!(!predict(&schema, &bindings(&[("dest", "/home/tim/x")]), &unknown, &ctx(&cap, "fs.write")).is_valid());
        // Policy loaded with no read-only mounts: the same path is writable.
        let loaded = WorldState::new().with_empty_read_only_policy();
        assert!(predict(&schema, &bindings(&[("dest", "/home/tim/x")]), &loaded, &ctx(&cap, "fs.write")).is_valid());
    }

    #[test]
    fn effects_assert_retract_and_set() {
        let state = WorldState::new()
            .with_node(Node::new("a", "File"))
            .with_node(Node::new("b", "Project"))
            .with_edge("a", "OLD", "b");
        let b = bindings(&[("a", "a"), ("b", "b")]);
        let after = apply_effects(
            &[
                Effect::AssertEdge { from: "a".into(), edge: "NEW".into(), to: "b".into() },
                Effect::RetractEdge { from: "a".into(), edge: "OLD".into(), to: "b".into() },
                Effect::SetField { bind: "a".into(), field: "tagged".into(), value: "yes".into() },
            ],
            &b,
            &state,
        )
        .expect("effects apply");
        assert!(after.has_edge("a", "NEW", "b"));
        assert!(!after.has_edge("a", "OLD", "b"));
        assert_eq!(after.node("a").unwrap().fields.get("tagged").unwrap(), "yes");
    }

    #[test]
    fn asserting_an_existing_edge_is_an_effect_error_not_a_no_op() {
        // The edge already exists. A silent no-op would let the derived
        // compensation (RetractEdge) delete a pre-existing edge the action never
        // created, so the assertion must fail instead.
        let state = WorldState::new()
            .with_node(Node::new("a", "File"))
            .with_node(Node::new("b", "Project"))
            .with_edge("a", "FILE_PART_OF", "b");
        let err = apply_effects(
            &[Effect::AssertEdge { from: "a".into(), edge: "FILE_PART_OF".into(), to: "b".into() }],
            &bindings(&[("a", "a"), ("b", "b")]),
            &state,
        );
        assert!(err.is_err());
        // The pre-existing edge is untouched (no partial mutation).
        assert!(state.has_edge("a", "FILE_PART_OF", "b"));
    }

    #[test]
    fn asserting_an_existing_node_is_an_effect_error_even_with_the_same_label() {
        let state = WorldState::new().with_node(Node::new("x", "File"));
        let err = apply_effects(
            &[Effect::AssertNode { bind: "x".into(), label: "File".into() }],
            &bindings(&[("x", "x")]),
            &state,
        );
        assert!(err.is_err());
    }

    #[test]
    fn effect_inverse_per_type() {
        let assert_edge = Effect::AssertEdge { from: "a".into(), edge: "E".into(), to: "b".into() };
        let retract_edge = Effect::RetractEdge { from: "a".into(), edge: "E".into(), to: "b".into() };
        assert_eq!(assert_edge.inverse(), Some(retract_edge.clone()));
        assert_eq!(retract_edge.inverse(), Some(assert_edge));
        assert_eq!(
            Effect::AssertNode { bind: "a".into(), label: "File".into() }.inverse(),
            Some(Effect::RetractNode { bind: "a".into() })
        );
        // Not invertible from the effect alone (prior state unknown).
        assert_eq!(Effect::RetractNode { bind: "a".into() }.inverse(), None);
        assert_eq!(
            Effect::SetField { bind: "a".into(), field: "t".into(), value: "y".into() }.inverse(),
            None
        );
    }

    #[test]
    fn compensation_reverses_order_and_inverts_each() {
        // Assert a node then an edge to it: the compensation retracts the edge
        // first, then the node (reverse order), so the node retraction never
        // strips an edge a later step still needs.
        let forward = [
            Effect::AssertNode { bind: "f".into(), label: "File".into() },
            Effect::AssertEdge { from: "f".into(), edge: "FILE_PART_OF".into(), to: "p".into() },
        ];
        assert_eq!(
            compensation_of(&forward),
            Some(vec![
                Effect::RetractEdge { from: "f".into(), edge: "FILE_PART_OF".into(), to: "p".into() },
                Effect::RetractNode { bind: "f".into() },
            ])
        );
    }

    #[test]
    fn compensation_is_none_when_any_effect_is_irreversible() {
        let forward = [
            Effect::AssertEdge { from: "a".into(), edge: "E".into(), to: "b".into() },
            Effect::SetField { bind: "a".into(), field: "t".into(), value: "y".into() },
        ];
        assert_eq!(compensation_of(&forward), None);
    }

    #[test]
    fn forward_then_compensation_restores_state() {
        // Soundness: applying a forward effect sequence and then its
        // compensation returns the world to its starting point.
        let original = WorldState::new().with_node(Node::new("p1", "Project"));
        let b = bindings(&[("f", "f1"), ("p", "p1")]);
        let forward = [
            Effect::AssertNode { bind: "f".into(), label: "File".into() },
            Effect::AssertEdge { from: "f".into(), edge: "FILE_PART_OF".into(), to: "p".into() },
        ];
        let after = apply_effects(&forward, &b, &original).expect("forward applies");
        assert!(after.node("f1").is_some());
        assert!(after.has_edge("f1", "FILE_PART_OF", "p1"));

        let comp = compensation_of(&forward).expect("forward is reversible");
        let restored = apply_effects(&comp, &b, &after).expect("compensation applies");
        // Back to the start: the node and edge the action added are gone, the
        // pre-existing project node remains.
        assert!(restored.node("f1").is_none());
        assert!(!restored.has_edge("f1", "FILE_PART_OF", "p1"));
        assert!(restored.node("p1").is_some());
    }

    #[test]
    fn retract_node_removes_its_edges() {
        let state = WorldState::new()
            .with_node(Node::new("a", "File"))
            .with_node(Node::new("b", "Project"))
            .with_edge("a", "FILE_PART_OF", "b");
        let after = apply_effects(
            &[Effect::RetractNode { bind: "a".into() }],
            &bindings(&[("a", "a")]),
            &state,
        )
        .expect("effects apply");
        assert!(after.node("a").is_none());
        assert!(!after.has_edge("a", "FILE_PART_OF", "b"));
    }

    #[test]
    fn an_unapplicable_effect_yields_an_effect_error_not_a_silent_no_op() {
        let cap = Capability::new(AccessTier::Full, ActionPermissions::suggest_only());
        // Preconditions trivially hold (none); the effect names a target `proj`
        // the action never bound, so the prediction must fail, not pretend the
        // edge was created.
        let schema = ActionSchema {
            action: "graph.write".to_string(),
            preconditions: vec![],
            effects: vec![Effect::AssertEdge {
                from: "file".to_string(),
                edge: "FILE_PART_OF".to_string(),
                to: "proj".to_string(),
            }],
            provenance: Provenance::Given,
        };
        let state = WorldState::new().with_node(Node::new("f1", "File"));
        let p = predict(&schema, &bindings(&[("file", "f1")]), &state, &ctx(&cap, "graph.write"));
        assert!(matches!(p, Prediction::EffectError { .. }), "got {p:?}");
        assert!(p.predicted_state().is_none());
    }

    #[test]
    fn an_edge_to_a_missing_node_is_a_dangling_effect_error() {
        // Both ends bound, but `proj` has no node: a dangling edge must fail.
        let state = WorldState::new().with_node(Node::new("f1", "File"));
        let err = apply_effects(
            &[Effect::AssertEdge { from: "file".into(), edge: "FILE_PART_OF".into(), to: "proj".into() }],
            &bindings(&[("file", "f1"), ("proj", "p1")]),
            &state,
        );
        assert!(err.is_err());
    }

    #[test]
    fn asserting_a_node_with_a_conflicting_label_is_an_effect_error() {
        let state = WorldState::new().with_node(Node::new("x", "File"));
        let err = apply_effects(
            &[Effect::AssertNode { bind: "x".into(), label: "Project".into() }],
            &bindings(&[("x", "x")]),
            &state,
        );
        assert!(err.is_err());
    }

    #[test]
    fn a_schema_for_a_different_action_is_rejected() {
        let cap = Capability::new(AccessTier::Full, ActionPermissions::suggest_only());
        let schema = tag_schema("/home/tim/repos/proj"); // action "graph.write"
        // The invocation is a different tool; the benign tag schema must not
        // stand in for it.
        let p = predict(
            &schema,
            &bindings(&[("file", "f1"), ("proj", "p1")]),
            &tag_state(),
            &ctx(&cap, "fs.move"),
        );
        assert!(matches!(p, Prediction::SchemaMismatch { .. }), "got {p:?}");
        assert!(p.predicted_state().is_none());
    }

    #[test]
    fn an_unapproved_learned_schema_is_untrusted() {
        let cap = Capability::new(AccessTier::Full, ActionPermissions::suggest_only());
        // A learned rule with no approver may not prove anything, even with
        // trivially-satisfiable (empty) preconditions.
        let schema = ActionSchema {
            action: "x".to_string(),
            preconditions: vec![],
            effects: vec![],
            provenance: Provenance::Learned { approved_by: String::new() },
        };
        let p = predict(&schema, &bindings(&[]), &WorldState::new(), &ctx(&cap, "x"));
        assert!(matches!(p, Prediction::UntrustedSchema { .. }), "got {p:?}");

        // The same rule with an approver is trusted and evaluates normally.
        let approved = ActionSchema {
            provenance: Provenance::Learned { approved_by: "curator".to_string() },
            ..schema
        };
        assert!(predict(&approved, &bindings(&[]), &WorldState::new(), &ctx(&cap, "x")).is_valid());
    }

    #[test]
    fn a_malformed_read_only_prefix_fails_closed() {
        let cap = Capability::new(AccessTier::Full, ActionPermissions::suggest_only());
        // A relative read-only prefix is untrustworthy: every path is then
        // treated as read-only rather than risking a false "writable".
        let state = WorldState::new().with_read_only("relative/ro");
        let schema = ActionSchema {
            action: "fs.write".to_string(),
            preconditions: vec![Predicate::NotReadOnly { bind: "dest".to_string() }],
            effects: vec![],
            provenance: Provenance::Given,
        };
        let p = predict(&schema, &bindings(&[("dest", "/home/tim/x")]), &state, &ctx(&cap, "fs.write"));
        assert!(!p.is_valid());
    }

    #[test]
    fn negated_edge_does_not_pass_when_an_endpoint_node_is_missing() {
        // No NodeExists guard, so the safety rests entirely on EdgeExists: the
        // `to` endpoint node is absent, so edge absence is unknown, and
        // Not(EdgeExists) must NOT pass on that missing evidence.
        let cap = Capability::new(AccessTier::Full, ActionPermissions::suggest_only());
        let schema = ActionSchema {
            action: "graph.write".to_string(),
            preconditions: vec![Predicate::Not(Box::new(Predicate::EdgeExists {
                from: "a".to_string(),
                edge: "FILE_PART_OF".to_string(),
                to: "b".to_string(),
            }))],
            effects: vec![],
            provenance: Provenance::Given,
        };
        let state = WorldState::new().with_node(Node::new("a1", "File")); // no node for b
        let p = predict(
            &schema,
            &bindings(&[("a", "a1"), ("b", "b_ghost")]),
            &state,
            &ctx(&cap, "graph.write"),
        );
        assert!(!p.is_valid(), "absence over a missing endpoint must not pass");
    }

    #[test]
    fn retracting_an_absent_target_is_an_effect_error() {
        // RetractEdge on an edge that is not present.
        let state = WorldState::new()
            .with_node(Node::new("a", "File"))
            .with_node(Node::new("b", "Project"));
        let edge_err = apply_effects(
            &[Effect::RetractEdge { from: "a".into(), edge: "FILE_PART_OF".into(), to: "b".into() }],
            &bindings(&[("a", "a"), ("b", "b")]),
            &state,
        );
        assert!(edge_err.is_err());

        // RetractNode on a node that is not present.
        let node_err = apply_effects(
            &[Effect::RetractNode { bind: "missing".into() }],
            &bindings(&[("missing", "ghost")]),
            &WorldState::new(),
        );
        assert!(node_err.is_err());
    }
}
