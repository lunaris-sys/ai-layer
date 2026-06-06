//! Build a bounded [`WorldState`] slice from the Knowledge Graph for the
//! declarative world model.
//!
//! The pure [`crate::world::predict`] interpreter evaluates an action's
//! preconditions and effects against a [`WorldState`]. It cannot touch the
//! filesystem or the graph, so something must materialise that state from a
//! real, bounded slice of the Knowledge Graph and resolve path facts to a
//! canonical form first. That is this module's job, behind two seams:
//!
//! - [`PathResolver`] canonicalizes an absolute path (resolving its symlinks
//!   and `..` components) at the ingestion boundary, so the interpreter only
//!   ever does lexical component-boundary containment. A `~` or relative
//!   operand is normalized to absolute upstream (the graph stores absolute
//!   paths); the resolver rejects a non-absolute input rather than guess a
//!   base directory.
//! - [`MountPolicy`] supplies the complete set of read-only mount prefixes,
//!   so a writable-path precondition has a trustworthy basis.
//!
//! The slice is bounded by the schema, not by graph reachability: it loads
//! only the nodes named by the action's bindings, the fields its predicates
//! read, and the edges its predicates check. There is no traversal, so a
//! malformed schema or a hostile graph cannot make the slice large.
//!
//! Everything fails closed. [`build_slice`] returns an error (rather than a
//! partial state) whenever it cannot faithfully load what the predicates
//! reference: a graph read fails, a precondition names a node with no
//! declared label, an id-keyed lookup is ambiguous, an identifier is unsafe
//! to interpolate, or the read-only policy cannot be loaded while a
//! writable-path precondition needs it. A node that genuinely does not exist
//! is *not* an error: its absence is real knowledge the interpreter turns
//! into a failed precondition. The caller treats any error as "do not
//! predict, refuse the action".
//!
//! Wiring this into the dispatcher as a pre-gate hook (selecting the schema
//! from a trusted registry, deriving bindings from the invocation's typed
//! arguments, lifting the suggest-mode cap on a `Valid` prediction) is a
//! later increment behind the same pure interpreter. Until that wiring lands,
//! [`build_slice`] has no non-test caller, so the module allows dead code; the
//! allowance goes away once the gate path calls it.
#![allow(dead_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use lunaris_ai_core::graph_schema::{FieldType, GraphSchema};
use serde_json::Value;

use crate::registry::TrustedActionSchema;
use crate::seams::GraphHandle;
use crate::world::{ActionSchema, Bindings, Effect, Node, Predicate, Provenance, WorldState};

/// Resolves a raw path string to a canonical, absolute, symlink-free form.
///
/// This is the trust boundary the pure interpreter relies on: a path enters
/// the [`WorldState`] only after being resolved here, so the interpreter can
/// do lexical component-boundary containment and reject anything it cannot
/// safely judge.
pub trait PathResolver: Send + Sync {
    /// Resolve `raw` to a canonical-absolute path, or fail. Failure causes
    /// the path field to be omitted from the slice, so a precondition that
    /// reads it fails closed.
    fn resolve(&self, raw: &str) -> Result<String, SliceError>;
}

/// The production [`PathResolver`]: `std::fs::canonicalize`, which resolves
/// symlinks and `..` and requires the path to exist.
///
/// The input must already be absolute (a relative path is ambiguous and is
/// rejected). An existing path is canonicalized directly. A path that does
/// not exist yet (a write target for a new file) is resolved by
/// canonicalizing its parent directory, which must exist, and re-appending
/// the final name; since the parent is symlink-resolved and the final
/// component is a normal name, the result is still a canonical absolute path
/// inside the resolved parent, with no symlink left to escape a prefix. A
/// path whose parent also does not exist fails to resolve, which is the safe
/// outcome (the dependent precondition fails closed).
///
/// This performs a blocking filesystem call; the agent's slice rate is low,
/// so it is called directly.
///
/// The new-file result is a point-in-time prediction input, not a standalone
/// write authorization: the parent or final component could be replaced with
/// a symlink before the action runs. An executor that acts on a predicted
/// write must reopen the path atomically (an opened canonical parent plus
/// no-follow / create-new semantics) rather than trust this string. That
/// executor does not exist yet (suggest-mode), so it is a documented contract
/// (design gap A2), the same TOCTOU boundary [`build_slice`] notes.
#[derive(Debug, Default, Clone, Copy)]
pub struct FsPathResolver;

impl FsPathResolver {
    fn to_utf8(path: PathBuf, raw: &str) -> Result<String, SliceError> {
        path.to_str()
            .map(|s| s.to_string())
            .ok_or_else(|| SliceError::PathResolve {
                raw: raw.to_string(),
                reason: "canonical path is not valid UTF-8".to_string(),
            })
    }
}

impl PathResolver for FsPathResolver {
    fn resolve(&self, raw: &str) -> Result<String, SliceError> {
        let p = Path::new(raw);
        if !p.is_absolute() {
            return Err(SliceError::PathResolve {
                raw: raw.to_string(),
                reason: "path is not absolute".to_string(),
            });
        }
        // An existing path: canonicalize it directly (resolves symlinks).
        if let Ok(canon) = std::fs::canonicalize(p) {
            return Self::to_utf8(canon, raw);
        }
        // `canonicalize` failed. The new-file fallback is only safe when the
        // final component genuinely does not exist. A path that exists yet
        // does not canonicalize is a dangling symlink (or an unreadable
        // component): rejected, because the lexical fallback would otherwise
        // bless a symlink that the executor follows out of the proven scope.
        // `symlink_metadata` does not follow the final component, so it
        // detects the link itself; intermediate symlinks are still resolved by
        // the parent `canonicalize` below.
        match std::fs::symlink_metadata(p) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Ok(_) => {
                return Err(SliceError::PathResolve {
                    raw: raw.to_string(),
                    reason: "path exists but does not resolve (e.g. a dangling symlink)".to_string(),
                });
            }
            Err(e) => {
                return Err(SliceError::PathResolve {
                    raw: raw.to_string(),
                    reason: format!("path could not be examined: {e}"),
                });
            }
        }
        // A not-yet-existing target: resolve the parent (which must exist) and
        // re-append the final name. `file_name()` is `None` for a path ending
        // in `.`/`..` or root, so the appended component is always a normal
        // name and cannot reintroduce a `..` escape.
        let (Some(parent), Some(file)) = (p.parent(), p.file_name()) else {
            return Err(SliceError::PathResolve {
                raw: raw.to_string(),
                reason: "path has no resolvable parent and final name".to_string(),
            });
        };
        let canon_parent = std::fs::canonicalize(parent).map_err(|e| SliceError::PathResolve {
            raw: raw.to_string(),
            reason: format!("parent directory does not resolve: {e}"),
        })?;
        Self::to_utf8(canon_parent.join(file), raw)
    }
}

/// Supplies the complete set of canonical-absolute read-only mount prefixes.
///
/// The slice loads this only when a precondition checks a writable path
/// ([`Predicate::NotReadOnly`]); if it cannot be loaded, [`build_slice`]
/// fails closed rather than leave the policy unknown.
pub trait MountPolicy: Send + Sync {
    /// The read-only mount prefixes, or an error if they cannot be
    /// determined.
    fn read_only_prefixes(&self) -> Result<BTreeSet<String>, SliceError>;
}

/// The production [`MountPolicy`]: the read-only mounts in `/proc/mounts`.
#[derive(Debug, Default, Clone, Copy)]
pub struct ProcMountsPolicy;

impl MountPolicy for ProcMountsPolicy {
    fn read_only_prefixes(&self) -> Result<BTreeSet<String>, SliceError> {
        let text = std::fs::read_to_string("/proc/mounts")
            .map_err(|e| SliceError::MountPolicy(e.to_string()))?;
        Ok(parse_proc_mounts_ro(&text))
    }
}

/// A fixed [`MountPolicy`] from a known set of prefixes (for callers that
/// derive the policy elsewhere, and for tests).
#[derive(Debug, Default, Clone)]
pub struct StaticMountPolicy {
    prefixes: BTreeSet<String>,
}

impl StaticMountPolicy {
    /// A policy with no read-only mounts (every canonical-absolute path is
    /// writable). `const` so it can back a `static` test seam.
    pub const fn empty() -> Self {
        Self {
            prefixes: BTreeSet::new(),
        }
    }

    /// A policy from an explicit set of read-only prefixes.
    pub fn new(prefixes: impl IntoIterator<Item = String>) -> Self {
        Self {
            prefixes: prefixes.into_iter().collect(),
        }
    }
}

impl MountPolicy for StaticMountPolicy {
    fn read_only_prefixes(&self) -> Result<BTreeSet<String>, SliceError> {
        Ok(self.prefixes.clone())
    }
}

/// Why a slice could not be faithfully built. Every variant means "refuse
/// the action": a partial or unverifiable slice must never be predicted on.
#[derive(Debug, thiserror::Error)]
pub enum SliceError {
    /// A graph read failed, so a referenced node or edge could not be
    /// established. (A node that does not exist returns zero rows and is not
    /// an error.)
    #[error("graph read failed: {0}")]
    GraphRead(String),
    /// A precondition or effect uses `bind` as a node, but no predicate
    /// declares that binding's node label, so it cannot be loaded, validated,
    /// or collision-checked. A well-formed schema declares `NodeExists` for
    /// every node it reads or mutates.
    #[error("node binding {0:?} has no declared label, so it cannot be loaded")]
    NoLabel(String),
    /// A binding is declared with two different labels in the same schema.
    #[error("node binding {0:?} is declared with conflicting labels")]
    AmbiguousLabel(String),
    /// An id-keyed node lookup returned more than one row, so the slice
    /// cannot be trusted to be unambiguous.
    #[error("node id {0:?} matched more than one node")]
    AmbiguousNode(String),
    /// Two bindings resolve to the same id under different labels. The world
    /// state keys nodes by id alone, so it cannot represent both; the slice
    /// refuses rather than silently collapse them.
    #[error("node id {0:?} is used under more than one label in the same slice")]
    IdCollision(String),
    /// A binding is used both as a node (it has a declared label) and as a
    /// filesystem path (a `NotReadOnly` operand). A parameter cannot be both.
    #[error("binding {0:?} is used both as a node and as a path")]
    ContradictoryBinding(String),
    /// The invocation supplied an operand the schema does not reference. A
    /// prediction must prove exactly the operands the action will use, so an
    /// extra operand (which the schema never constrains) is refused rather
    /// than allowed to ride along on a proof that ignored it.
    #[error("operand {0:?} is not a parameter of the action's schema")]
    UnexpectedOperand(String),
    /// A graph result did not have the expected shape (e.g. an edge-count
    /// query returned other than a single numeric `cnt`), so it cannot be
    /// trusted to mean "absent".
    #[error("graph result was malformed: {0}")]
    MalformedResult(String),
    /// The schema uses a node-level mutation this increment cannot soundly
    /// predict on a bounded, id-keyed slice: `AssertNode` needs cross-label
    /// identity resolution, and `RetractNode`'s real blast radius is every
    /// incident edge the slice did not load. Both are refused rather than
    /// predicted with an understated effect; they land with the richer
    /// identity / blast-radius world-model follow-up.
    #[error("effect is not yet predictable: {0}")]
    UnsupportedEffect(String),
    /// A label, field, or edge type is not a safe Cypher identifier
    /// (`[A-Za-z_][A-Za-z0-9_]*`), so it must not be interpolated into a query
    /// or handed on to an executor.
    #[error("unsafe identifier {0:?} in schema")]
    BadIdentifier(String),
    /// The schema describes a different action than the one being invoked.
    /// Refused before any read, so a benign schema cannot drive graph reads
    /// for an action it does not describe.
    #[error("schema describes {schema_action:?}, not the invoked {invocation:?}")]
    SchemaMismatch {
        /// The action the schema describes.
        schema_action: String,
        /// The action actually being invoked.
        invocation: String,
    },
    /// The schema's provenance is not trusted (an unapproved learned rule), so
    /// it may not drive reads or prove anything. Refused before any read.
    #[error("schema is not trusted: {0}")]
    UntrustedSchema(String),
    /// The schema references the Knowledge Graph in a way the canonical schema
    /// does not allow (an unknown node label, a field that does not exist on a
    /// label, an unknown edge type, or an edge between the wrong endpoint
    /// labels), so a prediction over it would assert an impossible post-state.
    #[error("schema does not match the knowledge graph: {0}")]
    SchemaViolation(String),
    /// The schema is too large to bound the slice's reads: too many
    /// predicates, node bindings, or edges, or too deeply nested. Capped
    /// before the first read so a malformed rule cannot serialize an unbounded
    /// number of graph queries.
    #[error("schema is too large to slice: {0}")]
    SchemaTooLarge(String),
    /// A binding value cannot be safely escaped into a Cypher string literal
    /// (it contains a NUL or a control character).
    #[error("binding value {0:?} cannot be safely escaped")]
    BadBinding(String),
    /// A path could not be resolved to a canonical-absolute form.
    #[error("could not resolve path {raw:?}: {reason}")]
    PathResolve {
        /// The raw path that failed to resolve.
        raw: String,
        /// Why resolution failed.
        reason: String,
    },
    /// The read-only mount policy could not be loaded while a precondition
    /// needs it.
    #[error("could not load the read-only mount policy: {0}")]
    MountPolicy(String),
}

/// Largest total number of preconditions plus effects a schema may have.
const MAX_SLICE_PREDICATES: usize = 256;
/// Largest number of distinct node bindings the slice will load (one query
/// each).
const MAX_SLICE_NODES: usize = 64;
/// Largest number of edges the slice will check (one query each).
const MAX_SLICE_EDGES: usize = 64;
/// Largest nesting depth of a predicate (`Not` chains), bounding the walk's
/// recursion so a pathological schema cannot overflow the stack.
const MAX_PREDICATE_DEPTH: usize = 32;

/// What the schema's predicates and effects require the slice to load: the
/// label per node binding, the fields to read per binding, which of those
/// fields are paths, the edges to check, and whether a writable-path check
/// needs the read-only policy.
#[derive(Debug, Default)]
struct SliceSpec {
    /// Node binding to its declared label (from `NodeExists` / `AssertNode`).
    labels: BTreeMap<String, String>,
    /// Every binding used in a node position anywhere (precondition or
    /// effect). Each must have a declared label, so it can be loaded,
    /// KG-validated, and collision-checked; without one an effect operand
    /// could alias a loaded node by id and bypass validation. A path binding
    /// must never overlap this set either.
    node_binds: BTreeSet<String>,
    /// Fields to read per node binding (from `FieldCmp` / `PathUnder`).
    fields: BTreeMap<String, BTreeSet<String>>,
    /// Of `fields`, the ones that hold a path and must be canonicalised.
    path_fields: BTreeMap<String, BTreeSet<String>>,
    /// Edges to check for existence (from `EdgeExists` / `RetractEdge`).
    edges: BTreeSet<(String, String, String)>,
    /// Bindings that hold a filesystem path (`NotReadOnly`) and so must be
    /// canonicalized before they enter the evaluation.
    path_bindings: BTreeSet<String>,
    /// Whether any precondition checks a writable path (`NotReadOnly`).
    needs_read_only: bool,
    /// A binding found declared with conflicting labels, if any.
    label_conflict: Option<String>,
    /// Whether a predicate nested deeper than [`MAX_PREDICATE_DEPTH`] was seen
    /// (the walk stops descending at that point).
    too_deep: bool,
    /// A node-level mutation the slice cannot soundly predict yet, if any.
    /// `AssertNode` would need a cross-label id check the id-keyed state cannot
    /// do; `RetractNode`'s real blast radius is every edge incident to the
    /// node, which the bounded slice does not load (loading them all would be
    /// an unbounded traversal). Carries the effect name for the refusal.
    node_mutation: Option<&'static str>,
    /// Every label, field, and edge-type identifier the schema names, from
    /// preconditions and effects alike. Validated as safe identifiers before
    /// any read, so an effect-only identifier (a `SetField` field, an
    /// `AssertEdge` edge type) cannot pass unvalidated to a later executor.
    idents: BTreeSet<String>,
}

impl SliceSpec {
    /// Extract the load requirements from a schema's preconditions and
    /// effects. Fails if a binding is declared with conflicting labels, or a
    /// precondition reads a node binding that no predicate gives a label.
    fn extract(schema: &ActionSchema) -> Result<Self, SliceError> {
        // Bound the schema size before walking, so the read count the slice
        // derives from it is bounded too.
        let total = schema.preconditions.len() + schema.effects.len();
        if total > MAX_SLICE_PREDICATES {
            return Err(SliceError::SchemaTooLarge(format!(
                "{total} predicates, more than the {MAX_SLICE_PREDICATES} allowed"
            )));
        }
        let mut spec = SliceSpec::default();
        for p in &schema.preconditions {
            spec.walk_precondition(p, 0);
        }
        for e in &schema.effects {
            spec.walk_effect(e);
        }
        if spec.too_deep {
            return Err(SliceError::SchemaTooLarge(format!(
                "a predicate is nested deeper than the {MAX_PREDICATE_DEPTH} allowed"
            )));
        }
        if spec.node_binds.len() > MAX_SLICE_NODES {
            return Err(SliceError::SchemaTooLarge(format!(
                "{} node bindings, more than the {MAX_SLICE_NODES} allowed",
                spec.node_binds.len()
            )));
        }
        if spec.edges.len() > MAX_SLICE_EDGES {
            return Err(SliceError::SchemaTooLarge(format!(
                "{} edges, more than the {MAX_SLICE_EDGES} allowed",
                spec.edges.len()
            )));
        }
        // A node-level mutation cannot be soundly predicted on a bounded,
        // id-keyed slice: creation needs a cross-label id check, and deletion
        // cascades to every incident edge the slice did not load. Refuse
        // rather than predict an understated blast radius; these land with the
        // richer identity / blast-radius model (a world-model follow-up).
        if let Some(effect) = spec.node_mutation {
            return Err(SliceError::UnsupportedEffect(format!(
                "{effect}: a node-level mutation the bounded slice cannot represent"
            )));
        }
        // Validate every identifier the schema names, up front and before any
        // read, so an effect-only identifier (a field, an edge type) cannot
        // reach an executor unchecked.
        if let Some(ident) = spec.idents.iter().find(|i| !is_safe_ident(i)) {
            return Err(SliceError::BadIdentifier(ident.clone()));
        }
        if let Some(bind) = spec.label_conflict.take() {
            return Err(SliceError::AmbiguousLabel(bind));
        }
        // Every node a precondition reads must be loadable, so it must have a
        // declared label. (Effect-only bindings without a label are simply
        // not loaded; the effect then fails closed in `apply_effects`.)
        // A parameter is either an entity or a path, never both: a binding
        // used anywhere as a node (a precondition or effect operand) and as a
        // `NotReadOnly` path operand is a contradictory schema. Checked before
        // the label requirement so a path operand reports the more specific
        // error, and so a canonicalized path is never treated as a node id.
        if let Some(bind) = spec.path_bindings.iter().find(|b| spec.node_binds.contains(*b)) {
            return Err(SliceError::ContradictoryBinding(bind.clone()));
        }
        // Every node a precondition OR a supported effect touches must have a
        // declared label, so it can be loaded, KG-validated, and
        // collision-checked. Without one, an effect operand could alias a
        // loaded node by id and mutate it while bypassing validation.
        if let Some(bind) = spec.node_binds.iter().find(|b| !spec.labels.contains_key(*b)) {
            return Err(SliceError::NoLabel(bind.clone()));
        }
        Ok(spec)
    }

    fn set_label(&mut self, bind: &str, label: &str) {
        match self.labels.get(bind) {
            Some(existing) if existing != label => {
                self.label_conflict.get_or_insert_with(|| bind.to_string());
            }
            _ => {
                self.labels.insert(bind.to_string(), label.to_string());
            }
        }
    }

    fn add_field(&mut self, bind: &str, field: &str) {
        self.fields.entry(bind.to_string()).or_default().insert(field.to_string());
    }

    /// Mark a binding used as a node, by a precondition or an effect. Every
    /// such binding must have a declared label.
    fn mark_node(&mut self, bind: &str) {
        self.node_binds.insert(bind.to_string());
    }

    /// Record an identifier (a label, field, or edge type) the schema names,
    /// for the up-front safe-identifier check.
    fn add_ident(&mut self, ident: &str) {
        self.idents.insert(ident.to_string());
    }

    fn add_path_field(&mut self, bind: &str, field: &str) {
        self.add_field(bind, field);
        self.path_fields
            .entry(bind.to_string())
            .or_default()
            .insert(field.to_string());
    }

    fn walk_precondition(&mut self, p: &Predicate, depth: usize) {
        if depth > MAX_PREDICATE_DEPTH {
            self.too_deep = true;
            return;
        }
        match p {
            Predicate::NodeExists { bind, label } => {
                self.set_label(bind, label);
                self.add_ident(label);
                self.mark_node(bind);
            }
            Predicate::FieldCmp { bind, field, .. } => {
                self.add_field(bind, field);
                self.add_ident(field);
                self.mark_node(bind);
            }
            Predicate::EdgeExists { from, edge, to } => {
                self.add_ident(edge);
                self.mark_node(from);
                self.mark_node(to);
                self.edges.insert((from.clone(), edge.clone(), to.clone()));
            }
            Predicate::PathUnder { bind, field, .. } => {
                self.add_path_field(bind, field);
                self.add_ident(field);
                self.mark_node(bind);
            }
            Predicate::PathUnderField {
                inner,
                inner_field,
                outer,
                outer_field,
            } => {
                self.add_path_field(inner, inner_field);
                self.add_path_field(outer, outer_field);
                self.add_ident(inner_field);
                self.add_ident(outer_field);
                self.mark_node(inner);
                self.mark_node(outer);
            }
            // The capability layer is consulted directly, not loaded.
            Predicate::CapabilityAllows => {}
            // The bound value is a path literal, not a node; it must be
            // canonicalized at the ingestion boundary, and the read-only
            // policy is needed to judge it.
            Predicate::NotReadOnly { bind } => {
                self.needs_read_only = true;
                self.path_bindings.insert(bind.clone());
            }
            Predicate::Not(inner) => self.walk_precondition(inner, depth + 1),
        }
    }

    fn walk_effect(&mut self, e: &Effect) {
        match e {
            // Node-level mutations: not soundly predictable on a bounded,
            // id-keyed slice, so flag them for refusal in `extract`.
            Effect::AssertNode { .. } => {
                self.node_mutation.get_or_insert("AssertNode");
            }
            Effect::RetractNode { .. } => {
                self.node_mutation.get_or_insert("RetractNode");
            }
            // Needs the edge's prior presence to retract it; load that edge.
            Effect::RetractEdge { from, edge, to } => {
                self.add_ident(edge);
                self.mark_node(from);
                self.mark_node(to);
                self.edges.insert((from.clone(), edge.clone(), to.clone()));
            }
            // AssertEdge / SetField need their nodes present; those nodes are
            // loaded if a precondition declared their label, otherwise the
            // effect fails closed in `apply_effects`. Their operands are still
            // node uses, so a path binding must not overlap them, and their
            // identifiers (the edge type, the field) are validated up front.
            Effect::SetField { bind, field, .. } => {
                self.add_ident(field);
                self.mark_node(bind);
            }
            // Load the target edge's prior presence (as RetractEdge does), so a
            // strict AssertEdge sees a pre-existing edge and fails closed even
            // when the schema omits a Not(EdgeExists) precondition. Without this
            // the slice would have no edge row, the assertion would look like a
            // clean create, and the derived inverse could later delete an edge
            // the action did not create (an unsound rollback). graph.write
            // already loads it via its Not(EdgeExists) precondition, so this is
            // a no-op there and the safeguard for any future schema.
            Effect::AssertEdge { from, edge, to } => {
                self.add_ident(edge);
                self.mark_node(from);
                self.mark_node(to);
                self.edges.insert((from.clone(), edge.clone(), to.clone()));
            }
        }
    }
}

/// Build a bounded [`WorldState`] slice for `schema` grounded by `bindings`.
///
/// `action_id` is the trusted id of the action actually being invoked. The
/// schema's `action` must match it and its provenance must be trusted, both
/// checked before any read, so an unrelated or unapproved schema cannot drive
/// graph reads it would only be rejected for later.
///
/// This is the private inner builder, taking a raw [`ActionSchema`]. It is not
/// reachable outside this module: the only entry the gate path uses is
/// [`build_slice_trusted`], which requires a registry-resolved
/// [`TrustedActionSchema`], so a raw, possibly model-controlled schema can
/// never reach these graph reads. The action and provenance checks below
/// remain as a defensive floor.
///
/// Loads only what the schema's predicates and effects reference: the bound
/// nodes (by id, with their declared label), the fields the predicates read,
/// the edges they check, and the read-only mount policy if a writable-path
/// precondition needs it. Path fields are resolved to canonical-absolute
/// form through `paths` before they enter the state.
///
/// Fails closed: any inability to faithfully load a referenced fact is an
/// error, and the caller must refuse the action rather than predict on a
/// partial state. A node that genuinely does not exist (zero rows) is not an
/// error; its absence is real knowledge the interpreter turns into a failed
/// precondition.
///
/// Returns the world state *and* the bindings to predict with: a
/// `NotReadOnly` path operand is a filesystem fact too, so it is canonicalized
/// through the same boundary as a graph path field, and the caller must use
/// the returned bindings (a path that could not be resolved is dropped, so
/// the dependent precondition fails closed).
///
/// The world state keys nodes by id, while the graph keys them by
/// `(label, id)`. The slice therefore refuses an id reused across labels among
/// the declared bindings (`SliceError::IdCollision`) rather than collapse two
/// distinct graph nodes. It also refuses node-level mutations
/// (`SliceError::UnsupportedEffect`): it cannot tell whether an `AssertNode`
/// target's id is already used under a different label (a single-label query
/// never sees it), and a `RetractNode` would cascade to every incident edge
/// the bounded slice never loaded. Both land with the richer identity /
/// blast-radius model; edge and field effects are bounded and fail closed
/// under id keying.
///
/// The slice is a point-in-time view assembled from independent reads (the
/// graph query API exposes no snapshot or transaction), so a prediction over
/// it proves safety only at that instant. A concurrent graph write can make a
/// just-read absence stale before the action runs. Closing that window is the
/// executor's job, not this function's: an action must re-check its
/// preconditions atomically at write time and be idempotent (design gap A2,
/// optimistic concurrency plus a short prediction lifetime). That executor
/// does not exist yet (this is suggest-mode), so it is a documented boundary,
/// not something `build_slice` can enforce.
async fn build_slice(
    schema: &ActionSchema,
    action_id: &str,
    bindings: &Bindings,
    graph: &dyn GraphHandle,
    paths: &dyn PathResolver,
    mounts: &dyn MountPolicy,
) -> Result<(WorldState, Bindings), SliceError> {
    // Establish schema trust BEFORE any external read. A schema for a
    // different action, or an unapproved learned rule, must not drive graph
    // reads (leaking timing or errors, consuming capacity) even though
    // `predict` would later reject it; these mirror `predict`'s own checks.
    if schema.action != action_id {
        return Err(SliceError::SchemaMismatch {
            schema_action: schema.action.clone(),
            invocation: action_id.to_string(),
        });
    }
    if let Provenance::Learned { approved_by } = &schema.provenance {
        if approved_by.trim().is_empty() {
            return Err(SliceError::UntrustedSchema(
                "learned rule without an approver".to_string(),
            ));
        }
    }
    let spec = SliceSpec::extract(schema)?;
    // Validate the schema against the canonical KG schema before any read, so
    // an effect that asserts an impossible relationship or field is refused
    // rather than predicted Valid on a post-state the graph could never hold.
    validate_against_kg(schema, &spec.labels)?;
    // Every operand must be a parameter the schema references. The schema
    // constrains only the operands it names, so an extra one would ride along
    // on a prediction that ignored it; refuse it rather than prove a partial
    // operand set for an action that will execute with more.
    if let Some(extra) = bindings
        .keys()
        .find(|k| !spec.node_binds.contains(*k) && !spec.path_bindings.contains(*k))
    {
        return Err(SliceError::UnexpectedOperand(extra.clone()));
    }
    let mut state = WorldState::new();
    let mut loaded: BTreeSet<String> = BTreeSet::new();

    // Reject an id reused across labels up front, over the declared bindings
    // (before any graph read). The id-keyed world state cannot hold two labels
    // for one id, and an effect on an absent-but-declared binding could
    // otherwise alias onto a different loaded node that shares that id.
    let mut declared: BTreeMap<&str, &str> = BTreeMap::new();
    for (bind, label) in &spec.labels {
        let Some(id) = bindings.get(bind) else {
            continue;
        };
        match declared.get(id.as_str()) {
            Some(existing) if *existing != label.as_str() => {
                return Err(SliceError::IdCollision(id.clone()));
            }
            _ => {
                declared.insert(id.as_str(), label.as_str());
            }
        }
    }

    // Load every node binding that has a declared label and a bound id.
    for (bind, label) in &spec.labels {
        let Some(id) = bindings.get(bind) else {
            // Unbound: cannot query. The interpreter fails closed on the
            // absent binding, so nothing is loaded for it.
            continue;
        };
        if !is_safe_ident(label) {
            return Err(SliceError::BadIdentifier(label.clone()));
        }
        let read_fields = spec.fields.get(bind);
        let path_fields = spec.path_fields.get(bind);

        // RETURN the id (presence marker) plus every field a predicate reads.
        let mut columns: BTreeSet<&str> = BTreeSet::new();
        columns.insert("id");
        if let Some(fs) = read_fields {
            for f in fs {
                if !is_safe_ident(f) {
                    return Err(SliceError::BadIdentifier(f.clone()));
                }
                columns.insert(f.as_str());
            }
        }
        let id_lit = escape_cypher_literal(id)?;
        let return_clause = columns
            .iter()
            .map(|c| format!("n.{c} AS {c}"))
            .collect::<Vec<_>>()
            .join(", ");
        // `LIMIT 2` bounds the read: at most two rows come back, enough to
        // detect a broken uniqueness invariant (`AmbiguousNode`) without
        // materializing an unbounded result.
        let cypher = format!("MATCH (n:{label} {{id: '{id_lit}'}}) RETURN {return_clause} LIMIT 2");

        let rows = graph
            .query(&cypher)
            .await
            .map_err(|e| SliceError::GraphRead(e.to_string()))?;
        match rows.len() {
            0 => continue, // node genuinely absent: real knowledge, not loaded
            1 => {}
            _ => return Err(SliceError::AmbiguousNode(id.clone())),
        }
        let row = &rows[0];

        // The single row must actually be the requested node: a faithful
        // response returns its id. A degraded response (an empty or
        // mismatched row) must not be accepted as proof of existence, or
        // `NodeExists` and dependent effects would pass without the graph
        // having established the node.
        match row.get("id").and_then(cell_to_string) {
            Some(returned) if returned == *id => {}
            _ => {
                return Err(SliceError::MalformedResult(format!(
                    "node query for {id:?} did not return a matching id"
                )))
            }
        }

        let mut node = Node::new(id.clone(), label.clone());
        if let Some(fs) = read_fields {
            for field in fs {
                let Some(cell) = row.get(field) else { continue };
                let Some(value) = cell_to_string(cell) else { continue };
                let value = if path_fields.is_some_and(|pf| pf.contains(field)) {
                    // A path that cannot be canonicalised is omitted, so the
                    // dependent precondition fails closed rather than judging
                    // an unresolved (and possibly escaping) path.
                    match paths.resolve(&value) {
                        Ok(canonical) => canonical,
                        Err(_) => continue,
                    }
                } else {
                    value
                };
                node = node.with_field(field, value);
            }
        }
        state = state.with_node(node);
        loaded.insert(id.clone());
    }

    // Load each referenced edge whose endpoints were both loaded. If an
    // endpoint is absent, the edge cannot exist; the interpreter already
    // reports edge presence over a missing endpoint as unknown (fail closed),
    // so we simply do not insert it.
    for (from, edge, to) in &spec.edges {
        let (Some(from_id), Some(to_id)) = (bindings.get(from), bindings.get(to)) else {
            continue;
        };
        if !loaded.contains(from_id) || !loaded.contains(to_id) {
            continue;
        }
        let (Some(from_label), Some(to_label)) = (spec.labels.get(from), spec.labels.get(to))
        else {
            continue;
        };
        for ident in [from_label.as_str(), to_label.as_str(), edge.as_str()] {
            if !is_safe_ident(ident) {
                return Err(SliceError::BadIdentifier(ident.to_string()));
            }
        }
        let from_lit = escape_cypher_literal(from_id)?;
        let to_lit = escape_cypher_literal(to_id)?;
        let cypher = format!(
            "MATCH (a:{from_label} {{id: '{from_lit}'}})-[:{edge}]->(b:{to_label} {{id: '{to_lit}'}}) RETURN count(*) AS cnt"
        );
        let rows = graph
            .query(&cypher)
            .await
            .map_err(|e| SliceError::GraphRead(e.to_string()))?;
        // A count query must return exactly one numeric `cnt`. Treating a
        // missing row or a non-numeric cell as zero would turn a degraded or
        // version-skewed response into a false "edge absent", which a
        // `Not(EdgeExists)` precondition would read as a safe-to-proceed proof.
        let count = match rows.as_slice() {
            [single] => single
                .get("cnt")
                .and_then(cell_to_i64)
                .ok_or_else(|| SliceError::MalformedResult("edge count missing or non-numeric".to_string()))?,
            _ => {
                return Err(SliceError::MalformedResult(
                    "edge count did not return exactly one row".to_string(),
                ))
            }
        };
        if count > 0 {
            state = state.with_edge(from_id.clone(), edge.clone(), to_id.clone());
        }
    }

    // Load the read-only policy only if a precondition judges a writable
    // path. Failure to load it is fatal: without a known policy the
    // interpreter treats every path as read-only, but failing loudly here
    // gives a clearer reason than a precondition that silently never holds.
    if spec.needs_read_only {
        let prefixes = mounts.read_only_prefixes()?;
        if prefixes.is_empty() {
            state = state.with_empty_read_only_policy();
        } else {
            for prefix in prefixes {
                state = state.with_read_only(prefix);
            }
        }
    }

    // Canonicalize the path operands the same way as graph path fields: a
    // `NotReadOnly` argument is a filesystem fact, so a symlink in it must be
    // resolved before the lexical containment check, or it could be judged
    // writable while resolving inside a read-only mount. A path that cannot be
    // resolved is dropped from the bindings, so its `NotReadOnly` check fails
    // closed (the operand is then unknown to the interpreter).
    let mut out_bindings = bindings.clone();
    for bind in &spec.path_bindings {
        let Some(raw) = out_bindings.get(bind) else {
            continue;
        };
        match paths.resolve(raw) {
            Ok(canonical) => {
                out_bindings.insert(bind.clone(), canonical);
            }
            Err(_) => {
                out_bindings.remove(bind);
            }
        }
    }

    Ok((state, out_bindings))
}

/// Build a slice for a registry-resolved trusted schema. This is the only
/// way the gate path reaches the slice builder: it requires a
/// [`TrustedActionSchema`] (which the registry alone constructs), so a raw,
/// possibly model-controlled schema can never drive these graph reads.
pub(crate) async fn build_slice_trusted(
    trusted: &TrustedActionSchema,
    action_id: &str,
    bindings: &Bindings,
    graph: &dyn GraphHandle,
    paths: &dyn PathResolver,
    mounts: &dyn MountPolicy,
) -> Result<(WorldState, Bindings), SliceError> {
    build_slice(trusted.schema(), action_id, bindings, graph, paths, mounts).await
}

/// Convert a graph cell to the string form the world state stores. A null or
/// non-scalar cell has no string value (the field is omitted, so a
/// comparison over it fails closed).
fn cell_to_string(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        Value::Bool(b) => Some(b.to_string()),
        Value::Null | Value::Array(_) | Value::Object(_) => None,
    }
}

/// Read a graph cell as an integer (for `count(*)`), or `None` if it is not a
/// number.
fn cell_to_i64(v: &Value) -> Option<i64> {
    match v {
        Value::Number(n) => n.as_i64(),
        _ => None,
    }
}

/// Whether a string is a safe Cypher identifier to interpolate as a label,
/// field, or edge type: `[A-Za-z_][A-Za-z0-9_]*`.
fn is_safe_ident(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// The largest binding id the slice will interpolate into a query. Generous
/// for any real id or path, but bounds the query size a hostile or malformed
/// binding can produce.
const MAX_BINDING_LEN: usize = 8192;

/// Escape a value for a single-quoted Cypher string literal. Rejects an
/// over-long value (a query-size bound), NUL and control characters (abnormal
/// in an id and a footgun in a literal); escapes the backslash and the single
/// quote.
pub(crate) fn escape_cypher_literal(s: &str) -> Result<String, SliceError> {
    if s.len() > MAX_BINDING_LEN {
        return Err(SliceError::BadBinding(s.to_string()));
    }
    if s.chars().any(|c| c.is_control()) {
        return Err(SliceError::BadBinding(s.to_string()));
    }
    Ok(s.replace('\\', "\\\\").replace('\'', "\\'"))
}

/// Decode the octal escapes `/proc/mounts` uses for space, tab, newline, and
/// backslash in a field. Backslash is decoded last so its replacement cannot
/// reintroduce an escape sequence.
fn unescape_mount_field(s: &str) -> String {
    s.replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
}

/// Parse the read-only mount points from `/proc/mounts` content: the mount
/// point (field 2) of every line whose options (field 4) contain the `ro`
/// token.
fn parse_proc_mounts_ro(text: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for line in text.lines() {
        let mut fields = line.split_whitespace();
        let _device = fields.next();
        let Some(mount_point) = fields.next() else {
            continue;
        };
        let _fstype = fields.next();
        let Some(options) = fields.next() else {
            continue;
        };
        if options.split(',').any(|opt| opt == "ro") {
            let mount_point = unescape_mount_field(mount_point);
            if mount_point.starts_with('/') {
                out.insert(mount_point);
            }
        }
    }
    out
}

/// Validate a schema's Knowledge-Graph references against the canonical
/// graph schema, before any read: every declared label must be a real node
/// table, every field a real column on its label, and every edge a real
/// relationship between the right endpoint labels. This stops a prediction
/// from asserting an impossible post-state (e.g. `FILE_PART_OF` between
/// `File` and `App`, or setting a column that does not exist) on which a
/// later gate could be lifted. `labels` maps each binding to its declared
/// node label (from [`SliceSpec`]).
fn validate_against_kg(
    schema: &ActionSchema,
    labels: &BTreeMap<String, String>,
) -> Result<(), SliceError> {
    let kg = GraphSchema::knowledge_graph();
    for label in labels.values() {
        if kg.node(label).is_none() {
            return Err(SliceError::SchemaViolation(format!(
                "unknown node label {label:?}"
            )));
        }
    }
    for p in &schema.preconditions {
        validate_predicate_against_kg(p, labels, &kg)?;
    }
    for e in &schema.effects {
        validate_effect_against_kg(e, labels, &kg)?;
    }
    Ok(())
}

/// A field referenced on a binding must exist on that binding's declared
/// label. A binding with no declared label (an effect-only operand) is
/// skipped here; it fails closed at load or in `apply_effects`.
fn check_field(
    labels: &BTreeMap<String, String>,
    kg: &GraphSchema,
    bind: &str,
    field: &str,
) -> Result<(), SliceError> {
    if let Some(label) = labels.get(bind) {
        if kg.field_type(label, field).is_none() {
            return Err(SliceError::SchemaViolation(format!(
                "field {field:?} does not exist on node label {label:?}"
            )));
        }
    }
    Ok(())
}

/// An edge type must be a real relationship, and when both endpoint labels
/// are known they must match the canonical endpoints for that relationship.
fn check_edge(
    labels: &BTreeMap<String, String>,
    kg: &GraphSchema,
    from: &str,
    edge: &str,
    to: &str,
) -> Result<(), SliceError> {
    let Some(es) = kg.edge(edge) else {
        return Err(SliceError::SchemaViolation(format!("unknown edge type {edge:?}")));
    };
    if let (Some(from_label), Some(to_label)) = (labels.get(from), labels.get(to)) {
        if es.from != from_label || es.to != to_label {
            return Err(SliceError::SchemaViolation(format!(
                "edge {edge:?} connects {}->{}, not {from_label:?}->{to_label:?}",
                es.from, es.to
            )));
        }
    }
    Ok(())
}

fn validate_predicate_against_kg(
    p: &Predicate,
    labels: &BTreeMap<String, String>,
    kg: &GraphSchema,
) -> Result<(), SliceError> {
    match p {
        Predicate::FieldCmp { bind, field, .. } | Predicate::PathUnder { bind, field, .. } => {
            check_field(labels, kg, bind, field)
        }
        Predicate::PathUnderField {
            inner,
            inner_field,
            outer,
            outer_field,
        } => {
            check_field(labels, kg, inner, inner_field)?;
            check_field(labels, kg, outer, outer_field)
        }
        Predicate::EdgeExists { from, edge, to } => check_edge(labels, kg, from, edge, to),
        Predicate::Not(inner) => validate_predicate_against_kg(inner, labels, kg),
        Predicate::NodeExists { .. } | Predicate::CapabilityAllows | Predicate::NotReadOnly { .. } => {
            Ok(())
        }
    }
}

fn validate_effect_against_kg(
    e: &Effect,
    labels: &BTreeMap<String, String>,
    kg: &GraphSchema,
) -> Result<(), SliceError> {
    match e {
        Effect::SetField { bind, field, value } => {
            check_field(labels, kg, bind, field)?;
            // The written value must be representable as the field's type, or
            // the prediction would assert a post-state the graph cannot hold
            // (and the executor would reject or coerce it).
            if let Some(label) = labels.get(bind) {
                if let Some(ty) = kg.field_type(label, field) {
                    let representable = match ty {
                        FieldType::Text => true,
                        FieldType::Int => value.parse::<i64>().is_ok(),
                        FieldType::Bool => value == "true" || value == "false",
                    };
                    if !representable {
                        return Err(SliceError::SchemaViolation(format!(
                            "value {value:?} is not a valid {ty:?} for field {field:?} on {label:?}"
                        )));
                    }
                }
            }
            Ok(())
        }
        Effect::AssertEdge { from, edge, to } | Effect::RetractEdge { from, edge, to } => {
            check_edge(labels, kg, from, edge, to)
        }
        // Node-level mutations are already refused before this runs.
        Effect::AssertNode { .. } | Effect::RetractNode { .. } => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seams::{GraphError, GraphHandle};
    use crate::world::{predict, CmpOp, EvalContext, Prediction, Provenance};
    use lunaris_ai_core::capability::{
        AccessTier, ActionKind, ActionPermissions, BaselineMode, Capability,
    };
    use std::collections::HashMap;

    // ---- test doubles ------------------------------------------------------

    /// A graph that returns canned rows when the query contains a needle, and
    /// optionally errors when it contains another.
    struct MockGraph {
        rules: Vec<(String, Vec<HashMap<String, Value>>)>,
        err_on: Option<String>,
    }

    impl MockGraph {
        fn new() -> Self {
            Self {
                rules: Vec::new(),
                err_on: None,
            }
        }
        fn on(mut self, needle: &str, rows: Vec<HashMap<String, Value>>) -> Self {
            self.rules.push((needle.to_string(), rows));
            self
        }
        fn err_on(mut self, needle: &str) -> Self {
            self.err_on = Some(needle.to_string());
            self
        }
    }

    #[async_trait::async_trait]
    impl GraphHandle for MockGraph {
        async fn query(
            &self,
            cypher: &str,
        ) -> Result<Vec<HashMap<String, Value>>, GraphError> {
            if let Some(e) = &self.err_on {
                if cypher.contains(e.as_str()) {
                    return Err(GraphError::Failed("boom".to_string()));
                }
            }
            for (needle, rows) in &self.rules {
                if cypher.contains(needle.as_str()) {
                    return Ok(rows.clone());
                }
            }
            Ok(Vec::new())
        }
    }

    struct MockResolver {
        map: BTreeMap<String, String>,
    }
    impl MockResolver {
        fn new(pairs: &[(&str, &str)]) -> Self {
            Self {
                map: pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(),
            }
        }
    }
    impl PathResolver for MockResolver {
        fn resolve(&self, raw: &str) -> Result<String, SliceError> {
            self.map.get(raw).cloned().ok_or_else(|| SliceError::PathResolve {
                raw: raw.to_string(),
                reason: "no mapping".to_string(),
            })
        }
    }

    fn row(pairs: &[(&str, Value)]) -> HashMap<String, Value> {
        pairs.iter().map(|(k, v)| (k.to_string(), v.clone())).collect()
    }

    fn bindings(pairs: &[(&str, &str)]) -> Bindings {
        pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
    }

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

    fn capability() -> Capability {
        Capability::new(AccessTier::Full, ActionPermissions::suggest_only())
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

    // ---- pure helpers ------------------------------------------------------

    #[test]
    fn parses_read_only_mounts_from_proc_mounts() {
        let text = "\
sysfs /sys sysfs rw,nosuid 0 0
/dev/sda1 / ext4 rw,relatime 0 0
/dev/sdb1 /mnt/ro ext4 ro,relatime 0 0
overlay /usr/lib/modules overlay ro,lowerdir=x 0 0
tmpfs /tmp\\040dir tmpfs rw 0 0
";
        let ro = parse_proc_mounts_ro(text);
        assert!(ro.contains("/mnt/ro"));
        assert!(ro.contains("/usr/lib/modules"));
        assert!(!ro.contains("/"));
        assert!(!ro.iter().any(|m| m.contains("tmp")));
    }

    #[test]
    fn ro_token_is_matched_whole_not_as_substring() {
        // "rootcontext" or a value containing "ro" must not count as ro.
        let text = "/dev/x /data ext4 rw,errors=remount-ro 0 0\n";
        assert!(parse_proc_mounts_ro(text).is_empty());
    }

    #[test]
    fn unescapes_octal_in_mount_fields() {
        assert_eq!(unescape_mount_field("/mnt/a\\040b"), "/mnt/a b");
        assert_eq!(unescape_mount_field("/mnt/a\\134b"), "/mnt/a\\b");
    }

    #[test]
    fn safe_identifiers_only() {
        assert!(is_safe_ident("File"));
        assert!(is_safe_ident("root_path"));
        assert!(is_safe_ident("_x9"));
        assert!(!is_safe_ident(""));
        assert!(!is_safe_ident("9bad"));
        assert!(!is_safe_ident("a-b"));
        assert!(!is_safe_ident("a') DETACH DELETE (n"));
    }

    #[test]
    fn escapes_quote_and_backslash_rejects_control() {
        assert_eq!(escape_cypher_literal("a'b").unwrap(), "a\\'b");
        assert_eq!(escape_cypher_literal("a\\b").unwrap(), "a\\\\b");
        assert!(escape_cypher_literal("a\nb").is_err());
        assert!(escape_cypher_literal("a\0b").is_err());
    }

    #[test]
    fn cell_conversions() {
        assert_eq!(cell_to_string(&Value::String("x".into())), Some("x".to_string()));
        assert_eq!(cell_to_string(&serde_json::json!(90)), Some("90".to_string()));
        assert_eq!(cell_to_string(&Value::Bool(true)), Some("true".to_string()));
        assert_eq!(cell_to_string(&Value::Null), None);
        assert_eq!(cell_to_i64(&serde_json::json!(3)), Some(3));
        assert_eq!(cell_to_i64(&Value::String("3".into())), None);
    }

    // ---- dependency extraction --------------------------------------------

    #[test]
    fn extract_collects_labels_fields_paths_and_edges() {
        let spec = SliceSpec::extract(&tag_schema("/r")).unwrap();
        assert_eq!(spec.labels.get("file"), Some(&"File".to_string()));
        assert_eq!(spec.labels.get("proj"), Some(&"Project".to_string()));
        assert!(spec.path_fields.get("file").unwrap().contains("path"));
        assert!(spec.edges.contains(&(
            "file".to_string(),
            "FILE_PART_OF".to_string(),
            "proj".to_string()
        )));
        assert!(!spec.needs_read_only);
    }

    #[test]
    fn extract_rejects_a_precondition_node_without_a_label() {
        let schema = ActionSchema {
            action: "a".to_string(),
            preconditions: vec![Predicate::FieldCmp {
                bind: "x".to_string(),
                field: "f".to_string(),
                op: CmpOp::Eq,
                value: "v".to_string(),
            }],
            effects: vec![],
            provenance: Provenance::Given,
        };
        assert!(matches!(SliceSpec::extract(&schema), Err(SliceError::NoLabel(b)) if b == "x"));
    }

    #[test]
    fn extract_rejects_conflicting_labels() {
        let schema = ActionSchema {
            action: "a".to_string(),
            preconditions: vec![
                Predicate::NodeExists {
                    bind: "x".to_string(),
                    label: "File".to_string(),
                },
                Predicate::NodeExists {
                    bind: "x".to_string(),
                    label: "Project".to_string(),
                },
            ],
            effects: vec![],
            provenance: Provenance::Given,
        };
        assert!(matches!(SliceSpec::extract(&schema), Err(SliceError::AmbiguousLabel(b)) if b == "x"));
    }

    // ---- build_slice end to end (observed through predict) -----------------

    #[tokio::test]
    async fn builds_a_slice_a_valid_prediction_can_use() {
        let schema = tag_schema("/home/tim/proj");
        let graph = MockGraph::new()
            .on(
                "n:File {id: 'f1'}",
                vec![row(&[
                    ("id", Value::String("f1".into())),
                    ("path", Value::String("/raw/f".into())),
                ])],
            )
            .on("n:Project {id: 'p1'}", vec![row(&[("id", Value::String("p1".into()))])])
            // no FILE_PART_OF edge yet
            .on("count(*) AS cnt", vec![row(&[("cnt", serde_json::json!(0))])]);
        let resolver = MockResolver::new(&[("/raw/f", "/home/tim/proj/foo.rs")]);
        let mounts = StaticMountPolicy::empty();

        let (state, bnd) = build_slice(
            &schema,
            "graph.write",
            &bindings(&[("file", "f1"), ("proj", "p1")]),
            &graph,
            &resolver,
            &mounts,
        )
        .await
        .unwrap();

        let cap = capability();
        let p = predict(&schema, &bnd, &state, &ctx(&cap, "graph.write"));
        assert!(p.is_valid(), "expected Valid, got {p:?}");
    }

    #[tokio::test]
    async fn an_assert_edge_schema_without_an_absence_precondition_fails_closed_when_the_edge_exists() {
        // Schema drift: the rule asserts the edge but omits Not(EdgeExists). The
        // slice still loads the target edge's prior presence (the strict-create
        // safeguard), so when the real graph already has the edge the assertion
        // is an EffectError, not a Valid clean create whose derived inverse would
        // delete an edge the action never created.
        let schema = ActionSchema {
            action: "graph.write".to_string(),
            preconditions: vec![
                Predicate::NodeExists { bind: "file".to_string(), label: "File".to_string() },
                Predicate::NodeExists { bind: "proj".to_string(), label: "Project".to_string() },
                // No Not(EdgeExists), deliberately.
            ],
            effects: vec![Effect::AssertEdge {
                from: "file".to_string(),
                edge: "FILE_PART_OF".to_string(),
                to: "proj".to_string(),
            }],
            provenance: Provenance::Given,
        };
        let graph = MockGraph::new()
            .on("n:File {id: 'f1'}", vec![row(&[("id", Value::String("f1".into()))])])
            .on("n:Project {id: 'p1'}", vec![row(&[("id", Value::String("p1".into()))])])
            // The FILE_PART_OF edge already exists in the real graph.
            .on("count(*) AS cnt", vec![row(&[("cnt", serde_json::json!(1))])]);
        let (state, bnd) = build_slice(
            &schema,
            "graph.write",
            &bindings(&[("file", "f1"), ("proj", "p1")]),
            &graph,
            &MockResolver::new(&[]),
            &StaticMountPolicy::empty(),
        )
        .await
        .unwrap();
        let cap = capability();
        let p = predict(&schema, &bnd, &state, &ctx(&cap, "graph.write"));
        assert!(matches!(p, Prediction::EffectError { .. }), "got {p:?}");
        assert!(p.predicted_state().is_none());
    }

    #[tokio::test]
    async fn an_unresolvable_path_field_is_omitted_and_fails_closed() {
        let schema = tag_schema("/home/tim/proj");
        let graph = MockGraph::new()
            .on(
                "n:File {id: 'f1'}",
                vec![row(&[
                    ("id", Value::String("f1".into())),
                    ("path", Value::String("/gone".into())),
                ])],
            )
            .on("n:Project {id: 'p1'}", vec![row(&[("id", Value::String("p1".into()))])])
            .on("count(*) AS cnt", vec![row(&[("cnt", serde_json::json!(0))])]);
        // The resolver has no mapping for "/gone": resolution fails.
        let resolver = MockResolver::new(&[]);
        let (state, bnd) = build_slice(
            &schema,
            "graph.write",
            &bindings(&[("file", "f1"), ("proj", "p1")]),
            &graph,
            &resolver,
            &StaticMountPolicy::empty(),
        )
        .await
        .unwrap();
        let cap = capability();
        let p = predict(&schema, &bnd, &state, &ctx(&cap, "graph.write"));
        // PathUnder cannot be judged: the precondition fails closed.
        assert!(matches!(
            p,
            Prediction::PreconditionsFailed { failed }
                if failed.iter().any(|x| matches!(x, Predicate::PathUnder { .. }))
        ));
    }

    #[tokio::test]
    async fn an_absent_node_is_not_an_error_but_fails_the_precondition() {
        let schema = tag_schema("/home/tim/proj");
        // The project node returns no rows (absent); the file is present.
        let graph = MockGraph::new()
            .on(
                "n:File {id: 'f1'}",
                vec![row(&[
                    ("id", Value::String("f1".into())),
                    ("path", Value::String("/raw/f".into())),
                ])],
            )
            .on("count(*) AS cnt", vec![row(&[("cnt", serde_json::json!(0))])]);
        let resolver = MockResolver::new(&[("/raw/f", "/home/tim/proj/foo.rs")]);
        let (state, bnd) = build_slice(
            &schema,
            "graph.write",
            &bindings(&[("file", "f1"), ("proj", "p1")]),
            &graph,
            &resolver,
            &StaticMountPolicy::empty(),
        )
        .await
        .unwrap();
        let cap = capability();
        let p = predict(&schema, &bnd, &state, &ctx(&cap, "graph.write"));
        assert!(matches!(
            p,
            Prediction::PreconditionsFailed { failed }
                if failed.iter().any(|x| matches!(x, Predicate::NodeExists { bind, .. } if bind == "proj"))
        ));
    }

    #[tokio::test]
    async fn a_present_edge_fails_the_negation() {
        let schema = tag_schema("/home/tim/proj");
        let graph = MockGraph::new()
            .on(
                "n:File {id: 'f1'}",
                vec![row(&[
                    ("id", Value::String("f1".into())),
                    ("path", Value::String("/raw/f".into())),
                ])],
            )
            .on("n:Project {id: 'p1'}", vec![row(&[("id", Value::String("p1".into()))])])
            // the edge already exists
            .on("count(*) AS cnt", vec![row(&[("cnt", serde_json::json!(1))])]);
        let resolver = MockResolver::new(&[("/raw/f", "/home/tim/proj/foo.rs")]);
        let (state, bnd) = build_slice(
            &schema,
            "graph.write",
            &bindings(&[("file", "f1"), ("proj", "p1")]),
            &graph,
            &resolver,
            &StaticMountPolicy::empty(),
        )
        .await
        .unwrap();
        let cap = capability();
        let p = predict(&schema, &bnd, &state, &ctx(&cap, "graph.write"));
        assert!(matches!(
            p,
            Prediction::PreconditionsFailed { failed }
                if failed.iter().any(|x| matches!(x, Predicate::Not(_)))
        ));
    }

    #[tokio::test]
    async fn a_graph_read_failure_is_a_slice_error() {
        let schema = tag_schema("/home/tim/proj");
        let graph = MockGraph::new().err_on("n:File");
        let err = build_slice(
            &schema,
            "graph.write",
            &bindings(&[("file", "f1"), ("proj", "p1")]),
            &graph,
            &MockResolver::new(&[]),
            &StaticMountPolicy::empty(),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, SliceError::GraphRead(_)));
    }

    #[tokio::test]
    async fn an_ambiguous_id_lookup_is_a_slice_error() {
        let schema = tag_schema("/home/tim/proj");
        let graph = MockGraph::new().on(
            "n:File {id: 'f1'}",
            vec![
                row(&[("path", Value::String("/a".into()))]),
                row(&[("path", Value::String("/b".into()))]),
            ],
        );
        let err = build_slice(
            &schema,
            "graph.write",
            &bindings(&[("file", "f1"), ("proj", "p1")]),
            &graph,
            &MockResolver::new(&[("/a", "/a"), ("/b", "/b")]),
            &StaticMountPolicy::empty(),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, SliceError::AmbiguousNode(id) if id == "f1"));
    }

    #[tokio::test]
    async fn a_writable_path_precondition_loads_the_mount_policy() {
        // A schema whose only precondition is NotReadOnly on a path binding.
        let schema = ActionSchema {
            action: "fs.write".to_string(),
            preconditions: vec![Predicate::NotReadOnly {
                bind: "dest".to_string(),
            }],
            effects: vec![],
            provenance: Provenance::Given,
        };
        let cap = capability();

        // Under a read-only mount: the precondition fails. The resolver maps
        // the operand to itself (already canonical).
        let (state, bnd) = build_slice(
            &schema,
            "fs.write",
            &bindings(&[("dest", "/mnt/ro/x")]),
            &MockGraph::new(),
            &MockResolver::new(&[("/mnt/ro/x", "/mnt/ro/x")]),
            &StaticMountPolicy::new(["/mnt/ro".to_string()]),
        )
        .await
        .unwrap();
        let p = predict(&schema, &bnd, &state, &ctx(&cap, "fs.write"));
        assert!(matches!(p, Prediction::PreconditionsFailed { .. }));

        // No read-only mounts: the same write is permitted.
        let (state, bnd) = build_slice(
            &schema,
            "fs.write",
            &bindings(&[("dest", "/home/tim/x")]),
            &MockGraph::new(),
            &MockResolver::new(&[("/home/tim/x", "/home/tim/x")]),
            &StaticMountPolicy::empty(),
        )
        .await
        .unwrap();
        let p = predict(&schema, &bnd, &state, &ctx(&cap, "fs.write"));
        assert!(p.is_valid(), "expected Valid, got {p:?}");
    }

    #[tokio::test]
    async fn a_mount_policy_failure_fails_closed() {
        struct FailingPolicy;
        impl MountPolicy for FailingPolicy {
            fn read_only_prefixes(&self) -> Result<BTreeSet<String>, SliceError> {
                Err(SliceError::MountPolicy("no /proc".to_string()))
            }
        }
        let schema = ActionSchema {
            action: "fs.write".to_string(),
            preconditions: vec![Predicate::NotReadOnly {
                bind: "dest".to_string(),
            }],
            effects: vec![],
            provenance: Provenance::Given,
        };
        let err = build_slice(
            &schema,
            "fs.write",
            &bindings(&[("dest", "/home/tim/x")]),
            &MockGraph::new(),
            &MockResolver::new(&[]),
            &FailingPolicy,
        )
        .await
        .unwrap_err();
        assert!(matches!(err, SliceError::MountPolicy(_)));
    }

    #[tokio::test]
    async fn a_binding_id_with_a_quote_is_escaped_into_the_query() {
        // The id carries a single quote; the query the mock sees must contain
        // the escaped form, never a raw quote that could close the literal.
        struct Spy {
            seen: std::sync::Mutex<Vec<String>>,
        }
        #[async_trait::async_trait]
        impl GraphHandle for Spy {
            async fn query(
                &self,
                cypher: &str,
            ) -> Result<Vec<HashMap<String, Value>>, GraphError> {
                self.seen.lock().unwrap().push(cypher.to_string());
                Ok(Vec::new())
            }
        }
        let schema = ActionSchema {
            action: "a".to_string(),
            preconditions: vec![Predicate::NodeExists {
                bind: "x".to_string(),
                label: "File".to_string(),
            }],
            effects: vec![],
            provenance: Provenance::Given,
        };
        let spy = Spy {
            seen: std::sync::Mutex::new(Vec::new()),
        };
        build_slice(
            &schema,
            "a",
            &bindings(&[("x", "a'b")]),
            &spy,
            &MockResolver::new(&[]),
            &StaticMountPolicy::empty(),
        )
        .await
        .unwrap();
        let seen = spy.seen.lock().unwrap();
        assert!(seen[0].contains("'a\\'b'"), "query was {:?}", seen[0]);
        assert!(seen[0].contains("LIMIT 2"), "node query must be bounded: {:?}", seen[0]);
    }

    #[tokio::test]
    async fn set_field_values_are_type_checked_against_the_kg() {
        let with_set = |field: &str, value: &str| ActionSchema {
            action: "a".to_string(),
            preconditions: vec![Predicate::NodeExists {
                bind: "p".to_string(),
                label: "Project".to_string(),
            }],
            effects: vec![Effect::SetField {
                bind: "p".to_string(),
                field: field.to_string(),
                value: value.to_string(),
            }],
            provenance: Provenance::Given,
        };
        let run = |schema: ActionSchema| async move {
            build_slice(
                &schema,
                "a",
                &bindings(&[("p", "p1")]),
                &MockGraph::new(),
                &MockResolver::new(&[]),
                &StaticMountPolicy::empty(),
            )
            .await
        };
        // confidence is an Int column; non-numeric text is not representable.
        assert!(matches!(
            run(with_set("confidence", "lots")).await,
            Err(SliceError::SchemaViolation(_))
        ));
        // promoted is a Bool column; arbitrary text is not representable.
        assert!(matches!(
            run(with_set("promoted", "maybe")).await,
            Err(SliceError::SchemaViolation(_))
        ));
        // A valid Int value passes validation.
        assert!(run(with_set("confidence", "90")).await.is_ok());
    }

    #[tokio::test]
    async fn an_over_long_binding_id_is_rejected() {
        let schema = ActionSchema {
            action: "a".to_string(),
            preconditions: vec![Predicate::NodeExists {
                bind: "x".to_string(),
                label: "File".to_string(),
            }],
            effects: vec![],
            provenance: Provenance::Given,
        };
        let huge = "a".repeat(MAX_BINDING_LEN + 1);
        let err = build_slice(
            &schema,
            "a",
            &bindings(&[("x", huge.as_str())]),
            &MockGraph::new(),
            &MockResolver::new(&[]),
            &StaticMountPolicy::empty(),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, SliceError::BadBinding(_)));
    }

    #[tokio::test]
    async fn an_id_reused_across_labels_is_rejected() {
        // Two bindings resolve to the same id under different labels: the
        // id-keyed world state cannot hold both, so the slice refuses.
        let schema = ActionSchema {
            action: "a".to_string(),
            preconditions: vec![
                Predicate::NodeExists {
                    bind: "a".to_string(),
                    label: "File".to_string(),
                },
                Predicate::NodeExists {
                    bind: "b".to_string(),
                    label: "Project".to_string(),
                },
            ],
            effects: vec![],
            provenance: Provenance::Given,
        };
        let graph = MockGraph::new()
            .on("n:File", vec![row(&[("id", Value::String("x".into()))])])
            .on("n:Project", vec![row(&[("id", Value::String("x".into()))])]);
        let err = build_slice(
            &schema,
            "a",
            &bindings(&[("a", "x"), ("b", "x")]),
            &graph,
            &MockResolver::new(&[]),
            &StaticMountPolicy::empty(),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, SliceError::IdCollision(id) if id == "x"));
    }

    #[tokio::test]
    async fn a_writable_path_that_resolves_into_a_read_only_mount_is_blocked() {
        // The classic symlink hole: the raw operand looks writable, but it
        // resolves inside a read-only mount. Canonicalizing the binding before
        // the lexical check closes it.
        let schema = ActionSchema {
            action: "fs.write".to_string(),
            preconditions: vec![Predicate::NotReadOnly {
                bind: "dest".to_string(),
            }],
            effects: vec![],
            provenance: Provenance::Given,
        };
        let (state, bnd) = build_slice(
            &schema,
            "fs.write",
            &bindings(&[("dest", "/link/x")]),
            &MockGraph::new(),
            &MockResolver::new(&[("/link/x", "/mnt/ro/real")]),
            &StaticMountPolicy::new(["/mnt/ro".to_string()]),
        )
        .await
        .unwrap();
        assert_eq!(bnd.get("dest"), Some(&"/mnt/ro/real".to_string()));
        let cap = capability();
        let p = predict(&schema, &bnd, &state, &ctx(&cap, "fs.write"));
        assert!(matches!(p, Prediction::PreconditionsFailed { .. }));
    }

    #[tokio::test]
    async fn a_malformed_edge_count_is_a_slice_error() {
        let schema = tag_schema("/home/tim/proj");
        let graph = MockGraph::new()
            .on(
                "n:File {id: 'f1'}",
                vec![row(&[
                    ("id", Value::String("f1".into())),
                    ("path", Value::String("/raw/f".into())),
                ])],
            )
            .on("n:Project {id: 'p1'}", vec![row(&[("id", Value::String("p1".into()))])])
            // a degraded response: cnt is not numeric
            .on("count(*) AS cnt", vec![row(&[("cnt", Value::String("1".into()))])]);
        let err = build_slice(
            &schema,
            "graph.write",
            &bindings(&[("file", "f1"), ("proj", "p1")]),
            &graph,
            &MockResolver::new(&[("/raw/f", "/home/tim/proj/foo.rs")]),
            &StaticMountPolicy::empty(),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, SliceError::MalformedResult(_)));
    }

    #[tokio::test]
    async fn an_id_reused_across_labels_is_rejected_even_when_one_is_absent() {
        // The dangerous case: File(x) is absent (zero rows) but Project(x) is
        // present. Without an up-front declared-id check, an effect on the
        // absent File binding could alias onto the loaded Project node. The
        // check fires before any read, so no graph rows are even needed.
        let schema = ActionSchema {
            action: "a".to_string(),
            preconditions: vec![
                Predicate::Not(Box::new(Predicate::NodeExists {
                    bind: "file".to_string(),
                    label: "File".to_string(),
                })),
                Predicate::NodeExists {
                    bind: "proj".to_string(),
                    label: "Project".to_string(),
                },
            ],
            effects: vec![],
            provenance: Provenance::Given,
        };
        let err = build_slice(
            &schema,
            "a",
            &bindings(&[("file", "x"), ("proj", "x")]),
            &MockGraph::new(), // returns no rows for anything
            &MockResolver::new(&[]),
            &StaticMountPolicy::empty(),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, SliceError::IdCollision(id) if id == "x"));
    }

    #[tokio::test]
    async fn a_node_row_without_the_requested_id_is_malformed() {
        let schema = ActionSchema {
            action: "a".to_string(),
            preconditions: vec![Predicate::NodeExists {
                bind: "x".to_string(),
                label: "File".to_string(),
            }],
            effects: vec![],
            provenance: Provenance::Given,
        };
        // A degraded single row that omits the id column.
        let graph = MockGraph::new().on(
            "n:File {id: 'f1'}",
            vec![row(&[("path", Value::String("/p".into()))])],
        );
        let err = build_slice(
            &schema,
            "a",
            &bindings(&[("x", "f1")]),
            &graph,
            &MockResolver::new(&[]),
            &StaticMountPolicy::empty(),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, SliceError::MalformedResult(_)));

        // A row that returns a different id is also malformed.
        let graph = MockGraph::new().on(
            "n:File {id: 'f1'}",
            vec![row(&[("id", Value::String("other".into()))])],
        );
        let err = build_slice(
            &schema,
            "a",
            &bindings(&[("x", "f1")]),
            &graph,
            &MockResolver::new(&[]),
            &StaticMountPolicy::empty(),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, SliceError::MalformedResult(_)));
    }

    #[test]
    fn fs_resolver_rejects_a_dangling_symlink_but_resolves_a_new_file() {
        use std::os::unix::fs::symlink;
        let base = std::env::temp_dir().join("lunaris-slice-resolver-test");
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        let resolver = FsPathResolver;

        // A dangling symlink exists but does not resolve: it must be rejected,
        // not blessed as a lexical new-file path.
        let link = base.join("link");
        symlink(base.join("missing-target"), &link).unwrap();
        assert!(
            resolver.resolve(link.to_str().unwrap()).is_err(),
            "a dangling symlink must be rejected"
        );

        // A genuinely new file in a real directory resolves via its parent.
        let new_file = base.join("new.txt");
        let resolved = resolver.resolve(new_file.to_str().unwrap()).unwrap();
        assert!(resolved.ends_with("new.txt"), "resolved to {resolved}");

        // A relative path is ambiguous and rejected.
        assert!(resolver.resolve("relative/path").is_err());

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn a_binding_used_as_both_node_and_path_is_rejected() {
        let schema = ActionSchema {
            action: "a".to_string(),
            preconditions: vec![
                Predicate::NodeExists {
                    bind: "x".to_string(),
                    label: "File".to_string(),
                },
                Predicate::NotReadOnly {
                    bind: "x".to_string(),
                },
            ],
            effects: vec![],
            provenance: Provenance::Given,
        };
        assert!(matches!(
            SliceSpec::extract(&schema),
            Err(SliceError::ContradictoryBinding(b)) if b == "x"
        ));
    }

    #[test]
    fn an_effect_node_operand_cannot_reuse_a_path_binding() {
        // `NotReadOnly` makes `dest` a path; a supported effect must not then
        // treat the canonicalized path as a node id (File nodes are keyed by
        // path, so it could alias onto a real node). (RetractNode is covered
        // separately: it is refused outright as a node-level mutation.)
        for effect in [
            Effect::SetField {
                bind: "dest".to_string(),
                field: "f".to_string(),
                value: "v".to_string(),
            },
            Effect::AssertEdge {
                from: "dest".to_string(),
                edge: "E".to_string(),
                to: "dest".to_string(),
            },
        ] {
            let schema = ActionSchema {
                action: "a".to_string(),
                preconditions: vec![Predicate::NotReadOnly {
                    bind: "dest".to_string(),
                }],
                effects: vec![effect],
                provenance: Provenance::Given,
            };
            assert!(
                matches!(SliceSpec::extract(&schema), Err(SliceError::ContradictoryBinding(b)) if b == "dest"),
                "an effect reusing a path binding as a node must be rejected"
            );
        }
    }

    #[test]
    fn node_level_mutations_are_refused_until_the_richer_model_exists() {
        // AssertNode (create) and RetractNode (delete) both have effects the
        // bounded, id-keyed slice cannot soundly represent, so they are
        // refused rather than predicted with an understated impact.
        for effect in [
            Effect::AssertNode {
                bind: "new".to_string(),
                label: "Tag".to_string(),
            },
            Effect::RetractNode {
                bind: "old".to_string(),
            },
        ] {
            let schema = ActionSchema {
                action: "graph.write".to_string(),
                preconditions: vec![Predicate::NodeExists {
                    bind: "old".to_string(),
                    label: "Tag".to_string(),
                }],
                effects: vec![effect],
                provenance: Provenance::Given,
            };
            assert!(
                matches!(SliceSpec::extract(&schema), Err(SliceError::UnsupportedEffect(_))),
                "a node-level mutation must be refused"
            );
        }
    }

    #[test]
    fn fs_resolver_rejects_a_tilde_path() {
        // `~` is not absolute; expanding it is an upstream concern, not the
        // filesystem trust boundary's.
        assert!(FsPathResolver.resolve("~/Repositories/x").is_err());
    }

    #[tokio::test]
    async fn an_untrusted_or_wrong_action_schema_does_no_reads() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        struct CountingGraph(AtomicUsize);
        #[async_trait::async_trait]
        impl GraphHandle for CountingGraph {
            async fn query(
                &self,
                _cypher: &str,
            ) -> Result<Vec<HashMap<String, Value>>, GraphError> {
                self.0.fetch_add(1, Ordering::SeqCst);
                Ok(Vec::new())
            }
        }

        // An unapproved learned schema is refused before any read.
        let learned = ActionSchema {
            action: "graph.write".to_string(),
            preconditions: vec![Predicate::NodeExists {
                bind: "file".to_string(),
                label: "File".to_string(),
            }],
            effects: vec![],
            provenance: Provenance::Learned {
                approved_by: "   ".to_string(),
            },
        };
        let graph = CountingGraph(AtomicUsize::new(0));
        let err = build_slice(
            &learned,
            "graph.write",
            &bindings(&[("file", "f1")]),
            &graph,
            &MockResolver::new(&[]),
            &StaticMountPolicy::empty(),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, SliceError::UntrustedSchema(_)));
        assert_eq!(graph.0.load(Ordering::SeqCst), 0, "an untrusted schema must not read");

        // A schema for a different action than the invocation is refused
        // before any read.
        let given = tag_schema("/r");
        let graph = CountingGraph(AtomicUsize::new(0));
        let err = build_slice(
            &given,
            "different.action",
            &bindings(&[("file", "f1"), ("proj", "p1")]),
            &graph,
            &MockResolver::new(&[]),
            &StaticMountPolicy::empty(),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, SliceError::SchemaMismatch { .. }));
        assert_eq!(graph.0.load(Ordering::SeqCst), 0, "a wrong-action schema must not read");
    }

    #[test]
    fn effect_only_identifiers_are_validated() {
        // An AssertEdge whose edge type is not a safe identifier.
        let schema = ActionSchema {
            action: "a".to_string(),
            preconditions: vec![
                Predicate::NodeExists {
                    bind: "x".to_string(),
                    label: "File".to_string(),
                },
                Predicate::NodeExists {
                    bind: "y".to_string(),
                    label: "Project".to_string(),
                },
            ],
            effects: vec![Effect::AssertEdge {
                from: "x".to_string(),
                edge: "bad-edge".to_string(),
                to: "y".to_string(),
            }],
            provenance: Provenance::Given,
        };
        assert!(matches!(SliceSpec::extract(&schema), Err(SliceError::BadIdentifier(_))));

        // A SetField whose field name is not a safe identifier.
        let schema = ActionSchema {
            action: "a".to_string(),
            preconditions: vec![Predicate::NodeExists {
                bind: "x".to_string(),
                label: "File".to_string(),
            }],
            effects: vec![Effect::SetField {
                bind: "x".to_string(),
                field: "bad field".to_string(),
                value: "v".to_string(),
            }],
            provenance: Provenance::Given,
        };
        assert!(matches!(SliceSpec::extract(&schema), Err(SliceError::BadIdentifier(_))));
    }

    #[test]
    fn an_unlabelled_effect_operand_is_refused() {
        // An effect touches a node that no precondition gives a label. Under
        // id keying it could alias a loaded node and mutate it while skipping
        // KG validation, so it is refused before any prediction.
        for effect in [
            Effect::SetField {
                bind: "y".to_string(),
                field: "name".to_string(),
                value: "v".to_string(),
            },
            Effect::AssertEdge {
                from: "x".to_string(),
                edge: "FILE_PART_OF".to_string(),
                to: "y".to_string(),
            },
            Effect::RetractEdge {
                from: "x".to_string(),
                edge: "FILE_PART_OF".to_string(),
                to: "y".to_string(),
            },
        ] {
            // `x` is labelled, `y` is not.
            let schema = ActionSchema {
                action: "a".to_string(),
                preconditions: vec![Predicate::NodeExists {
                    bind: "x".to_string(),
                    label: "File".to_string(),
                }],
                effects: vec![effect],
                provenance: Provenance::Given,
            };
            assert!(
                matches!(SliceSpec::extract(&schema), Err(SliceError::NoLabel(b)) if b == "y"),
                "an unlabelled effect operand must be refused"
            );
        }
    }

    #[tokio::test]
    async fn schema_violations_against_the_kg_are_refused_before_reads() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        struct CountingGraph(AtomicUsize);
        #[async_trait::async_trait]
        impl GraphHandle for CountingGraph {
            async fn query(
                &self,
                _cypher: &str,
            ) -> Result<Vec<HashMap<String, Value>>, GraphError> {
                self.0.fetch_add(1, Ordering::SeqCst);
                Ok(Vec::new())
            }
        }
        let run = |schema: ActionSchema, binds: Vec<(&'static str, &'static str)>| async move {
            let graph = CountingGraph(AtomicUsize::new(0));
            let owned: Vec<(String, String)> =
                binds.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
            let b: Bindings = owned.into_iter().collect();
            let r = build_slice(
                &schema,
                "a",
                &b,
                &graph,
                &MockResolver::new(&[]),
                &StaticMountPolicy::empty(),
            )
            .await;
            (r, graph.0.load(Ordering::SeqCst))
        };

        // An edge between the wrong endpoint labels (FILE_PART_OF is File->Project).
        let wrong_edge = ActionSchema {
            action: "a".to_string(),
            preconditions: vec![
                Predicate::NodeExists {
                    bind: "f".to_string(),
                    label: "File".to_string(),
                },
                Predicate::NodeExists {
                    bind: "app".to_string(),
                    label: "App".to_string(),
                },
                Predicate::EdgeExists {
                    from: "f".to_string(),
                    edge: "FILE_PART_OF".to_string(),
                    to: "app".to_string(),
                },
            ],
            effects: vec![],
            provenance: Provenance::Given,
        };
        let (r, reads) = run(wrong_edge, vec![("f", "f1"), ("app", "a1")]).await;
        assert!(matches!(r, Err(SliceError::SchemaViolation(_))));
        assert_eq!(reads, 0, "a schema violation must be caught before any read");

        // A field that does not exist on the label.
        let bad_field = ActionSchema {
            action: "a".to_string(),
            preconditions: vec![
                Predicate::NodeExists {
                    bind: "f".to_string(),
                    label: "File".to_string(),
                },
                Predicate::FieldCmp {
                    bind: "f".to_string(),
                    field: "nonexistent".to_string(),
                    op: CmpOp::Eq,
                    value: "x".to_string(),
                },
            ],
            effects: vec![],
            provenance: Provenance::Given,
        };
        let (r, _) = run(bad_field, vec![("f", "f1")]).await;
        assert!(matches!(r, Err(SliceError::SchemaViolation(_))));

        // An unknown edge type.
        let bad_edge = ActionSchema {
            action: "a".to_string(),
            preconditions: vec![
                Predicate::NodeExists {
                    bind: "f".to_string(),
                    label: "File".to_string(),
                },
                Predicate::NodeExists {
                    bind: "p".to_string(),
                    label: "Project".to_string(),
                },
                Predicate::EdgeExists {
                    from: "f".to_string(),
                    edge: "NO_SUCH_EDGE".to_string(),
                    to: "p".to_string(),
                },
            ],
            effects: vec![],
            provenance: Provenance::Given,
        };
        let (r, _) = run(bad_edge, vec![("f", "f1"), ("p", "p1")]).await;
        assert!(matches!(r, Err(SliceError::SchemaViolation(_))));
    }

    // The graph.write registry rule needs the file's path under the project's
    // root, both nodes present, and the edge absent.
    fn graph_write_graph(file_path: &str) -> MockGraph {
        MockGraph::new()
            .on(
                "n:File {id: 'f1'}",
                vec![row(&[
                    ("id", Value::String("f1".into())),
                    ("path", Value::String(file_path.into())),
                ])],
            )
            .on(
                "n:Project {id: 'p1'}",
                vec![row(&[
                    ("id", Value::String("p1".into())),
                    ("root_path", Value::String("/home/tim/proj".into())),
                ])],
            )
            .on("count(*) AS cnt", vec![row(&[("cnt", serde_json::json!(0))])])
    }

    #[tokio::test]
    async fn build_slice_trusted_runs_through_the_registry() {
        // The gate path's entry: a registry-resolved schema drives the slice.
        // The file lies under the project root, so the link is provable.
        let trusted = crate::registry::lookup("graph.write").expect("graph.write registered");
        let graph = graph_write_graph("/raw/under");
        let resolver = MockResolver::new(&[
            ("/raw/under", "/home/tim/proj/foo.rs"),
            ("/home/tim/proj", "/home/tim/proj"),
        ]);
        let (state, bnd) = build_slice_trusted(
            &trusted,
            "graph.write",
            &bindings(&[("file", "f1"), ("project", "p1")]),
            &graph,
            &resolver,
            &StaticMountPolicy::empty(),
        )
        .await
        .unwrap();
        let cap = capability();
        let p = predict(trusted.schema(), &bnd, &state, &ctx(&cap, "graph.write"));
        assert!(p.is_valid(), "expected Valid, got {p:?}");
    }

    #[tokio::test]
    async fn the_registry_rule_refuses_a_file_outside_the_project_root() {
        // A file that does NOT lie under the project root must not validate:
        // the rule proves project membership, not merely that both nodes exist.
        let trusted = crate::registry::lookup("graph.write").unwrap();
        let graph = graph_write_graph("/raw/outside");
        let resolver = MockResolver::new(&[
            ("/raw/outside", "/home/tim/elsewhere/foo.rs"),
            ("/home/tim/proj", "/home/tim/proj"),
        ]);
        let (state, bnd) = build_slice_trusted(
            &trusted,
            "graph.write",
            &bindings(&[("file", "f1"), ("project", "p1")]),
            &graph,
            &resolver,
            &StaticMountPolicy::empty(),
        )
        .await
        .unwrap();
        let cap = capability();
        let p = predict(trusted.schema(), &bnd, &state, &ctx(&cap, "graph.write"));
        assert!(
            matches!(
                &p,
                Prediction::PreconditionsFailed { failed }
                    if failed.iter().any(|x| matches!(x, Predicate::PathUnderField { .. }))
            ),
            "expected the path-containment precondition to fail, got {p:?}"
        );
    }

    #[test]
    fn oversized_schemas_are_refused() {
        // More node bindings than the slice will load.
        let many_nodes = ActionSchema {
            action: "a".to_string(),
            preconditions: (0..=MAX_SLICE_NODES)
                .map(|i| Predicate::NodeExists {
                    bind: format!("n{i}"),
                    label: "File".to_string(),
                })
                .collect(),
            effects: vec![],
            provenance: Provenance::Given,
        };
        assert!(matches!(
            SliceSpec::extract(&many_nodes),
            Err(SliceError::SchemaTooLarge(_))
        ));

        // A predicate nested deeper than the walk will descend.
        let mut deep = Predicate::CapabilityAllows;
        for _ in 0..(MAX_PREDICATE_DEPTH + 2) {
            deep = Predicate::Not(Box::new(deep));
        }
        let deep_schema = ActionSchema {
            action: "a".to_string(),
            preconditions: vec![deep],
            effects: vec![],
            provenance: Provenance::Given,
        };
        assert!(matches!(
            SliceSpec::extract(&deep_schema),
            Err(SliceError::SchemaTooLarge(_))
        ));
    }
}
