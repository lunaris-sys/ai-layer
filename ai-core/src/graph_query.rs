//! Structured graph query DSL.
//!
//! Foundation §5.5 requires the AI daemon to turn a natural-language
//! prompt into a graph query that touches only known node types,
//! performs no writes, and is bounded. Letting the model emit raw
//! Cypher and validating it textually is bypassable: unlabeled
//! patterns slip past a namespace allowlist, and a `LIMIT` in a
//! non-final clause defeats a "has LIMIT" check.
//!
//! This module closes that class of bug structurally. The AI emits a
//! [`GraphQuery`] as JSON. Every label, edge, and field is validated
//! against [`crate::graph_schema`] and a capability
//! [`QueryScope`] *before* any Cypher exists. The daemon then builds
//! the Cypher itself via [`GraphQuery::to_cypher`]. The model never
//! produces Cypher text, so Cypher injection is not possible: the
//! query *structure* is constrained by the typed DSL, and the query
//! *values* are encoded as Cypher literals by deterministic,
//! tested code.
//!
//! The two-model-call shape from Foundation §5.5 is preserved by the
//! pipeline: call 1 produces this DSL, call 2 formats the result.
//! Only the intermediate format is JSON instead of Cypher text.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::graph_schema::{FieldType, GraphSchema};

/// Maximum traversal depth. Equals Foundation's "≤ 5 hops default";
/// because the DSL caps it structurally, the built Cypher can never
/// exceed it.
pub const MAX_TRAVERSE_STEPS: usize = 5;

/// Maximum filters on a single node pattern.
pub const MAX_FILTERS_PER_NODE: usize = 20;

/// Maximum fields a query may return.
pub const MAX_SELECT_FIELDS: usize = 50;

/// Upper clamp on the result limit.
pub const MAX_LIMIT: u32 = 1000;

/// Maximum length of the built Cypher string. Stays below the
/// Knowledge Daemon's 64 KiB request cap with headroom.
pub const MAX_CYPHER_BYTES: usize = 60 * 1024;

/// A structured graph query. The AI emits this as JSON.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GraphQuery {
    /// Root node pattern.
    pub from: NodePattern,
    /// Traversal steps from the root node.
    #[serde(default)]
    pub traverse: Vec<TraverseStep>,
    /// Fields to return. Must be non-empty.
    pub select: Vec<FieldRef>,
    /// Optional ordering.
    #[serde(default)]
    pub order_by: Option<OrderSpec>,
    /// Result cap. Clamped to `[1, MAX_LIMIT]` at build time.
    pub limit: u32,
}

/// A node pattern: a binding plus a schema label plus filters.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct NodePattern {
    /// Binding name used to refer to this node (e.g. `f`). Must be a
    /// valid identifier; it is the only DSL string that reaches the
    /// Cypher verbatim, so it is identifier-checked.
    pub bind: String,
    /// Node label. Must exist in the schema and be in scope.
    pub label: String,
    /// Property filters on this node.
    #[serde(default)]
    pub filters: Vec<Filter>,
}

/// One traversal hop.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TraverseStep {
    /// Relationship type. Must exist in the schema and be in scope.
    pub edge: String,
    /// Direction relative to the previous node.
    pub direction: Direction,
    /// Target node pattern.
    pub to: NodePattern,
}

/// Traversal direction.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum Direction {
    /// Follow the edge from the previous node outward.
    Outgoing,
    /// Follow the edge into the previous node.
    Incoming,
}

/// A property filter.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Filter {
    /// Property name on the owning node.
    pub field: String,
    /// Comparison operator.
    pub op: FilterOp,
    /// Typed comparison value.
    pub value: TypedValue,
}

/// Filter comparison operator.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum FilterOp {
    /// Equality.
    Eq,
    /// Inequality.
    Ne,
    /// Less than. Numeric fields only.
    Lt,
    /// Less than or equal. Numeric fields only.
    Le,
    /// Greater than. Numeric fields only.
    Gt,
    /// Greater than or equal. Numeric fields only.
    Ge,
    /// Substring containment. Text fields only.
    Contains,
    /// Prefix match. Text fields only.
    StartsWith,
}

/// A typed filter value. Deserialised untagged so the AI can write a
/// JSON string / number / bool directly.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(untagged)]
pub enum TypedValue {
    /// Boolean value.
    Bool(bool),
    /// Integer value. Timestamps are integers.
    Int(i64),
    /// Text value.
    Text(String),
}

impl TypedValue {
    /// The field type this value is compatible with.
    fn matches(&self, ty: FieldType) -> bool {
        matches!(
            (self, ty),
            (TypedValue::Bool(_), FieldType::Bool)
                | (TypedValue::Int(_), FieldType::Int)
                | (TypedValue::Text(_), FieldType::Text)
        )
    }
}

/// A reference to a returned field.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FieldRef {
    /// Binding of the node the field belongs to.
    pub bind: String,
    /// Property name.
    pub field: String,
}

/// Ordering specification.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OrderSpec {
    /// Binding of the node to order by.
    pub bind: String,
    /// Property name to order by.
    pub field: String,
    /// Descending order if true, ascending otherwise.
    #[serde(default)]
    pub descending: bool,
}

/// Foundation §8.4 read access tier.
///
/// Each tier corresponds to a label set via [`QueryScope::for_tier`].
/// Foundation models this as a single *global* AI access level (not a
/// per-caller grant): the user picks one level in Settings. Phase 9-α
/// pinned it to `Minimal`; S16 wires it live from `ai.toml`'s
/// `access_level`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessTier {
    /// Tier 0: no graph access at all.
    Minimal,
    /// Tier 1: current-session activity (Session, App, Event).
    SessionScoped,
    /// Tier 2: project structure (Project, File, Directory). The
    /// Focus Mode default.
    ProjectScoped,
    /// Tier 3: time-windowed activity across most node types.
    TimeScoped,
    /// Tier 4: full read access.
    Full,
}

/// Capability scope: the set of node and edge labels the caller may
/// touch. Derived from the caller's read tier (Foundation §8.4.6);
/// Phase 9-γ S16 adds the dynamic session/project/time constraints.
#[derive(Debug, Clone)]
pub struct QueryScope {
    allowed: std::collections::BTreeSet<String>,
}

impl QueryScope {
    /// Build a scope from an explicit label set.
    pub fn new<I, S>(labels: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            allowed: labels.into_iter().map(Into::into).collect(),
        }
    }

    /// A scope permitting every node and edge label in the schema
    /// (Foundation §8.4.6 tier 4, "Full read").
    pub fn full(schema: &GraphSchema) -> Self {
        let mut allowed = std::collections::BTreeSet::new();
        allowed.extend(schema.node_labels().map(str::to_string));
        allowed.extend(schema.edge_labels().map(str::to_string));
        Self { allowed }
    }

    /// Build a scope for a Foundation §8.4 read tier.
    ///
    /// Each tier maps to a fixed label allowlist: a `SessionScoped`
    /// caller cannot even name a `File` or `Project` label, a
    /// `ProjectScoped` one cannot name a `Session`, and so on. This
    /// coarse label gate is the enforcement S16 ships and is strictly
    /// better than unconditional full access.
    ///
    /// The finer *value* constraints Foundation attaches to tiers 1-3
    /// (only the current session's data, only the active project's
    /// subgraph, only within a configurable lookback window) are a
    /// follow-up: they need mandatory anchor/filter injection at Cypher
    /// compile time (the model cannot be trusted to self-restrict) plus
    /// a context source the daemon does not have yet — the current
    /// session id, the Focus-Mode active project, and the lookback
    /// setting. The same follow-up carries Focus Mode's automatic shift
    /// to Project-scoped while a project is focused.
    pub fn for_tier(tier: AccessTier, schema: &GraphSchema) -> Self {
        match tier {
            AccessTier::Minimal => Self::new(Vec::<&str>::new()),
            AccessTier::SessionScoped => {
                Self::new(["Session", "App", "Event", "ACTIVE_IN", "EMITTED_BY"])
            }
            AccessTier::ProjectScoped => Self::new([
                "Project",
                "File",
                "Directory",
                "FILE_PART_OF",
                "DIR_PART_OF",
            ]),
            AccessTier::TimeScoped => Self::new([
                "File",
                "App",
                "Session",
                "Event",
                "UserAction",
                "ACCESSED_BY",
                "ACTIVE_IN",
                "EMITTED_BY",
                "DERIVED_FROM",
            ]),
            AccessTier::Full => Self::full(schema),
        }
    }

    /// Whether `label` is permitted.
    pub fn permits(&self, label: &str) -> bool {
        self.allowed.contains(label)
    }

    /// Whether the scope permits no labels at all (the `Minimal`
    /// tier). A daemon holding an empty scope cannot answer any
    /// query, so the dispatch layer rejects up front instead of
    /// burning provider calls on a query that will always fail
    /// validation.
    pub fn is_empty(&self) -> bool {
        self.allowed.is_empty()
    }
}

/// Validation errors. Every variant is reported back to the model so
/// it can retry (Foundation §5.5).
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum DslError {
    /// A node label is not in the schema.
    #[error("unknown node label '{label}'")]
    UnknownLabel {
        /// The offending label.
        label: String,
    },
    /// An edge type is not in the schema.
    #[error("unknown edge type '{edge}'")]
    UnknownEdge {
        /// The offending edge type.
        edge: String,
    },
    /// A field does not exist on the given label.
    #[error("unknown field '{field}' on node '{label}'")]
    UnknownField {
        /// Owning node label.
        label: String,
        /// The offending field.
        field: String,
    },
    /// A label is valid but not permitted by the caller's scope.
    #[error("label '{label}' is outside the caller's access scope")]
    LabelNotInScope {
        /// The offending label.
        label: String,
    },
    /// A traversal step's edge does not connect the given labels in
    /// the requested direction.
    #[error(
        "edge '{edge}' does not connect '{from}' to '{to}' in the requested direction"
    )]
    EdgeEndpointMismatch {
        /// Edge type.
        edge: String,
        /// Source label as used in the query.
        from: String,
        /// Target label as used in the query.
        to: String,
    },
    /// A filter's value type does not match the field's type, or the
    /// operator is not valid for the field type.
    #[error("filter on '{label}.{field}' has an incompatible operator or value type")]
    FilterTypeMismatch {
        /// Owning node label.
        label: String,
        /// Field name.
        field: String,
    },
    /// A binding name is not a valid identifier.
    #[error("'{value}' is not a valid binding identifier")]
    InvalidBinding {
        /// The offending binding string.
        value: String,
    },
    /// Two node patterns share a binding name.
    #[error("binding '{bind}' is used by more than one node")]
    DuplicateBinding {
        /// The duplicated binding.
        bind: String,
    },
    /// A `select` / `order_by` refers to an unknown binding.
    #[error("reference to unknown binding '{bind}'")]
    UnknownBinding {
        /// The offending binding.
        bind: String,
    },
    /// The `select` list is empty.
    #[error("query selects no fields")]
    EmptySelect,
    /// The traversal exceeds [`MAX_TRAVERSE_STEPS`].
    #[error("query has {count} traversal steps, max is {MAX_TRAVERSE_STEPS}")]
    TooManyTraverseSteps {
        /// Actual step count.
        count: usize,
    },
    /// A node has more than [`MAX_FILTERS_PER_NODE`] filters.
    #[error("node '{bind}' has {count} filters, max is {MAX_FILTERS_PER_NODE}")]
    TooManyFilters {
        /// Owning binding.
        bind: String,
        /// Actual filter count.
        count: usize,
    },
    /// The `select` list exceeds [`MAX_SELECT_FIELDS`].
    #[error("query selects {count} fields, max is {MAX_SELECT_FIELDS}")]
    TooManySelectFields {
        /// Actual field count.
        count: usize,
    },
}

/// Errors from building Cypher out of a validated query.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum BuildError {
    /// A text value contained a character that cannot be safely
    /// encoded as a Cypher string literal (a control character other
    /// than tab / newline / carriage return).
    #[error("text value contains an unencodable control character")]
    UnencodableValue,
    /// The built Cypher exceeds [`MAX_CYPHER_BYTES`].
    #[error("built Cypher exceeds the size limit")]
    CypherTooLong,
}

impl GraphQuery {
    /// Validate the query against the schema and the caller's scope.
    ///
    /// On success the query is structurally safe to build Cypher
    /// from. On error the returned [`DslError`] is fed back to the
    /// model for a retry (Foundation §5.5).
    pub fn validate(&self, schema: &GraphSchema, scope: &QueryScope) -> Result<(), DslError> {
        if self.traverse.len() > MAX_TRAVERSE_STEPS {
            return Err(DslError::TooManyTraverseSteps {
                count: self.traverse.len(),
            });
        }
        if self.select.is_empty() {
            return Err(DslError::EmptySelect);
        }
        if self.select.len() > MAX_SELECT_FIELDS {
            return Err(DslError::TooManySelectFields {
                count: self.select.len(),
            });
        }

        // Collect bindings (bind -> label), checking identifiers and
        // uniqueness as we go.
        let mut bindings: HashMap<String, String> = HashMap::new();
        check_node_pattern(&self.from, schema, scope, &mut bindings)?;

        // Validate the traversal chain. `prev_label` tracks the node
        // the next edge departs from.
        let mut prev_label = self.from.label.clone();
        for step in &self.traverse {
            let edge = schema
                .edge(&step.edge)
                .ok_or_else(|| DslError::UnknownEdge {
                    edge: step.edge.clone(),
                })?;
            if !scope.permits(&step.edge) {
                return Err(DslError::LabelNotInScope {
                    label: step.edge.clone(),
                });
            }
            check_node_pattern(&step.to, schema, scope, &mut bindings)?;

            // The edge must connect prev_label and step.to.label in
            // the requested direction.
            let (want_from, want_to) = match step.direction {
                Direction::Outgoing => (prev_label.as_str(), step.to.label.as_str()),
                Direction::Incoming => (step.to.label.as_str(), prev_label.as_str()),
            };
            if edge.from != want_from || edge.to != want_to {
                return Err(DslError::EdgeEndpointMismatch {
                    edge: step.edge.clone(),
                    from: prev_label.clone(),
                    to: step.to.label.clone(),
                });
            }
            prev_label = step.to.label.clone();
        }

        // Validate select + order_by against the collected bindings.
        for field_ref in &self.select {
            check_field_ref(&field_ref.bind, &field_ref.field, &bindings, schema)?;
        }
        if let Some(order) = &self.order_by {
            check_field_ref(&order.bind, &order.field, &bindings, schema)?;
        }
        Ok(())
    }

    /// All node labels and edge types this query references. The
    /// post-build self-check uses this to confirm the builder emitted
    /// only the labels the validated DSL declared.
    pub fn referenced_labels(&self) -> std::collections::BTreeSet<String> {
        let mut out = std::collections::BTreeSet::new();
        out.insert(self.from.label.clone());
        for step in &self.traverse {
            out.insert(step.edge.clone());
            out.insert(step.to.label.clone());
        }
        out
    }

    /// Build a read-only Cypher query string. Assumes [`validate`]
    /// has already succeeded against the same schema.
    ///
    /// [`validate`]: GraphQuery::validate
    pub fn to_cypher(&self) -> Result<String, BuildError> {
        let mut cypher = String::new();

        // MATCH clause.
        cypher.push_str("MATCH ");
        cypher.push_str(&node_pattern_cypher(&self.from));
        for step in &self.traverse {
            match step.direction {
                Direction::Outgoing => {
                    cypher.push_str(&format!("-[:{}]->", step.edge));
                }
                Direction::Incoming => {
                    cypher.push_str(&format!("<-[:{}]-", step.edge));
                }
            }
            cypher.push_str(&node_pattern_cypher(&step.to));
        }
        cypher.push('\n');

        // WHERE clause: every filter on every node, AND-joined.
        let mut conditions: Vec<String> = Vec::new();
        collect_conditions(&self.from, &mut conditions)?;
        for step in &self.traverse {
            collect_conditions(&step.to, &mut conditions)?;
        }
        if !conditions.is_empty() {
            cypher.push_str("WHERE ");
            cypher.push_str(&conditions.join(" AND "));
            cypher.push('\n');
        }

        // RETURN clause.
        cypher.push_str("RETURN ");
        let returns: Vec<String> = self
            .select
            .iter()
            .map(|f| format!("{}.{}", f.bind, f.field))
            .collect();
        cypher.push_str(&returns.join(", "));
        cypher.push('\n');

        // ORDER BY clause.
        if let Some(order) = &self.order_by {
            cypher.push_str(&format!(
                "ORDER BY {}.{} {}\n",
                order.bind,
                order.field,
                if order.descending { "DESC" } else { "ASC" }
            ));
        }

        // LIMIT clause: always final, always present. The DSL field
        // is a u32, clamped here to [1, MAX_LIMIT].
        let limit = self.limit.clamp(1, MAX_LIMIT);
        cypher.push_str(&format!("LIMIT {limit}"));

        if cypher.len() > MAX_CYPHER_BYTES {
            return Err(BuildError::CypherTooLong);
        }
        Ok(cypher)
    }
}

fn check_node_pattern(
    pattern: &NodePattern,
    schema: &GraphSchema,
    scope: &QueryScope,
    bindings: &mut HashMap<String, String>,
) -> Result<(), DslError> {
    if !is_valid_identifier(&pattern.bind) {
        return Err(DslError::InvalidBinding {
            value: pattern.bind.clone(),
        });
    }
    if bindings.contains_key(&pattern.bind) {
        return Err(DslError::DuplicateBinding {
            bind: pattern.bind.clone(),
        });
    }
    let node = schema
        .node(&pattern.label)
        .ok_or_else(|| DslError::UnknownLabel {
            label: pattern.label.clone(),
        })?;
    if !scope.permits(&pattern.label) {
        return Err(DslError::LabelNotInScope {
            label: pattern.label.clone(),
        });
    }
    if pattern.filters.len() > MAX_FILTERS_PER_NODE {
        return Err(DslError::TooManyFilters {
            bind: pattern.bind.clone(),
            count: pattern.filters.len(),
        });
    }
    for filter in &pattern.filters {
        let field_ty = schema
            .field_type(&pattern.label, &filter.field)
            .ok_or_else(|| DslError::UnknownField {
                label: pattern.label.clone(),
                field: filter.field.clone(),
            })?;
        if !filter_is_well_typed(filter, field_ty) {
            return Err(DslError::FilterTypeMismatch {
                label: pattern.label.clone(),
                field: filter.field.clone(),
            });
        }
    }
    let _ = node;
    bindings.insert(pattern.bind.clone(), pattern.label.clone());
    Ok(())
}

fn check_field_ref(
    bind: &str,
    field: &str,
    bindings: &HashMap<String, String>,
    schema: &GraphSchema,
) -> Result<(), DslError> {
    let label = bindings
        .get(bind)
        .ok_or_else(|| DslError::UnknownBinding {
            bind: bind.to_string(),
        })?;
    if schema.field_type(label, field).is_none() {
        return Err(DslError::UnknownField {
            label: label.clone(),
            field: field.to_string(),
        });
    }
    Ok(())
}

/// A filter is well-typed when the value type matches the field type
/// and the operator is valid for that type.
fn filter_is_well_typed(filter: &Filter, field_ty: FieldType) -> bool {
    if !filter.value.matches(field_ty) {
        return false;
    }
    match filter.op {
        FilterOp::Eq | FilterOp::Ne => true,
        FilterOp::Lt | FilterOp::Le | FilterOp::Gt | FilterOp::Ge => {
            field_ty == FieldType::Int
        }
        FilterOp::Contains | FilterOp::StartsWith => field_ty == FieldType::Text,
    }
}

fn is_valid_identifier(s: &str) -> bool {
    if s.is_empty() || s.len() > 64 {
        return false;
    }
    let mut chars = s.chars();
    let first = chars.next().expect("non-empty");
    if !(first.is_ascii_alphabetic() || first == '_') {
        return false;
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn node_pattern_cypher(pattern: &NodePattern) -> String {
    // `bind` is identifier-checked and `label` is schema-checked, so
    // both are safe to interpolate.
    format!("({}:{})", pattern.bind, pattern.label)
}

fn collect_conditions(pattern: &NodePattern, out: &mut Vec<String>) -> Result<(), BuildError> {
    for filter in &pattern.filters {
        let op = match filter.op {
            FilterOp::Eq => "=",
            FilterOp::Ne => "<>",
            FilterOp::Lt => "<",
            FilterOp::Le => "<=",
            FilterOp::Gt => ">",
            FilterOp::Ge => ">=",
            FilterOp::Contains => "CONTAINS",
            FilterOp::StartsWith => "STARTS WITH",
        };
        let literal = encode_literal(&filter.value)?;
        out.push(format!(
            "{}.{} {} {}",
            pattern.bind, filter.field, op, literal
        ));
    }
    Ok(())
}

/// Encode a typed value as a Cypher literal.
///
/// Text values become single-quoted Cypher string literals with
/// backslash, quote, and tab / newline / carriage-return escaped.
/// Any other control character is rejected rather than emitted, so a
/// value cannot smuggle structure into the query.
pub fn encode_literal(value: &TypedValue) -> Result<String, BuildError> {
    match value {
        TypedValue::Bool(b) => Ok(if *b { "true".into() } else { "false".into() }),
        TypedValue::Int(n) => Ok(n.to_string()),
        TypedValue::Text(s) => {
            let mut out = String::with_capacity(s.len() + 2);
            out.push('\'');
            for ch in s.chars() {
                match ch {
                    '\\' => out.push_str("\\\\"),
                    '\'' => out.push_str("\\'"),
                    '\n' => out.push_str("\\n"),
                    '\r' => out.push_str("\\r"),
                    '\t' => out.push_str("\\t"),
                    c if (c as u32) < 0x20 => return Err(BuildError::UnencodableValue),
                    c => out.push(c),
                }
            }
            out.push('\'');
            Ok(out)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schema() -> GraphSchema {
        GraphSchema::knowledge_graph()
    }

    fn full_scope() -> QueryScope {
        QueryScope::full(&schema())
    }

    fn node(bind: &str, label: &str) -> NodePattern {
        NodePattern {
            bind: bind.to_string(),
            label: label.to_string(),
            filters: vec![],
        }
    }

    fn field(bind: &str, field: &str) -> FieldRef {
        FieldRef {
            bind: bind.to_string(),
            field: field.to_string(),
        }
    }

    fn simple_file_query() -> GraphQuery {
        GraphQuery {
            from: node("f", "File"),
            traverse: vec![],
            select: vec![field("f", "path")],
            order_by: None,
            limit: 50,
        }
    }

    #[test]
    fn valid_simple_query_passes_and_builds() {
        let q = simple_file_query();
        q.validate(&schema(), &full_scope()).expect("valid");
        let cypher = q.to_cypher().expect("builds");
        assert!(cypher.contains("MATCH (f:File)"));
        assert!(cypher.contains("RETURN f.path"));
        assert!(cypher.trim_end().ends_with("LIMIT 50"));
    }

    #[test]
    fn unknown_label_is_rejected() {
        let mut q = simple_file_query();
        q.from.label = "SecretTable".to_string();
        let err = q.validate(&schema(), &full_scope()).expect_err("reject");
        assert_eq!(
            err,
            DslError::UnknownLabel {
                label: "SecretTable".to_string()
            }
        );
    }

    #[test]
    fn label_outside_scope_is_rejected() {
        // Scope permits only Project, query asks for File.
        let scope = QueryScope::new(["Project"]);
        let err = simple_file_query()
            .validate(&schema(), &scope)
            .expect_err("reject");
        assert_eq!(
            err,
            DslError::LabelNotInScope {
                label: "File".to_string()
            }
        );
    }

    #[test]
    fn unknown_field_in_select_is_rejected() {
        let mut q = simple_file_query();
        q.select = vec![field("f", "secret_column")];
        let err = q.validate(&schema(), &full_scope()).expect_err("reject");
        assert_eq!(
            err,
            DslError::UnknownField {
                label: "File".to_string(),
                field: "secret_column".to_string()
            }
        );
    }

    #[test]
    fn empty_select_is_rejected() {
        let mut q = simple_file_query();
        q.select = vec![];
        assert_eq!(
            q.validate(&schema(), &full_scope()),
            Err(DslError::EmptySelect)
        );
    }

    #[test]
    fn invalid_binding_identifier_is_rejected() {
        let mut q = simple_file_query();
        // The classic injection attempt: smuggle Cypher via the bind.
        q.from.bind = "f) RETURN n //".to_string();
        let err = q.validate(&schema(), &full_scope()).expect_err("reject");
        assert!(matches!(err, DslError::InvalidBinding { .. }));
    }

    #[test]
    fn duplicate_binding_is_rejected() {
        let q = GraphQuery {
            from: node("x", "File"),
            traverse: vec![TraverseStep {
                edge: "ACCESSED_BY".to_string(),
                direction: Direction::Outgoing,
                to: node("x", "App"),
            }],
            select: vec![field("x", "id")],
            order_by: None,
            limit: 10,
        };
        let err = q.validate(&schema(), &full_scope()).expect_err("reject");
        assert_eq!(
            err,
            DslError::DuplicateBinding {
                bind: "x".to_string()
            }
        );
    }

    #[test]
    fn valid_traversal_passes_and_builds() {
        let q = GraphQuery {
            from: node("f", "File"),
            traverse: vec![TraverseStep {
                edge: "ACCESSED_BY".to_string(),
                direction: Direction::Outgoing,
                to: node("a", "App"),
            }],
            select: vec![field("f", "path"), field("a", "name")],
            order_by: None,
            limit: 25,
        };
        q.validate(&schema(), &full_scope()).expect("valid");
        let cypher = q.to_cypher().expect("builds");
        assert!(cypher.contains("(f:File)-[:ACCESSED_BY]->(a:App)"));
        assert!(cypher.contains("RETURN f.path, a.name"));
    }

    #[test]
    fn wrong_edge_direction_is_rejected() {
        // ACCESSED_BY is File -> App. Asking App -[ACCESSED_BY]-> File
        // outgoing must fail.
        let q = GraphQuery {
            from: node("a", "App"),
            traverse: vec![TraverseStep {
                edge: "ACCESSED_BY".to_string(),
                direction: Direction::Outgoing,
                to: node("f", "File"),
            }],
            select: vec![field("a", "name")],
            order_by: None,
            limit: 10,
        };
        let err = q.validate(&schema(), &full_scope()).expect_err("reject");
        assert!(matches!(err, DslError::EdgeEndpointMismatch { .. }));
    }

    #[test]
    fn incoming_direction_reverses_the_endpoint_check() {
        // App <-[ACCESSED_BY]- File is the valid incoming form.
        let q = GraphQuery {
            from: node("a", "App"),
            traverse: vec![TraverseStep {
                edge: "ACCESSED_BY".to_string(),
                direction: Direction::Incoming,
                to: node("f", "File"),
            }],
            select: vec![field("a", "name")],
            order_by: None,
            limit: 10,
        };
        q.validate(&schema(), &full_scope()).expect("valid");
        let cypher = q.to_cypher().expect("builds");
        assert!(cypher.contains("(a:App)<-[:ACCESSED_BY]-(f:File)"));
    }

    #[test]
    fn unknown_edge_is_rejected() {
        let q = GraphQuery {
            from: node("f", "File"),
            traverse: vec![TraverseStep {
                edge: "WRITES_TO".to_string(),
                direction: Direction::Outgoing,
                to: node("a", "App"),
            }],
            select: vec![field("f", "path")],
            order_by: None,
            limit: 10,
        };
        let err = q.validate(&schema(), &full_scope()).expect_err("reject");
        assert_eq!(
            err,
            DslError::UnknownEdge {
                edge: "WRITES_TO".to_string()
            }
        );
    }

    #[test]
    fn too_many_traverse_steps_is_rejected() {
        let mut q = simple_file_query();
        q.traverse = (0..MAX_TRAVERSE_STEPS + 1)
            .map(|i| TraverseStep {
                edge: "ACCESSED_BY".to_string(),
                direction: Direction::Outgoing,
                to: node(&format!("n{i}"), "App"),
            })
            .collect();
        let err = q.validate(&schema(), &full_scope()).expect_err("reject");
        assert!(matches!(err, DslError::TooManyTraverseSteps { .. }));
    }

    #[test]
    fn filter_type_mismatch_is_rejected() {
        // last_accessed is Int; filtering it with a Text value fails.
        let mut q = simple_file_query();
        q.from.filters = vec![Filter {
            field: "last_accessed".to_string(),
            op: FilterOp::Eq,
            value: TypedValue::Text("yesterday".to_string()),
        }];
        let err = q.validate(&schema(), &full_scope()).expect_err("reject");
        assert!(matches!(err, DslError::FilterTypeMismatch { .. }));
    }

    #[test]
    fn ordering_operator_on_text_field_is_rejected() {
        // `path` is Text; `Gt` is numeric-only.
        let mut q = simple_file_query();
        q.from.filters = vec![Filter {
            field: "path".to_string(),
            op: FilterOp::Gt,
            value: TypedValue::Text("a".to_string()),
        }];
        let err = q.validate(&schema(), &full_scope()).expect_err("reject");
        assert!(matches!(err, DslError::FilterTypeMismatch { .. }));
    }

    #[test]
    fn valid_filters_build_a_where_clause() {
        let mut q = simple_file_query();
        q.from.filters = vec![Filter {
            field: "last_accessed".to_string(),
            op: FilterOp::Gt,
            value: TypedValue::Int(1_747_000_000),
        }];
        q.validate(&schema(), &full_scope()).expect("valid");
        let cypher = q.to_cypher().expect("builds");
        assert!(cypher.contains("WHERE f.last_accessed > 1747000000"));
    }

    #[test]
    fn string_filter_value_is_quoted_and_escaped() {
        let mut q = simple_file_query();
        q.from.filters = vec![Filter {
            field: "path".to_string(),
            op: FilterOp::Contains,
            // Injection attempt inside the value.
            value: TypedValue::Text("o'clock' RETURN n //".to_string()),
        }];
        q.validate(&schema(), &full_scope()).expect("valid");
        let cypher = q.to_cypher().expect("builds");
        // The single quotes inside the value must be escaped, so the
        // literal stays one token and cannot end the string early.
        assert!(cypher.contains(r"f.path CONTAINS 'o\'clock\' RETURN n //'"));
    }

    #[test]
    fn order_by_builds_clause() {
        let mut q = simple_file_query();
        q.order_by = Some(OrderSpec {
            bind: "f".to_string(),
            field: "last_accessed".to_string(),
            descending: true,
        });
        q.validate(&schema(), &full_scope()).expect("valid");
        let cypher = q.to_cypher().expect("builds");
        assert!(cypher.contains("ORDER BY f.last_accessed DESC"));
    }

    #[test]
    fn order_by_unknown_binding_is_rejected() {
        let mut q = simple_file_query();
        q.order_by = Some(OrderSpec {
            bind: "ghost".to_string(),
            field: "id".to_string(),
            descending: false,
        });
        let err = q.validate(&schema(), &full_scope()).expect_err("reject");
        assert_eq!(
            err,
            DslError::UnknownBinding {
                bind: "ghost".to_string()
            }
        );
    }

    #[test]
    fn limit_is_clamped_into_range() {
        let mut q = simple_file_query();
        q.limit = 0;
        assert!(q.to_cypher().unwrap().trim_end().ends_with("LIMIT 1"));
        q.limit = 99_999;
        assert!(q
            .to_cypher()
            .unwrap()
            .trim_end()
            .ends_with(&format!("LIMIT {MAX_LIMIT}")));
    }

    #[test]
    fn encode_literal_rejects_control_characters() {
        let bad = TypedValue::Text("line\u{0}null".to_string());
        assert_eq!(encode_literal(&bad), Err(BuildError::UnencodableValue));
    }

    #[test]
    fn encode_literal_escapes_known_whitespace() {
        let v = TypedValue::Text("a\nb\tc".to_string());
        assert_eq!(encode_literal(&v).unwrap(), r"'a\nb\tc'");
    }

    #[test]
    fn dsl_json_round_trip() {
        let json = r#"
        {
          "from": { "bind": "f", "label": "File",
                    "filters": [
                      { "field": "last_accessed", "op": "gt", "value": 1747000000 }
                    ] },
          "traverse": [
            { "edge": "ACCESSED_BY", "direction": "outgoing",
              "to": { "bind": "a", "label": "App" } }
          ],
          "select": [
            { "bind": "f", "field": "path" },
            { "bind": "a", "field": "name" }
          ],
          "order_by": { "bind": "f", "field": "last_accessed", "descending": true },
          "limit": 50
        }
        "#;
        let q: GraphQuery = serde_json::from_str(json).expect("parses");
        q.validate(&schema(), &full_scope()).expect("valid");
        let cypher = q.to_cypher().expect("builds");
        assert!(cypher.contains("(f:File)-[:ACCESSED_BY]->(a:App)"));
        assert!(cypher.contains("WHERE f.last_accessed > 1747000000"));
        assert!(cypher.contains("ORDER BY f.last_accessed DESC"));
    }

    #[test]
    fn missing_required_field_fails_to_parse() {
        // `select` is required; omitting it must fail at parse time.
        let json = r#"{ "from": { "bind": "f", "label": "File" }, "limit": 10 }"#;
        assert!(serde_json::from_str::<GraphQuery>(json).is_err());
    }

    #[test]
    fn minimal_tier_permits_nothing() {
        let scope = QueryScope::for_tier(AccessTier::Minimal, &schema());
        let err = simple_file_query()
            .validate(&schema(), &scope)
            .expect_err("minimal tier denies File");
        assert!(matches!(err, DslError::LabelNotInScope { .. }));
    }

    #[test]
    fn project_scoped_tier_denies_sensitive_activity_labels() {
        // A low-privilege (Project-scoped) caller must not be able to
        // read activity / annotation labels. UserAction is the
        // activity log; an AI query over it would expose user
        // behaviour.
        let scope = QueryScope::for_tier(AccessTier::ProjectScoped, &schema());
        let q = GraphQuery {
            from: node("u", "UserAction"),
            traverse: vec![],
            select: vec![field("u", "action")],
            order_by: None,
            limit: 10,
        };
        let err = q.validate(&schema(), &scope).expect_err("must deny");
        assert_eq!(
            err,
            DslError::LabelNotInScope {
                label: "UserAction".to_string()
            }
        );
    }

    #[test]
    fn project_scoped_tier_permits_project_structure() {
        // The same low-privilege tier still allows the File/Project
        // queries the AI needs for "files in this project".
        let scope = QueryScope::for_tier(AccessTier::ProjectScoped, &schema());
        let q = GraphQuery {
            from: node("f", "File"),
            traverse: vec![TraverseStep {
                edge: "FILE_PART_OF".to_string(),
                direction: Direction::Outgoing,
                to: node("p", "Project"),
            }],
            select: vec![field("f", "path"), field("p", "name")],
            order_by: None,
            limit: 10,
        };
        q.validate(&schema(), &scope).expect("project structure allowed");
    }

    #[test]
    fn full_tier_permits_every_label() {
        let scope = QueryScope::for_tier(AccessTier::Full, &schema());
        let q = GraphQuery {
            from: node("u", "UserAction"),
            traverse: vec![],
            select: vec![field("u", "action")],
            order_by: None,
            limit: 10,
        };
        q.validate(&schema(), &scope).expect("full tier allows UserAction");
    }
}
