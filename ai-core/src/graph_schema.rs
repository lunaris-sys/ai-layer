//! Static description of the Lunaris Knowledge Graph schema.
//!
//! The structured query DSL (`graph_query`) validates every label,
//! edge, and field reference against this schema before a single
//! character of Cypher is built. It is also the schema-grounding the
//! AI sees in the query-generation prompt.
//!
//! ## Sync point
//!
//! This is a hand-maintained mirror of the `CREATE NODE TABLE` /
//! `CREATE REL TABLE` statements in `knowledge/src/graph.rs`. When the
//! Knowledge Graph schema changes there, this file must be updated to
//! match. Phase 9-γ should replace the hardcoded tables with a
//! dynamic load from the Knowledge Daemon (which owns the Foundation
//! §3 schema registry), removing the sync burden. Until then the
//! mismatch risk is a documented, accepted limitation.

/// Property type of a graph node field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldType {
    /// `STRING` column.
    Text,
    /// `INT64` column. Timestamps are stored as INT64 epoch values.
    Int,
    /// `BOOL` column.
    Bool,
}

/// One node table in the Knowledge Graph schema.
#[derive(Debug, Clone, Copy)]
pub struct NodeSchema {
    /// Node label, e.g. `File`.
    pub label: &'static str,
    /// Property columns and their types.
    pub fields: &'static [(&'static str, FieldType)],
}

/// One relationship table.
#[derive(Debug, Clone, Copy)]
pub struct EdgeSchema {
    /// Relationship type, e.g. `ACCESSED_BY`.
    pub label: &'static str,
    /// Source node label.
    pub from: &'static str,
    /// Destination node label.
    pub to: &'static str,
}

/// The Knowledge Graph schema: a set of node tables and edge tables.
#[derive(Debug, Clone, Copy)]
pub struct GraphSchema {
    nodes: &'static [NodeSchema],
    edges: &'static [EdgeSchema],
}

impl GraphSchema {
    /// The canonical Lunaris Knowledge Graph schema.
    pub fn knowledge_graph() -> Self {
        Self {
            nodes: NODES,
            edges: EDGES,
        }
    }

    /// Look up a node schema by label.
    pub fn node(&self, label: &str) -> Option<&'static NodeSchema> {
        self.nodes.iter().find(|n| n.label == label)
    }

    /// Look up an edge schema by relationship type.
    pub fn edge(&self, label: &str) -> Option<&'static EdgeSchema> {
        self.edges.iter().find(|e| e.label == label)
    }

    /// Resolve the type of a field on a node label. Returns `None`
    /// if either the label or the field is unknown.
    pub fn field_type(&self, label: &str, field: &str) -> Option<FieldType> {
        self.node(label)?
            .fields
            .iter()
            .find(|(name, _)| *name == field)
            .map(|(_, ty)| *ty)
    }

    /// All node labels in the schema.
    pub fn node_labels(&self) -> impl Iterator<Item = &'static str> {
        self.nodes.iter().map(|n| n.label)
    }

    /// All edge labels in the schema.
    pub fn edge_labels(&self) -> impl Iterator<Item = &'static str> {
        self.edges.iter().map(|e| e.label)
    }
}

/// Node tables. Mirrors `knowledge/src/graph.rs`.
const NODES: &[NodeSchema] = &[
    NodeSchema {
        label: "File",
        fields: &[
            ("id", FieldType::Text),
            ("path", FieldType::Text),
            ("app_id", FieldType::Text),
            ("last_accessed", FieldType::Int),
        ],
    },
    NodeSchema {
        label: "App",
        fields: &[("id", FieldType::Text), ("name", FieldType::Text)],
    },
    NodeSchema {
        label: "Session",
        fields: &[("id", FieldType::Text), ("started_at", FieldType::Int)],
    },
    NodeSchema {
        label: "Event",
        fields: &[
            ("id", FieldType::Text),
            ("type", FieldType::Text),
            ("timestamp", FieldType::Int),
            ("source", FieldType::Text),
        ],
    },
    NodeSchema {
        label: "UserAction",
        fields: &[
            ("id", FieldType::Text),
            ("category", FieldType::Text),
            ("action", FieldType::Text),
            ("subject", FieldType::Text),
            ("timestamp", FieldType::Int),
        ],
    },
    NodeSchema {
        label: "Project",
        fields: &[
            ("id", FieldType::Text),
            ("name", FieldType::Text),
            ("description", FieldType::Text),
            ("root_path", FieldType::Text),
            ("accent_color", FieldType::Text),
            ("icon", FieldType::Text),
            ("status", FieldType::Text),
            ("created_at", FieldType::Int),
            ("last_accessed", FieldType::Int),
            ("inferred", FieldType::Bool),
            ("confidence", FieldType::Int),
            ("promoted", FieldType::Bool),
            ("archived_at", FieldType::Int),
        ],
    },
    NodeSchema {
        label: "Directory",
        fields: &[
            ("id", FieldType::Text),
            ("path", FieldType::Text),
            ("name", FieldType::Text),
            ("project_id", FieldType::Text),
            ("created_at", FieldType::Int),
        ],
    },
    NodeSchema {
        label: "Annotation",
        fields: &[
            ("id", FieldType::Text),
            ("namespace", FieldType::Text),
            ("target_type", FieldType::Text),
            ("target_id", FieldType::Text),
            ("data", FieldType::Text),
            ("created_at", FieldType::Int),
            ("last_modified", FieldType::Int),
        ],
    },
    NodeSchema {
        label: "Summary",
        fields: &[
            ("id", FieldType::Text),
            ("type", FieldType::Text),
            ("app_id", FieldType::Text),
            ("access_count", FieldType::Int),
            ("primary_application", FieldType::Text),
            ("active_period_start", FieldType::Int),
            ("active_period_end", FieldType::Int),
        ],
    },
    NodeSchema {
        label: "PinnedMarker",
        fields: &[
            ("id", FieldType::Text),
            ("node_id", FieldType::Text),
            ("node_type", FieldType::Text),
            ("pinned_at", FieldType::Int),
        ],
    },
];

/// Relationship tables. Mirrors `knowledge/src/graph.rs`.
const EDGES: &[EdgeSchema] = &[
    EdgeSchema {
        label: "ACCESSED_BY",
        from: "File",
        to: "App",
    },
    EdgeSchema {
        label: "ACTIVE_IN",
        from: "App",
        to: "Session",
    },
    EdgeSchema {
        label: "EMITTED_BY",
        from: "Event",
        to: "App",
    },
    EdgeSchema {
        label: "DERIVED_FROM",
        from: "UserAction",
        to: "Event",
    },
    EdgeSchema {
        label: "FILE_PART_OF",
        from: "File",
        to: "Project",
    },
    EdgeSchema {
        label: "DIR_PART_OF",
        from: "Directory",
        to: "Project",
    },
    EdgeSchema {
        label: "SUMMARIZES",
        from: "Summary",
        to: "App",
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_node_lookup_succeeds() {
        let s = GraphSchema::knowledge_graph();
        assert!(s.node("File").is_some());
        assert!(s.node("Project").is_some());
        assert!(s.node("Annotation").is_some());
    }

    #[test]
    fn unknown_node_lookup_returns_none() {
        let s = GraphSchema::knowledge_graph();
        assert!(s.node("Secret").is_none());
        assert!(s.node("file").is_none(), "lookup is case-sensitive");
    }

    #[test]
    fn field_type_resolves_correctly() {
        let s = GraphSchema::knowledge_graph();
        assert_eq!(s.field_type("File", "path"), Some(FieldType::Text));
        assert_eq!(s.field_type("File", "last_accessed"), Some(FieldType::Int));
        assert_eq!(s.field_type("Project", "inferred"), Some(FieldType::Bool));
        assert_eq!(s.field_type("Project", "confidence"), Some(FieldType::Int));
    }

    #[test]
    fn unknown_field_returns_none() {
        let s = GraphSchema::knowledge_graph();
        assert_eq!(s.field_type("File", "secret_column"), None);
        assert_eq!(s.field_type("Nonexistent", "path"), None);
    }

    #[test]
    fn edge_lookup_returns_endpoints() {
        let s = GraphSchema::knowledge_graph();
        let e = s.edge("ACCESSED_BY").expect("known edge");
        assert_eq!(e.from, "File");
        assert_eq!(e.to, "App");
        let e = s.edge("FILE_PART_OF").expect("known edge");
        assert_eq!(e.from, "File");
        assert_eq!(e.to, "Project");
    }

    #[test]
    fn unknown_edge_returns_none() {
        let s = GraphSchema::knowledge_graph();
        assert!(s.edge("WRITES_TO").is_none());
    }

    #[test]
    fn schema_has_expected_table_counts() {
        let s = GraphSchema::knowledge_graph();
        assert_eq!(s.node_labels().count(), 10);
        assert_eq!(s.edge_labels().count(), 7);
    }
}
