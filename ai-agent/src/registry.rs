//! The trusted given-rule registry: the only source of action schemas the
//! predict-before-act path may prove anything about.
//!
//! A raw [`ActionSchema`] is forgeable, its `action` and `provenance` are just
//! fields, so the world-model interpreter must never be driven from one that
//! came from a behaviour or the model. This module holds the built-in
//! given rules and hands them out wrapped in a [`TrustedActionSchema`] whose
//! only constructor is private here. Code elsewhere can obtain one solely
//! through [`lookup`], keyed by the invoked tool/action id, so the type itself
//! is the proof that a schema was registry-resolved.
//!
//! Only `Provenance::Given` rules live here. Learned rules are induced,
//! approved, and admitted through a separate (later) path; `lookup` never
//! returns one.
//!
//! Until the gate path calls [`lookup`], it has no non-test caller, so the
//! module allows dead code; the allowance goes away once the gate resolves a
//! schema here.
#![allow(dead_code)]

use crate::world::{compensation_of, ActionSchema, Effect, Predicate, Provenance};

/// An [`ActionSchema`] the registry vouches for. Its single field is private
/// and it has no public constructor, so it can only be produced by [`lookup`]
/// in this module, never forged from untrusted input elsewhere in the crate.
pub(crate) struct TrustedActionSchema {
    schema: ActionSchema,
}

impl TrustedActionSchema {
    /// The wrapped schema, for the slice builder and interpreter to read. The
    /// accessor is crate-internal so reading the schema never lets other code
    /// reconstruct the trust token from it.
    pub(crate) fn schema(&self) -> &ActionSchema {
        &self.schema
    }

    /// Whether this action is reversible (Foundation B1): its effect sequence
    /// has a derivable compensation in the same world-model DSL. This grounds
    /// the gate's "reversible" predicate, which was previously assumed: an
    /// action with no compensation is irreversible, so it is high-impact and
    /// must always be confirmed (never lifted to autonomous preview). Reversible
    /// is defined conservatively (see [`compensation_of`]): an effect that needs
    /// prior state to undo (a field set, a node removal) is not auto-invertible,
    /// so a schema containing one is irreversible unless it later declares an
    /// explicit compensation.
    pub(crate) fn is_reversible(&self) -> bool {
        compensation_of(&self.schema.effects).is_some()
    }
}

/// Resolve the given-rule schema for an invoked action/tool id, or `None` if
/// no rule is registered. With no rule the predict-before-act path cannot
/// prove the action, so the gate keeps its conservative cap rather than lift
/// it.
pub(crate) fn lookup(action_id: &str) -> Option<TrustedActionSchema> {
    let schema = match action_id {
        "graph.write" => graph_write_link_schema(),
        // A registered but irreversible action, for the gate's
        // irreversible-always-confirms tests (a `SetField` cannot be inverted
        // from itself, so the schema has no derivable compensation).
        #[cfg(test)]
        "test.irreversible" => ActionSchema {
            action: "test.irreversible".to_string(),
            preconditions: vec![],
            effects: vec![Effect::SetField {
                bind: "x".to_string(),
                field: "f".to_string(),
                value: "v".to_string(),
            }],
            provenance: Provenance::Given,
        },
        _ => return None,
    };
    Some(TrustedActionSchema { schema })
}

/// The given rule for linking a file to the project it belongs to
/// (`FILE_PART_OF`). It proves the real invariant before the link may be
/// asserted: both nodes exist, the file's path lies under the project's root
/// (so an unrelated file/project pair cannot be linked), and the edge is not
/// already present. It creates a single edge, no node, so the bounded slice
/// can represent its full effect.
fn graph_write_link_schema() -> ActionSchema {
    ActionSchema {
        action: "graph.write".to_string(),
        preconditions: vec![
            Predicate::NodeExists {
                bind: "file".to_string(),
                label: "File".to_string(),
            },
            Predicate::NodeExists {
                bind: "project".to_string(),
                label: "Project".to_string(),
            },
            // The file must actually belong to the project: its path lies
            // under the project root. Without this the rule would prove only
            // that two unrelated nodes exist and authorise a corrupt link.
            Predicate::PathUnderField {
                inner: "file".to_string(),
                inner_field: "path".to_string(),
                outer: "project".to_string(),
                outer_field: "root_path".to_string(),
            },
            Predicate::Not(Box::new(Predicate::EdgeExists {
                from: "file".to_string(),
                edge: "FILE_PART_OF".to_string(),
                to: "project".to_string(),
            })),
        ],
        effects: vec![Effect::AssertEdge {
            from: "file".to_string(),
            edge: "FILE_PART_OF".to_string(),
            to: "project".to_string(),
        }],
        provenance: Provenance::Given,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_a_given_rule_for_a_known_action() {
        let trusted = lookup("graph.write").expect("graph.write is registered");
        assert_eq!(trusted.schema().action, "graph.write");
        // The registry only ever vouches for given rules.
        assert!(matches!(trusted.schema().provenance, Provenance::Given));
    }

    #[test]
    fn an_unknown_action_has_no_rule() {
        assert!(lookup("fs.delete").is_none());
        assert!(lookup("").is_none());
    }

    #[test]
    fn reversibility_is_grounded_in_derivable_compensation() {
        // The built-in link rule asserts one edge, which inverts cleanly, so it
        // is reversible (and may be lifted to autonomous preview when proven).
        assert!(lookup("graph.write").unwrap().is_reversible());
        // A schema whose effect cannot be inverted from itself alone (a field
        // set needs the prior value) is irreversible: the gate must always
        // confirm it.
        let irreversible = TrustedActionSchema {
            schema: ActionSchema {
                action: "x".to_string(),
                preconditions: vec![],
                effects: vec![Effect::SetField {
                    bind: "a".to_string(),
                    field: "f".to_string(),
                    value: "v".to_string(),
                }],
                provenance: Provenance::Given,
            },
        };
        assert!(!irreversible.is_reversible());
    }

    #[test]
    fn the_built_in_rule_creates_no_node() {
        // Node-level mutations are refused by the slice builder, so a given
        // rule must not contain one or it could never be sliced.
        let trusted = lookup("graph.write").unwrap();
        assert!(!trusted
            .schema()
            .effects
            .iter()
            .any(|e| matches!(e, Effect::AssertNode { .. } | Effect::RetractNode { .. })));
    }
}
