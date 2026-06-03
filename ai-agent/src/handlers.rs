//! Built-in workflow handlers (the code behind `kind: workflow` behaviours).

use crate::engine::{HandlerError, HandlerOutcome, HandlerRegistry, WorkflowHandler};
use crate::gate::ProposedAction;
use crate::seams::{AgentEvent, GraphHandle};

/// The registry of built-in workflow handlers, keyed by the manifest
/// `handler` id. The daemon registers these; third-party handlers are a
/// later, separately-trusted mechanism.
pub fn builtin_handlers() -> HandlerRegistry {
    let mut registry = HandlerRegistry::new();
    registry.insert(
        "auto_tag_by_project".to_string(),
        Box::new(AutoTagByProject) as Box<dyn WorkflowHandler>,
    );
    registry
}

/// `auto-tag-by-project`: tag a newly opened file with the project it
/// belongs to, resolved as the **most specific** project whose root path is
/// a (component-aware) prefix of the file path. If two projects are equally
/// specific the file is genuinely ambiguous, so the behaviour does not guess
/// (design-doc gap G2); it reaches a terminal condition instead.
pub struct AutoTagByProject;

#[async_trait::async_trait]
impl WorkflowHandler for AutoTagByProject {
    async fn run(
        &self,
        event: &AgentEvent,
        graph: &dyn GraphHandle,
    ) -> Result<HandlerOutcome, HandlerError> {
        let Some(path) = event.fields.get("path") else {
            // The event trigger filters on `path`, so this is unreachable in
            // practice; treated as no-op rather than an error.
            return Ok(HandlerOutcome::Terminal("no_path".to_string()));
        };

        let rows = graph
            .query("MATCH (p:Project) RETURN p.id AS id, p.root_path AS root_path")
            .await
            .map_err(|e| HandlerError(e.to_string()))?;

        // Projects whose root is a component-aware prefix of the path, with
        // the prefix length (longer = more specific).
        let mut matches: Vec<(usize, &str)> = rows
            .iter()
            .filter_map(|row| {
                let id = row.get("id")?.as_str()?;
                let root = row.get("root_path")?.as_str()?;
                path_within(path, root).then_some((root.len(), id))
            })
            .collect();

        let Some(max_len) = matches.iter().map(|(len, _)| *len).max() else {
            return Ok(HandlerOutcome::Terminal("no_matching_project".to_string()));
        };
        matches.retain(|(len, _)| *len == max_len);

        match matches.as_slice() {
            [(_, id)] => Ok(HandlerOutcome::Propose(ProposedAction {
                tool: "graph.write".to_string(),
                summary: format!("Tag {path} as part of project {id}"),
            })),
            // Equally-specific candidates: ambiguous, do not guess (G2).
            _ => Ok(HandlerOutcome::Terminal("ambiguous_project".to_string())),
        }
    }
}

/// Whether `path` lies within the directory `root`, respecting component
/// boundaries: `root` itself or any descendant, but not a sibling whose name
/// merely starts with `root` (e.g. `/a/lib` does not contain `/a/library`).
fn path_within(path: &str, root: &str) -> bool {
    if root.is_empty() {
        return false;
    }
    let root = root.strip_suffix('/').unwrap_or(root);
    path == root || path.strip_prefix(root).is_some_and(|rest| rest.starts_with('/'))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeMap, HashMap};

    use crate::seams::GraphError;

    /// A graph returning canned project rows (or an error).
    struct FakeGraph(Result<Vec<HashMap<String, serde_json::Value>>, ()>);

    #[async_trait::async_trait]
    impl GraphHandle for FakeGraph {
        async fn query(
            &self,
            _cypher: &str,
        ) -> Result<Vec<HashMap<String, serde_json::Value>>, GraphError> {
            self.0
                .clone()
                .map_err(|_| GraphError::Failed("boom".to_string()))
        }
    }

    fn projects(pairs: &[(&str, &str)]) -> FakeGraph {
        let rows = pairs
            .iter()
            .map(|(id, root)| {
                HashMap::from([
                    ("id".to_string(), serde_json::Value::from(*id)),
                    ("root_path".to_string(), serde_json::Value::from(*root)),
                ])
            })
            .collect();
        FakeGraph(Ok(rows))
    }

    fn opened(path: &str) -> AgentEvent {
        AgentEvent {
            id: "e1".to_string(),
            event_type: "file.opened".to_string(),
            fields: BTreeMap::from([("path".to_string(), path.to_string())]),
            external_content: false,
        }
    }

    async fn run(graph: &FakeGraph, path: &str) -> HandlerOutcome {
        AutoTagByProject.run(&opened(path), graph).await.unwrap()
    }

    #[test]
    fn path_within_respects_component_boundaries() {
        assert!(path_within("/a/proj/foo.rs", "/a/proj"));
        assert!(path_within("/a/proj", "/a/proj")); // the root itself
        assert!(path_within("/a/proj/foo.rs", "/a/proj/")); // trailing slash on root
        assert!(!path_within("/a/project/foo.rs", "/a/proj")); // sibling prefix, not contained
        assert!(!path_within("/b/foo.rs", "/a/proj"));
    }

    #[tokio::test]
    async fn proposes_the_matching_project() {
        let g = projects(&[("proj-a", "~/Repositories/lunaris-sys"), ("proj-b", "~/Other")]);
        match run(&g, "~/Repositories/lunaris-sys/foo.rs").await {
            HandlerOutcome::Propose(action) => {
                assert_eq!(action.tool, "graph.write");
                assert!(action.summary.contains("proj-a"));
            }
            other => panic!("expected a proposal, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn most_specific_nested_project_wins() {
        let g = projects(&[
            ("outer", "~/Repositories/lunaris-sys"),
            ("inner", "~/Repositories/lunaris-sys/desktop-shell"),
        ]);
        match run(&g, "~/Repositories/lunaris-sys/desktop-shell/src/x.rs").await {
            HandlerOutcome::Propose(action) => assert!(action.summary.contains("inner")),
            other => panic!("expected the inner project, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn no_match_and_ambiguous_reach_terminals() {
        let none = projects(&[("proj-a", "~/Repositories/lunaris-sys")]);
        assert!(matches!(
            run(&none, "~/Downloads/x.pdf").await,
            HandlerOutcome::Terminal(t) if t == "no_matching_project"
        ));

        // Two projects claiming the same root: ambiguous, do not guess.
        let tie = projects(&[("a", "~/shared"), ("b", "~/shared")]);
        assert!(matches!(
            run(&tie, "~/shared/x.rs").await,
            HandlerOutcome::Terminal(t) if t == "ambiguous_project"
        ));
    }

    #[tokio::test]
    async fn a_graph_error_propagates_as_handler_error() {
        let g = FakeGraph(Err(()));
        let err = AutoTagByProject
            .run(&opened("~/Repositories/lunaris-sys/foo.rs"), &g)
            .await
            .unwrap_err();
        assert!(format!("{err}").contains("boom"));
    }
}
