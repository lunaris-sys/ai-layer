//! Integration test: load the shipped behaviour fixtures from disk and
//! check they parse + validate against the manifest schema. These three
//! `SKILL.md` files are the B0 keystone — the real behaviours the design
//! was dry-run against (`docs/architecture/ai-agent-behaviours-dryrun.md`).
//! If the schema and a real authored behaviour drift apart, this fails.

use std::fs;
use std::path::PathBuf;

use lunaris_ai_agent::behaviour::{parse, BaselineMode, BehaviourKind, Disposition, ReadScope};

fn behaviours_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("behaviours")
}

fn load(name: &str) -> lunaris_ai_agent::behaviour::Behaviour {
    let path = behaviours_dir().join(name).join("SKILL.md");
    let content = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    parse(&content).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

#[test]
fn every_shipped_behaviour_parses_and_validates() {
    let dir = behaviours_dir();
    let mut count = 0;
    for entry in fs::read_dir(&dir).expect("behaviours dir exists") {
        let entry = entry.expect("dir entry");
        if !entry.file_type().expect("file type").is_dir() {
            continue;
        }
        let skill = entry.path().join("SKILL.md");
        let content = fs::read_to_string(&skill)
            .unwrap_or_else(|e| panic!("read {}: {e}", skill.display()));
        parse(&content).unwrap_or_else(|e| panic!("{} failed to validate: {e}", skill.display()));
        count += 1;
    }
    assert!(count >= 3, "expected at least the three shipped behaviours");
}

#[test]
fn auto_tag_is_a_project_scoped_workflow() {
    let b = load("auto-tag-by-project");
    assert_eq!(b.manifest.kind, BehaviourKind::Workflow);
    // It queries Project nodes, so it must declare project-scoped read —
    // session scope cannot see Project labels (G3 / Codex review).
    assert_eq!(b.manifest.reads, ReadScope::Project);
    assert!(b.manifest.handler.is_some(), "a workflow names its handler");
}

#[test]
fn tidy_downloads_is_a_bounded_supervised_agent() {
    let b = load("tidy-downloads");
    assert_eq!(b.manifest.kind, BehaviourKind::Agent);
    assert_eq!(b.manifest.mode, BaselineMode::Supervised);
    assert_eq!(b.manifest.reads, ReadScope::Full);
    let budget = b.manifest.budget.as_ref().expect("an agent is bounded");
    assert!(budget.max_steps > 0 && budget.max_wall_ms > 0);
}

#[test]
fn meeting_prep_is_suggest_only_and_project_scoped() {
    let b = load("meeting-prep");
    assert_eq!(b.manifest.kind, BehaviourKind::Agent);
    assert_eq!(b.manifest.mode, BaselineMode::Suggest);
    assert_eq!(b.manifest.reads, ReadScope::Project);
    // "nothing found" must be silent — the P3 value floor.
    assert_eq!(
        b.manifest.terminal.get("nothing_relevant_found"),
        Some(&Disposition::Silent)
    );
}
