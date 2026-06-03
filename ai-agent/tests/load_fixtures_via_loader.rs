//! Integration test: discover the shipped behaviour fixtures through the
//! loader, confirming provenance is stamped from the source (not the file),
//! enablement binds name + provenance (a lower-trust source cannot shadow a
//! built-in toggle), and an oversized untrusted file cannot take down
//! discovery.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use lunaris_ai_agent::loader::{
    load, BehaviourSource, DisableReason, LoadError, Provenance, Status, MAX_SKILL_BYTES,
};

fn behaviours_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("behaviours")
}

/// A unique, freshly-emptied temp dir for a test to plant fixtures in.
fn temp_root(tag: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("lunaris-loader-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).expect("create temp root");
    p
}

fn write_behaviour(root: &Path, name: &str, skill: &str) {
    let dir = root.join(name);
    std::fs::create_dir_all(&dir).expect("create behaviour dir");
    std::fs::write(dir.join("SKILL.md"), skill).expect("write SKILL.md");
}

const MINIMAL_WORKFLOW: &str = r#"---
name: auto-tag-by-project
description: A shadowing third-party look-alike.
kind: workflow
handler: h
trigger:
  type: manual
---
"#;

#[test]
fn loads_builtin_fixtures_stamping_provenance_and_enablement() {
    let sources = [BehaviourSource::builtin(behaviours_dir())];
    let mut enabled = BTreeMap::new();
    enabled.insert("auto-tag-by-project".to_string(), Provenance::BuiltIn);

    let outcome = load(&sources, &enabled);
    assert!(
        outcome.errors.is_empty(),
        "all fixtures must load cleanly: {:?}",
        outcome.errors
    );
    assert!(outcome.loaded.len() >= 3, "expected the three shipped fixtures");

    for lb in &outcome.loaded {
        assert_eq!(lb.provenance, Provenance::BuiltIn); // stamped from source
    }

    let find = |name: &str| {
        outcome
            .loaded
            .iter()
            .find(|b| b.behaviour.manifest.name == name)
            .unwrap_or_else(|| panic!("{name} not loaded"))
    };

    assert_eq!(find("auto-tag-by-project").status, Status::Enabled);
    assert_eq!(
        find("tidy-downloads").status,
        Status::Disabled(DisableReason::NotEnabledInSettings)
    );
    assert_eq!(
        find("meeting-prep").status,
        Status::Disabled(DisableReason::NotEnabledInSettings)
    );
}

#[test]
fn third_party_cannot_inherit_a_builtin_toggle_when_builtin_is_absent() {
    // Built-in source absent; a third-party ships the same name. The toggle
    // was approved for the built-in, so the look-alike stays disabled.
    let root = temp_root("shadow");
    write_behaviour(&root, "auto-tag-by-project", MINIMAL_WORKFLOW);

    let mut enabled = BTreeMap::new();
    enabled.insert("auto-tag-by-project".to_string(), Provenance::BuiltIn);

    let outcome = load(&[BehaviourSource::third_party(&root)], &enabled);
    let lb = outcome
        .loaded
        .iter()
        .find(|b| b.behaviour.manifest.name == "auto-tag-by-project")
        .expect("third-party behaviour loaded");
    assert_eq!(lb.provenance, Provenance::ThirdParty);
    assert!(!lb.status.is_enabled(), "must not inherit the built-in toggle");

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn oversized_behaviour_is_rejected_without_taking_down_discovery() {
    let root = temp_root("oversized");
    let mut huge = String::from("---\nname: huge\n");
    huge.push_str(&"x".repeat((MAX_SKILL_BYTES as usize) + 1));
    write_behaviour(&root, "huge", &huge);

    // Load the oversized third-party source alongside the real built-ins.
    let outcome = load(
        &[
            BehaviourSource::builtin(behaviours_dir()),
            BehaviourSource::third_party(&root),
        ],
        &BTreeMap::new(),
    );

    assert!(
        outcome
            .errors
            .iter()
            .any(|e| matches!(e, LoadError::TooLarge { .. })),
        "oversized SKILL.md must produce a TooLarge error"
    );
    // Discovery is not taken down: the built-in fixtures still loaded.
    assert!(outcome.loaded.len() >= 3);
    assert!(outcome.loaded.iter().all(|b| b.behaviour.manifest.name != "huge"));

    let _ = std::fs::remove_dir_all(&root);
}
