//! Integration test: the router matches a real `file.opened` event against
//! the shipped fixtures loaded through the loader — enabled + trigger-type
//! + filter all considered together.

use std::collections::BTreeMap;
use std::path::PathBuf;

use lunaris_ai_agent::loader::{load, BehaviourSource, Provenance};
use lunaris_ai_agent::router::matching_behaviours;

fn behaviours_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("behaviours")
}

fn fields(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
    pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
}

#[test]
fn routes_a_file_opened_event_to_the_enabled_workflow() {
    let mut enabled = BTreeMap::new();
    enabled.insert("auto-tag-by-project".to_string(), Provenance::BuiltIn);
    let outcome = load(&[BehaviourSource::builtin(behaviours_dir())], &enabled);
    assert!(outcome.errors.is_empty(), "fixtures load: {:?}", outcome.errors);

    // A normal source file matches auto-tag-by-project (enabled, event
    // trigger on file.opened, filter passes).
    let matched = matching_behaviours(
        "file.opened",
        &fields(&[("path", "~/Repositories/lunaris-sys/foo.rs")]),
        &outcome.loaded,
    );
    assert_eq!(matched.len(), 1);
    assert_eq!(matched[0].behaviour.manifest.name, "auto-tag-by-project");

    // A cache path is excluded by the behaviour's filter.
    let cache = matching_behaviours(
        "file.opened",
        &fields(&[("path", "~/.cache/thing")]),
        &outcome.loaded,
    );
    assert!(cache.is_empty(), "cache paths must be filtered out");

    // A different event type matches nothing.
    let other = matching_behaviours("window.focused", &fields(&[]), &outcome.loaded);
    assert!(other.is_empty());
}

#[test]
fn disabled_behaviours_never_match() {
    // Nothing enabled: even a matching event routes to no behaviour.
    let outcome = load(&[BehaviourSource::builtin(behaviours_dir())], &BTreeMap::new());
    let matched = matching_behaviours(
        "file.opened",
        &fields(&[("path", "~/foo.rs")]),
        &outcome.loaded,
    );
    assert!(matched.is_empty());
}
