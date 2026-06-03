//! Trigger router: the deterministic, no-LLM match from an incoming event
//! to the enabled behaviours that should run.
//!
//! This module is the *pure* matching logic — given an event's type and a
//! flat map of its payload fields, decide which enabled behaviours match.
//! Two layers:
//!
//! * **type match** — the behaviour trigger's `event` pattern against the
//!   event type, exact or dot-prefixed (`file.` matches `file.opened`);
//! * **filter** — an optional `<field> <op> <value>` expression evaluated
//!   against the event's payload fields (e.g. `path not_startswith
//!   ~/.cache`), so a behaviour only fires on the events it cares about.
//!
//! Event decoding (the prost `Event` → type + field map), the Event-Bus
//! subscription behind a `TriggerSource` seam, and per-behaviour burst
//! coalescing (design-doc gap G1) all live with the engine loop that
//! *consumes* this matcher; keeping the match itself pure makes it fully
//! unit-testable without any I/O.

use std::collections::BTreeMap;

use thiserror::Error;

use crate::behaviour::{Trigger, TriggerKind};
use crate::loader::LoadedBehaviour;

/// Match an event type against a behaviour trigger's `event` pattern.
///
/// `*` matches everything; a dot-suffixed pattern (`file.`) matches any
/// type under that prefix (`file.opened`, `file.closed`) but not an
/// unrelated `filezilla`; anything else is an exact match.
pub fn type_matches(pattern: &str, event_type: &str) -> bool {
    if pattern == "*" {
        true
    } else if pattern.ends_with('.') {
        event_type.starts_with(pattern)
    } else {
        pattern == event_type
    }
}

/// A comparison in a trigger filter expression.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterOp {
    /// `eq` — field equals value.
    Eq,
    /// `ne` — field does not equal value.
    Ne,
    /// `startswith` — field starts with value.
    StartsWith,
    /// `not_startswith` — field does not start with value.
    NotStartsWith,
    /// `contains` — field contains value.
    Contains,
}

impl FilterOp {
    fn parse(token: &str) -> Option<FilterOp> {
        match token {
            "eq" => Some(FilterOp::Eq),
            "ne" => Some(FilterOp::Ne),
            "startswith" => Some(FilterOp::StartsWith),
            "not_startswith" => Some(FilterOp::NotStartsWith),
            "contains" => Some(FilterOp::Contains),
            _ => None,
        }
    }
}

/// A parsed trigger filter: `<field> <op> <value>`. The value is a single
/// whitespace-delimited token (B1 keeps the grammar minimal; quoted /
/// multi-word values are a later extension).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Filter {
    /// The payload field name to test.
    pub field: String,
    /// The comparison.
    pub op: FilterOp,
    /// The literal value to compare against.
    pub value: String,
}

/// A malformed filter expression. Surfaced at behaviour load time so a bad
/// filter is rejected up front rather than silently never matching.
#[derive(Debug, Error, PartialEq)]
pub enum FilterError {
    /// Not exactly three whitespace-separated tokens.
    #[error("filter must be '<field> <op> <value>', got: {0:?}")]
    Malformed(String),
    /// The operator token is not recognised.
    #[error("unknown filter operator: {0:?}")]
    UnknownOp(String),
}

impl Filter {
    /// Parse a `<field> <op> <value>` expression.
    pub fn parse(expr: &str) -> Result<Filter, FilterError> {
        let tokens: Vec<&str> = expr.split_whitespace().collect();
        if tokens.len() != 3 {
            return Err(FilterError::Malformed(expr.to_string()));
        }
        let op = FilterOp::parse(tokens[1]).ok_or_else(|| FilterError::UnknownOp(tokens[1].to_string()))?;
        Ok(Filter {
            field: tokens[0].to_string(),
            op,
            value: tokens[2].to_string(),
        })
    }

    /// Evaluate against an event's payload fields.
    ///
    /// A **missing field never matches**, for *every* operator (including
    /// negations) — fail-closed. A filter is a guard that requires the
    /// named field; if a malformed or schema-drifted event lacks it, the
    /// behaviour does not fire rather than firing on incomplete data (so
    /// `path not_startswith ~/.cache` does not match an event with no
    /// `path`). Matching "field is absent" would need an explicit
    /// `exists`/`missing` operator, which the grammar does not yet have.
    pub fn eval(&self, fields: &BTreeMap<String, String>) -> bool {
        let Some(actual) = fields.get(&self.field) else {
            return false;
        };
        match self.op {
            FilterOp::Eq => actual == &self.value,
            FilterOp::Ne => actual != &self.value,
            FilterOp::StartsWith => actual.starts_with(&self.value),
            FilterOp::NotStartsWith => !actual.starts_with(&self.value),
            FilterOp::Contains => actual.contains(&self.value),
        }
    }
}

/// Whether a behaviour trigger matches an event. Schedule and manual
/// triggers are never matched by an event (they fire on a clock / explicit
/// invocation), so only event triggers can match here.
///
/// Returns `Err` only if the filter expression is malformed; callers that
/// loaded a validated behaviour will not see this (the loader rejects a
/// behaviour whose filter does not parse).
pub fn trigger_matches(
    trigger: &Trigger,
    event_type: &str,
    fields: &BTreeMap<String, String>,
) -> Result<bool, FilterError> {
    match trigger.kind {
        TriggerKind::Event => {
            let pattern = trigger.event.as_deref().unwrap_or("");
            if !type_matches(pattern, event_type) {
                return Ok(false);
            }
            match &trigger.filter {
                Some(expr) => Ok(Filter::parse(expr)?.eval(fields)),
                None => Ok(true),
            }
        }
        TriggerKind::Schedule | TriggerKind::Manual => Ok(false),
    }
}

/// The deterministic router match: every **enabled** behaviour whose
/// trigger matches the event. A behaviour with a malformed filter
/// fail-closes (does not match) rather than affecting the others; the
/// loader rejects such behaviours up front, so this is a backstop.
pub fn matching_behaviours<'a>(
    event_type: &str,
    fields: &BTreeMap<String, String>,
    behaviours: &'a [LoadedBehaviour],
) -> Vec<&'a LoadedBehaviour> {
    behaviours
        .iter()
        .filter(|lb| {
            lb.status.is_enabled()
                && trigger_matches(&lb.behaviour.manifest.trigger, event_type, fields)
                    .unwrap_or(false)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fields(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
    }

    #[test]
    fn type_match_exact_prefix_and_wildcard() {
        assert!(type_matches("file.opened", "file.opened"));
        assert!(!type_matches("file.opened", "file.closed"));
        assert!(type_matches("file.", "file.opened"));
        assert!(type_matches("file.", "file.closed"));
        assert!(!type_matches("file.", "filezilla")); // prefix keeps the dot
        assert!(type_matches("*", "anything.at.all"));
    }

    #[test]
    fn filter_parse_and_eval() {
        let f = Filter::parse("path not_startswith ~/.cache").expect("valid");
        assert_eq!(f.field, "path");
        assert_eq!(f.op, FilterOp::NotStartsWith);
        assert!(f.eval(&fields(&[("path", "~/Repositories/foo.rs")])));
        assert!(!f.eval(&fields(&[("path", "~/.cache/x")])));
        // A missing field never matches, even for a negation (fail-closed).
        assert!(!f.eval(&fields(&[])));
        assert!(!f.eval(&fields(&[("app_id", "x")]))); // wrong field present

        assert!(Filter::parse("app_id eq org.lunaris.files")
            .unwrap()
            .eval(&fields(&[("app_id", "org.lunaris.files")])));
    }

    #[test]
    fn filter_parse_rejects_malformed() {
        assert_eq!(
            Filter::parse("path startswith"),
            Err(FilterError::Malformed("path startswith".to_string()))
        );
        assert!(matches!(
            Filter::parse("path wat value"),
            Err(FilterError::UnknownOp(_))
        ));
    }

    #[test]
    fn schedule_and_manual_triggers_do_not_match_events() {
        let schedule = Trigger {
            kind: TriggerKind::Schedule,
            event: None,
            filter: None,
            every_secs: Some(60),
        };
        assert_eq!(trigger_matches(&schedule, "file.opened", &fields(&[])), Ok(false));
        let manual = Trigger {
            kind: TriggerKind::Manual,
            event: None,
            filter: None,
            every_secs: None,
        };
        assert_eq!(trigger_matches(&manual, "file.opened", &fields(&[])), Ok(false));
    }

    #[test]
    fn event_trigger_matches_type_then_filter() {
        let trigger = Trigger {
            kind: TriggerKind::Event,
            event: Some("file.opened".to_string()),
            filter: Some("path not_startswith ~/.cache".to_string()),
            every_secs: None,
        };
        assert_eq!(
            trigger_matches(&trigger, "file.opened", &fields(&[("path", "~/foo.rs")])),
            Ok(true)
        );
        // Filter excludes cache paths.
        assert_eq!(
            trigger_matches(&trigger, "file.opened", &fields(&[("path", "~/.cache/x")])),
            Ok(false)
        );
        // Wrong type never reaches the filter.
        assert_eq!(
            trigger_matches(&trigger, "window.focused", &fields(&[("path", "~/foo.rs")])),
            Ok(false)
        );
    }
}
