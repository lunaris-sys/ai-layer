//! Behaviour manifests: the declarative unit the agent dispatches.
//!
//! A behaviour is an **Agent Skills** directory — a `SKILL.md` file with a
//! YAML frontmatter block (delimited by `---`) followed by a markdown
//! body, plus optional bundled files. The frontmatter is the
//! machine-readable manifest the trigger router and the execution engine
//! act on; the body is the goal/instructions (for an `agent` behaviour) or
//! human-readable notes (for a `workflow` behaviour). See
//! `docs/architecture/ai-agent-design.md` §10.
//!
//! **Why YAML, not Lunaris' usual TOML.** Behaviours deliberately follow
//! the Agent Skills open standard (the same `SKILL.md` + YAML frontmatter
//! that Claude, pi, Warp and goose load from `.agents/skills/`). That
//! portability is the whole point of adopting the standard: a Lunaris
//! behaviour is readable as a skill by standard tooling (it ignores our
//! extra frontmatter keys), and a foreign skill's `name`/`description`/
//! body parse here. A Lunaris-private TOML frontmatter would break that —
//! no standard parser could read it — so this one file format departs from
//! the "TOML everywhere" config convention on purpose.
//!
//! Progressive disclosure: a loader preloads only `name`, `description`,
//! and `trigger` for routing; the full manifest + body are read on
//! dispatch. This module is the parse/validate layer that turns the
//! on-disk text into a typed [`Behaviour`]; loading from directories and
//! routing live elsewhere.
//!
//! Validation is fail-closed: a manifest that cannot be parsed or that
//! violates an invariant (an `agent` behaviour with no budget, a
//! `workflow` behaviour with no handler) is rejected, never loaded with a
//! guessed default — an unbounded or under-specified autonomous behaviour
//! is exactly what must not run.
//!
//! **What this layer does *not* enforce (by design — it is the contract,
//! not the engine).** The manifest *declares* intent; downstream phases
//! enforce it: the action mode is a ceiling combined with trusted Settings
//! by the gate (B1); tool-scope honouring and rejection of empty scopes on
//! sensitive tools is the tool registry / gate (B1); and the safety of a
//! state-changing tool like `fs.move` (destination-empty preconditions,
//! batch snapshot/undo, overwrite confirmation) is a property of the
//! world-model **action schema** (B2), enforced centrally per action for
//! every behaviour — never per-manifest prose or a bypassable per-manifest
//! flag. The behaviour fixtures under `behaviours/` are examples + parser
//! test data, not enabled/dispatched code.

use std::collections::BTreeMap;

pub use lunaris_ai_core::capability::{AccessTier, BaselineMode};
use serde::Deserialize;
use thiserror::Error;

/// Where a behaviour sits on the workflow↔agent spectrum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BehaviourKind {
    /// Developer-defined control flow (a code handler); little or no LLM.
    Workflow,
    /// A bounded agentic loop (LLM plans→acts→observes within a budget).
    Agent,
}

/// The minimum Knowledge-Graph read scope a behaviour needs to function.
///
/// Read access is a single *global* level (Foundation §8.4.6); a behaviour
/// declares the minimum it requires here. If the user's global read level
/// is lower, the behaviour is disabled with an explanation rather than
/// silently under-reading (design-doc gap G3). The names are the short
/// manifest vocabulary; [`ReadScope::tier`] maps to the canonical
/// [`AccessTier`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReadScope {
    /// Only content the user pasted in; no graph access (tier 0).
    Minimal,
    /// Current-session activity (tier 1).
    Session,
    /// Project structure — the Focus-Mode default (tier 2).
    Project,
    /// Time-windowed activity (tier 3).
    Time,
    /// Full read access (tier 4).
    Full,
}

impl ReadScope {
    /// Map to the canonical capability tier.
    pub fn tier(self) -> AccessTier {
        match self {
            ReadScope::Minimal => AccessTier::Minimal,
            ReadScope::Session => AccessTier::SessionScoped,
            ReadScope::Project => AccessTier::ProjectScoped,
            ReadScope::Time => AccessTier::TimeScoped,
            ReadScope::Full => AccessTier::Full,
        }
    }
}

/// How the outcome a behaviour terminates on is surfaced to the user
/// (design-doc gap F9). Attached per terminal condition so "nothing
/// found" can be silent (honouring the P3 value floor) while a useful
/// result pushes a notification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Disposition {
    /// Surface immediately (a notification), subject to DND/Focus/timing.
    Push,
    /// Hold for the next natural moment (Waypointer Suggestions).
    Queue,
    /// Write to the graph silently; surfaced only if the user asks.
    Store,
    /// Do not surface at all (e.g. "nothing relevant found").
    Silent,
}

/// What kind of thing dispatches a behaviour.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TriggerKind {
    /// An Event Bus event (see [`Trigger::event`] / [`Trigger::filter`]).
    Event,
    /// A periodic schedule (see [`Trigger::every_secs`]).
    Schedule,
    /// Only ever run when explicitly invoked (Waypointer / D-Bus).
    Manual,
}

/// What causes a behaviour to be dispatched.
///
/// A flat record (`{type, event?, filter?, every_secs?}`) rather than a
/// tagged enum: it keeps the on-disk YAML simple and deserialization
/// robust (serde's internally-tagged / untagged enum representations are
/// fragile under `serde_yaml`), at the cost of allowing field combinations
/// the type system would otherwise forbid. [`validate`] closes that gap —
/// an `event` trigger with `every_secs`, or a `manual` trigger with stray
/// fields, is rejected (fail-closed). Use [`TriggerKind`] to branch.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Trigger {
    /// Which kind of trigger this is.
    #[serde(rename = "type")]
    pub kind: TriggerKind,
    /// Event Bus type/prefix for an `event` trigger, e.g. `file.opened`.
    #[serde(default)]
    pub event: Option<String>,
    /// Optional filter expression over the event payload.
    #[serde(default)]
    pub filter: Option<String>,
    /// Interval in seconds for a `schedule` trigger.
    #[serde(default)]
    pub every_secs: Option<u64>,
}

/// Hard bounds on an `agent` behaviour's loop. Required for `agent`
/// behaviours: an unbounded background LLM loop must not exist.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Budget {
    /// Maximum plan→act→observe iterations.
    pub max_steps: u32,
    /// Maximum tokens across the run.
    pub max_tokens: u32,
    /// Maximum wall-clock milliseconds.
    pub max_wall_ms: u64,
}

/// The machine-readable manifest (the YAML frontmatter).
///
/// This is the behaviour's *declared contract* only. It deliberately does
/// **not** carry trust or enablement state: a `SKILL.md` is untrusted (it
/// may be downloaded or agent-authored), so it must not be able to enable
/// itself or claim its own provenance. Enablement is trusted Settings
/// state and **provenance is stamped by the loader from where the file was
/// found** (both land with the loader in B1); `deny_unknown_fields` means
/// a manifest that tries to set `enabled`/`provenance` is rejected
/// outright.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BehaviourManifest {
    /// Stable identifier, kebab-case.
    pub name: String,
    /// One line shown for routing/selection; preloaded.
    pub description: String,
    /// Workflow or agent.
    pub kind: BehaviourKind,
    /// What dispatches it.
    pub trigger: Trigger,
    /// The action mode the behaviour *requests* — a **ceiling, not a
    /// grant**. A [`BaselineMode`] (Suggest or Supervised), never
    /// Autonomous: a manifest cannot request autonomy (Foundation §8.4).
    /// Because the manifest is untrusted, this can only ever *narrow*
    /// authority: the gate (B1) resolves the effective mode as the lesser
    /// authority of this and the per-app mode from trusted Settings (a
    /// `min`), so a downloaded skill declaring `supervised` can never widen
    /// a Suggest-only configuration. Default Suggest.
    #[serde(default = "default_mode")]
    pub mode: BaselineMode,
    /// Minimum graph read scope the behaviour needs. Default Minimal
    /// (fail-closed): an undeclared behaviour reads nothing rather than
    /// being silently over-granted.
    #[serde(default = "default_reads")]
    pub reads: ReadScope,
    /// Scoped MCP tool grants: tool name → scope list (an empty list means
    /// the tool is granted without a finer scope restriction). This is the
    /// neurosymbolic tool-scope — the loop only ever sees these tools.
    #[serde(default)]
    pub tools: BTreeMap<String, Vec<String>>,
    /// Loop bounds. Required for `agent`, ignored for `workflow`.
    #[serde(default)]
    pub budget: Option<Budget>,
    /// Explicit terminal conditions, each mapped to how its outcome is
    /// surfaced. Never "until the LLM stops". Also evaluated as a
    /// *precondition* at run start (an already-satisfied terminal is an
    /// immediate no-op), not only as a loop exit.
    #[serde(default)]
    pub terminal: BTreeMap<String, Disposition>,
    /// Code handler id. Required for `workflow`, unused for `agent`.
    #[serde(default)]
    pub handler: Option<String>,
}

fn default_mode() -> BaselineMode {
    BaselineMode::Suggest
}

/// Sanity ceilings on loop bounds. These are not operational policy (the
/// engine/config may impose tighter limits) — they reject values so large
/// they defeat the very purpose of a bound, so an absurd or abusive
/// manifest cannot smuggle in an effectively-unbounded run.
const MAX_BUDGET_STEPS: u32 = 10_000;
const MAX_BUDGET_TOKENS: u32 = 100_000_000;
const MAX_BUDGET_WALL_MS: u64 = 86_400_000; // 24 hours

fn default_reads() -> ReadScope {
    ReadScope::Minimal
}

/// A behaviour name must be a stable, identifier-safe token: lowercase
/// ASCII alphanumerics joined by single hyphens, with no leading,
/// trailing, or doubled hyphen. Constraining the charset makes the name
/// safe to use as a directory name, a dispatch key, and a content-free
/// audit subject — even though the manifest itself is untrusted.
fn is_kebab_name(s: &str) -> bool {
    !s.is_empty()
        && !s.starts_with('-')
        && !s.ends_with('-')
        && !s.contains("--")
        && s.bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
}

/// A parsed behaviour: its manifest plus the markdown body.
#[derive(Debug, Clone)]
pub struct Behaviour {
    /// The frontmatter manifest.
    pub manifest: BehaviourManifest,
    /// The markdown body after the frontmatter (goal/instructions/notes).
    pub body: String,
}

/// Parse/validation failures. Every variant means the behaviour is not
/// loaded (fail-closed).
#[derive(Debug, Error, PartialEq)]
pub enum BehaviourError {
    /// No `---` frontmatter block at the start of the document.
    #[error("missing YAML frontmatter (expected a leading --- block)")]
    MissingFrontmatter,
    /// The frontmatter YAML did not parse / deserialize.
    #[error("invalid manifest: {0}")]
    InvalidManifest(String),
    /// A manifest invariant was violated.
    #[error("invalid behaviour '{name}': {reason}")]
    Invalid {
        /// The behaviour name (or "<unnamed>").
        name: String,
        /// Why it was rejected.
        reason: String,
    },
}

/// Split a `---`-delimited YAML frontmatter block from the markdown body.
///
/// Matches the Agent Skills convention: the document opens with `---`,
/// the frontmatter runs to the next line that is exactly `---`, and the
/// body is everything after. (Robust enough for trusted manifest files;
/// the standard tooling splits the same way.)
fn split_frontmatter(content: &str) -> Result<(&str, &str), BehaviourError> {
    let rest = content
        .strip_prefix("---\n")
        .or_else(|| content.strip_prefix("---\r\n"))
        .ok_or(BehaviourError::MissingFrontmatter)?;
    // The closing fence is a line that is exactly "---".
    let close = rest
        .find("\n---\n")
        .or_else(|| rest.find("\n---\r\n"))
        .or_else(|| rest.strip_suffix("\n---").map(|s| s.len()))
        .ok_or(BehaviourError::MissingFrontmatter)?;
    let front = &rest[..close];
    let after = &rest[close..];
    // Skip the closing fence line; the rest is the body.
    let body = after
        .trim_start_matches(['\r', '\n'])
        .strip_prefix("---")
        .map(|b| b.trim_start_matches(['\r', '\n']))
        .unwrap_or("");
    Ok((front, body))
}

/// Parse and validate a behaviour document (`SKILL.md` contents).
pub fn parse(content: &str) -> Result<Behaviour, BehaviourError> {
    let (front, body) = split_frontmatter(content)?;
    let manifest: BehaviourManifest =
        serde_yaml::from_str(front).map_err(|e| BehaviourError::InvalidManifest(e.to_string()))?;
    validate(&manifest, body)?;
    Ok(Behaviour {
        manifest,
        body: body.to_string(),
    })
}

/// Enforce the cross-field invariants the type system cannot.
fn validate(m: &BehaviourManifest, body: &str) -> Result<(), BehaviourError> {
    let reject = |reason: &str| {
        Err(BehaviourError::Invalid {
            name: if m.name.trim().is_empty() {
                "<unnamed>".to_string()
            } else {
                m.name.clone()
            },
            reason: reason.to_string(),
        })
    };

    if m.name.trim().is_empty() {
        return reject("name must not be empty");
    }
    if !is_kebab_name(&m.name) {
        return reject(
            "name must be kebab-case: lowercase a-z, 0-9 and single '-' (e.g. auto-tag-by-project)",
        );
    }
    if m.description.trim().is_empty() {
        return reject("description must not be empty");
    }

    // Trigger field combination (the flat record cannot encode this).
    let t = &m.trigger;
    match t.kind {
        TriggerKind::Event => {
            if t.event.as_deref().map(str::trim).unwrap_or("").is_empty() {
                return reject("an event trigger requires a non-empty `event`");
            }
            if t.every_secs.is_some() {
                return reject("an event trigger must not set `every_secs`");
            }
            // Reject a malformed filter at load time, so the router never
            // sees one (it would otherwise fail-closed and silently never
            // match).
            if let Some(expr) = &t.filter {
                if let Err(e) = crate::router::Filter::parse(expr) {
                    return reject(&format!("invalid trigger filter: {e}"));
                }
            }
        }
        TriggerKind::Schedule => {
            match t.every_secs {
                None => return reject("a schedule trigger requires `every_secs`"),
                // Zero would mean "re-dispatch with no delay" — a tight loop.
                Some(0) => return reject("a schedule trigger's `every_secs` must be greater than 0"),
                Some(_) => {}
            }
            if t.event.is_some() || t.filter.is_some() {
                return reject("a schedule trigger must not set `event` or `filter`");
            }
        }
        TriggerKind::Manual => {
            if t.event.is_some() || t.filter.is_some() || t.every_secs.is_some() {
                return reject("a manual trigger takes no fields");
            }
        }
    }

    match m.kind {
        BehaviourKind::Agent => {
            // A background agentic loop must be bounded and have a goal.
            let Some(budget) = m.budget.as_ref() else {
                return reject(
                    "an agent behaviour must declare a budget (unbounded loops are not allowed)",
                );
            };
            // Every bound must be a positive, sane value: a zero is not a
            // bound at all, and an absurd maximum defeats the purpose.
            if budget.max_steps == 0 || budget.max_tokens == 0 || budget.max_wall_ms == 0 {
                return reject("budget max_steps, max_tokens, and max_wall_ms must each be greater than 0");
            }
            if budget.max_steps > MAX_BUDGET_STEPS {
                return reject("budget max_steps exceeds the sanity ceiling");
            }
            if budget.max_tokens > MAX_BUDGET_TOKENS {
                return reject("budget max_tokens exceeds the sanity ceiling");
            }
            if budget.max_wall_ms > MAX_BUDGET_WALL_MS {
                return reject("budget max_wall_ms exceeds the sanity ceiling");
            }
            if body.trim().is_empty() {
                return reject("an agent behaviour must have a body (the goal/instructions)");
            }
            if m.terminal.is_empty() {
                return reject("an agent behaviour must declare at least one terminal condition");
            }
        }
        BehaviourKind::Workflow => {
            if m.handler.as_deref().map(str::trim).unwrap_or("").is_empty() {
                return reject("a workflow behaviour must name a handler");
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Raw strings: YAML indentation must survive verbatim. (A normal
    // string literal's `\<newline>` continuation strips leading
    // whitespace, which silently destroys the indentation.)
    const WORKFLOW: &str = r#"---
name: auto-tag-by-project
description: Tag a newly opened file with the project it belongs to.
kind: workflow
handler: auto_tag_by_project
reads: session
trigger:
  type: event
  event: file.opened
  filter: "path not_startswith ~/.cache"
tools:
  graph.query: []
  graph.write: [Project, FILE_PART_OF]
---

Resolve the project for the opened file and write a FILE_PART_OF edge.
"#;

    const AGENT: &str = r#"---
name: tidy-downloads
description: Sort files in ~/Downloads into project folders by topic.
kind: agent
mode: supervised
reads: full
tools:
  graph.query: []
  fs.move: []
trigger:
  type: manual
budget:
  max_steps: 8
  max_tokens: 20000
  max_wall_ms: 30000
terminal:
  downloads_empty_or_unsortable: store
  no_confident_moves: silent
---

Group the files in ~/Downloads by project and move them into per-project
subfolders. Move only high-confidence files; leave the rest.
"#;

    #[test]
    fn parses_a_workflow_behaviour() {
        let b = parse(WORKFLOW).expect("valid workflow");
        assert_eq!(b.manifest.name, "auto-tag-by-project");
        assert_eq!(b.manifest.kind, BehaviourKind::Workflow);
        assert_eq!(b.manifest.mode, BaselineMode::Suggest); // default
        assert_eq!(b.manifest.reads, ReadScope::Session);
        assert_eq!(b.manifest.trigger.kind, TriggerKind::Event);
        // The scoped tool round-trips with its scope list.
        assert_eq!(b.manifest.tools.len(), 2);
        assert!(b.manifest.tools.contains_key("graph.query"));
        assert_eq!(b.manifest.tools["graph.write"], ["Project", "FILE_PART_OF"]);
        assert!(b.body.contains("FILE_PART_OF"));
    }

    #[test]
    fn parses_an_agent_behaviour() {
        let b = parse(AGENT).expect("valid agent");
        assert_eq!(b.manifest.kind, BehaviourKind::Agent);
        assert_eq!(b.manifest.mode, BaselineMode::Supervised);
        assert_eq!(b.manifest.reads, ReadScope::Full);
        assert_eq!(b.manifest.reads.tier(), AccessTier::Full);
        assert_eq!(b.manifest.budget.as_ref().unwrap().max_steps, 8);
        // Terminal conditions carry their surfacing disposition.
        assert_eq!(
            b.manifest.terminal.get("no_confident_moves"),
            Some(&Disposition::Silent)
        );
        assert_eq!(b.manifest.trigger.kind, TriggerKind::Manual);
    }

    #[test]
    fn reads_defaults_to_minimal() {
        let src = r#"---
name: x
description: d
kind: workflow
handler: h
trigger:
  type: manual
---
"#;
        let b = parse(src).expect("valid");
        assert_eq!(b.manifest.reads, ReadScope::Minimal); // fail-closed default
    }

    #[test]
    fn rejects_missing_frontmatter() {
        assert!(matches!(
            parse("just a markdown file\n"),
            Err(BehaviourError::MissingFrontmatter)
        ));
    }

    #[test]
    fn rejects_agent_without_budget() {
        let src = r#"---
name: x
description: d
kind: agent
trigger:
  type: manual
terminal:
  done: store
---
goal
"#;
        let err = parse(src).unwrap_err();
        assert!(matches!(err, BehaviourError::Invalid { .. }));
        assert!(format!("{err}").contains("budget"));
    }

    #[test]
    fn rejects_agent_without_terminal_condition() {
        let src = r#"---
name: x
description: d
kind: agent
trigger:
  type: manual
budget:
  max_steps: 1
  max_tokens: 1
  max_wall_ms: 1
---
goal
"#;
        let err = parse(src).unwrap_err();
        assert!(format!("{err}").contains("terminal"));
    }

    #[test]
    fn rejects_workflow_without_handler() {
        let src = r#"---
name: x
description: d
kind: workflow
trigger:
  type: manual
---
"#;
        let err = parse(src).unwrap_err();
        assert!(format!("{err}").contains("handler"));
    }

    #[test]
    fn rejects_unknown_field() {
        // deny_unknown_fields catches typos / unsupported keys.
        let src = r#"---
name: x
description: d
kind: workflow
handler: h
wat: true
trigger:
  type: manual
---
"#;
        assert!(matches!(parse(src), Err(BehaviourError::InvalidManifest(_))));
    }

    #[test]
    fn autonomous_mode_is_not_representable_in_a_manifest() {
        // A manifest must not be able to request autonomy. `mode` is a
        // BaselineMode, so `autonomous` is not a valid value and fails to
        // parse — autonomy is only ever a per-app grant outside the file.
        let src = r#"---
name: x
description: d
kind: workflow
handler: h
mode: autonomous
trigger:
  type: manual
---
"#;
        assert!(matches!(parse(src), Err(BehaviourError::InvalidManifest(_))));
    }

    #[test]
    fn manifest_cannot_self_enable_or_claim_provenance() {
        // Trust state is not a manifest field: a file trying to enable
        // itself or declare its provenance is rejected outright
        // (deny_unknown_fields). Enablement is Settings state; provenance
        // is stamped by the loader from where the file was found.
        for field in ["enabled: true", "provenance: built-in"] {
            let src = format!(
                "---\nname: x\ndescription: d\nkind: workflow\nhandler: h\n{field}\ntrigger:\n  type: manual\n---\n"
            );
            assert!(
                matches!(parse(&src), Err(BehaviourError::InvalidManifest(_))),
                "manifest must reject self-declared `{field}`"
            );
        }
    }

    #[test]
    fn rejects_zero_schedule_interval() {
        let src = r#"---
name: x
description: d
kind: workflow
handler: h
trigger:
  type: schedule
  every_secs: 0
---
"#;
        let err = parse(src).unwrap_err();
        assert!(format!("{err}").contains("every_secs"));
    }

    #[test]
    fn rejects_zero_and_excessive_budget() {
        let zero = r#"---
name: x
description: d
kind: agent
trigger:
  type: manual
budget:
  max_steps: 8
  max_tokens: 1000
  max_wall_ms: 0
terminal:
  done: store
---
goal
"#;
        assert!(format!("{}", parse(zero).unwrap_err()).contains("greater than 0"));

        let huge = r#"---
name: x
description: d
kind: agent
trigger:
  type: manual
budget:
  max_steps: 4000000000
  max_tokens: 1000
  max_wall_ms: 1000
terminal:
  done: store
---
goal
"#;
        assert!(format!("{}", parse(huge).unwrap_err()).contains("ceiling"));
    }

    #[test]
    fn rejects_non_kebab_names() {
        for bad in ["Auto_Tag", "auto tag", "-leading", "trailing-", "dou--ble", "UPPER"] {
            let src = format!(
                "---\nname: \"{bad}\"\ndescription: d\nkind: workflow\nhandler: h\ntrigger:\n  type: manual\n---\n"
            );
            let err = parse(&src).unwrap_err();
            assert!(
                format!("{err}").contains("kebab-case"),
                "expected kebab-case rejection for {bad:?}, got {err}"
            );
        }
    }

    #[test]
    fn rejects_a_malformed_trigger_filter() {
        // A filter that is not `<field> <op> <value>` is caught at load.
        let src = r#"---
name: x
description: d
kind: workflow
handler: h
trigger:
  type: event
  event: file.opened
  filter: "path startswith"
---
"#;
        let err = parse(src).unwrap_err();
        assert!(format!("{err}").contains("filter"), "got: {err}");
    }
}
