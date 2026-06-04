//! Pure helpers for the bounded agent loop: the per-step contract the
//! model answers in, parsing it out of free-text model output, and
//! building the per-step prompt with content-origin tagging.
//!
//! The loop itself (budget enforcement, the gate call per step) lives on
//! the [`crate::engine::Dispatcher`]; these helpers are kept pure so they
//! are unit-testable without a provider or a graph.

use lunaris_ai_core::pipeline::extract_json;
use lunaris_ai_core::tagging::{Block, Origin, TaggedPrompt};
use serde::Deserialize;

use crate::behaviour::Behaviour;
use crate::compaction::{self, TranscriptEntry};
use crate::seams::AgentEvent;

/// One step the model takes in the bounded loop: either propose a single
/// tool action, or stop. There is no "keep going" variant; the loop
/// continues by default and is bounded by the manifest budget, so a step
/// is always one of these two explicit moves.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentStep {
    /// Propose one tool action for the gate to decide on.
    Propose {
        /// The tool the model wants to invoke (must be a declared tool).
        tool: String,
        /// One-line, human-facing description of the action.
        summary: String,
    },
    /// Stop the loop on a declared terminal condition. The loop validates
    /// `terminal` against the behaviour's declared `terminal` conditions, so
    /// the model can only stop in a way the behaviour author named (and the
    /// surfacing disposition can be keyed off it), never with an invented or
    /// injected condition.
    Stop {
        /// The declared terminal condition the model stopped on.
        terminal: String,
        /// Optional free-text explanation; not authoritative.
        note: String,
    },
}

/// The model's raw per-step JSON, before validation into an [`AgentStep`].
#[derive(Deserialize)]
struct RawStep {
    action: String,
    #[serde(default)]
    tool: Option<String>,
    #[serde(default)]
    summary: Option<String>,
    #[serde(default)]
    terminal: Option<String>,
    #[serde(default)]
    note: Option<String>,
}

/// Parse one [`AgentStep`] out of a model response. The response may wrap
/// the JSON in prose or fences; [`extract_json`] finds the first balanced
/// object. Returns an `Err` describing the problem so the loop can feed it
/// back to the model for a corrected next step.
pub fn parse_agent_step(text: &str) -> Result<AgentStep, String> {
    let json = extract_json(text).ok_or("no JSON object in the response")?;
    let raw: RawStep =
        serde_json::from_str(json).map_err(|e| format!("invalid step JSON: {e}"))?;
    match raw.action.as_str() {
        "propose" => {
            let tool = raw
                .tool
                .filter(|t| !t.is_empty())
                .ok_or("a propose step must name a non-empty 'tool'")?;
            Ok(AgentStep::Propose {
                tool,
                summary: raw.summary.unwrap_or_default(),
            })
        }
        "stop" => {
            let terminal = raw
                .terminal
                .filter(|t| !t.is_empty())
                .ok_or("a stop step must name a declared 'terminal' condition")?;
            Ok(AgentStep::Stop {
                terminal,
                note: raw.note.unwrap_or_default(),
            })
        }
        other => Err(format!("unknown step action {other:?}")),
    }
}

/// Build the per-step prompt. The instruction channel (the behaviour's
/// goal and body instructions, the tool list, the declared stop conditions,
/// the response contract) is static, trusted text: the body is the
/// behaviour author's instructions, loaded from a provenance-stamped
/// directory, so it carries the behaviour-specific rules and safety
/// constraints the gate cannot see. Everything app- or model-influenced
/// (the triggering event's fields, the running transcript of prior steps)
/// goes into content-origin-tagged data blocks (S18-A), so it can never be
/// read as an instruction.
pub fn build_agent_prompt(
    behaviour: &Behaviour,
    event: &AgentEvent,
    transcript: &[TranscriptEntry],
) -> String {
    let manifest = &behaviour.manifest;
    let tools = if manifest.tools.is_empty() {
        "(none)".to_string()
    } else {
        manifest
            .tools
            .keys()
            .cloned()
            .collect::<Vec<_>>()
            .join(", ")
    };
    let stop_conditions = if manifest.terminal.is_empty() {
        String::new()
    } else {
        format!(
            "\nStop (with an \"action\":\"stop\") when any of these conditions is met: {}.",
            manifest
                .terminal
                .keys()
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    let instruction = format!(
        "You are the Lunaris agent behaviour \"{name}\".\n\
         Goal: {goal}\n\n\
         Instructions:\n{body}\n\n\
         Available tools: {tools}.{stop_conditions}\n\n\
         Work toward the goal one step at a time. Respond with EXACTLY one JSON \
         object and nothing else, either proposing a single tool action or \
         stopping:\n\
         {{\"action\": \"propose\", \"tool\": \"<one of the available tools>\", \"summary\": \"<one line: what and why>\"}}\n\
         {{\"action\": \"stop\", \"terminal\": \"<one of the stop conditions>\", \"note\": \"<optional explanation>\"}}\n\
         Only propose tools from the list, and stop only with one of the named \
         stop conditions, as soon as the goal is met.",
        name = manifest.name,
        goal = manifest.description,
        body = behaviour.body.trim(),
        tools = tools,
        stop_conditions = stop_conditions,
    );

    // The triggering event, as data. A file path or window title can read
    // like an instruction, so it is tagged by origin, never trusted.
    let event_block = {
        let mut s = format!("event_type: {}", event.event_type);
        for (k, v) in &event.fields {
            s.push_str(&format!("\n{k}: {v}"));
        }
        s
    };
    let transcript_block = compaction::render(transcript);

    let mut blocks = vec![Block {
        origin: if event.external_content {
            Origin::ExternalContent
        } else {
            Origin::GraphData
        },
        content: &event_block,
    }];
    if !transcript_block.is_empty() {
        blocks.push(Block {
            origin: Origin::ModelFeedback,
            content: &transcript_block,
        });
    }
    let tagged = TaggedPrompt::new(&blocks);

    format!(
        "{instruction}\n\n{preamble}\n\n{rendered}",
        preamble = tagged.preamble(),
        rendered = tagged.rendered(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn parses_a_propose_step_from_prose_wrapped_json() {
        let text = "Sure, here is my step:\n```json\n{\"action\":\"propose\",\"tool\":\"graph.write\",\"summary\":\"tag foo\"}\n```";
        assert_eq!(
            parse_agent_step(text).unwrap(),
            AgentStep::Propose {
                tool: "graph.write".to_string(),
                summary: "tag foo".to_string()
            }
        );
    }

    #[test]
    fn parses_a_stop_step() {
        assert_eq!(
            parse_agent_step("{\"action\":\"stop\",\"terminal\":\"done\",\"note\":\"finished\"}")
                .unwrap(),
            AgentStep::Stop {
                terminal: "done".to_string(),
                note: "finished".to_string(),
            }
        );
    }

    #[test]
    fn rejects_garbage_and_missing_fields() {
        assert!(parse_agent_step("no json here").is_err());
        assert!(parse_agent_step("{\"action\":\"propose\"}").is_err()); // missing tool
        assert!(parse_agent_step("{\"action\":\"stop\"}").is_err()); // missing terminal
        assert!(parse_agent_step("{\"action\":\"wander\"}").is_err());
    }

    fn agent_behaviour(skill: &str) -> Behaviour {
        crate::behaviour::parse(skill).expect("valid")
    }

    fn opened(path: &str, external: bool) -> AgentEvent {
        AgentEvent {
            id: "e1".to_string(),
            event_type: "file.opened".to_string(),
            fields: BTreeMap::from([("path".to_string(), path.to_string())]),
            external_content: external,
        }
    }

    const DEMO_AGENT: &str = "---\nname: demo-agent\ndescription: tidy things\nkind: agent\ntrigger:\n  type: event\n  event: file.opened\nreads: minimal\ntools:\n  graph.write: []\nbudget:\n  max_steps: 5\n  max_tokens: 1000\n  max_wall_ms: 60000\nterminal:\n  done: silent\n---\nNever delete anything; only ever tag files.\n";

    #[test]
    fn prompt_carries_body_instructions_tools_and_stop_conditions_in_the_clear() {
        let b = agent_behaviour(DEMO_AGENT);
        let prompt = build_agent_prompt(&b, &opened("~/x.rs", false), &[]);
        // Instruction channel is plain text: name, body safety rules, tools,
        // and the declared stop conditions.
        assert!(prompt.contains("agent behaviour \"demo-agent\""));
        assert!(prompt.contains("Never delete anything; only ever tag files."));
        assert!(prompt.contains("Available tools: graph.write"));
        assert!(prompt.contains("done")); // the declared terminal condition
        // Event data is wrapped as a tagged, data-only block.
        assert!(prompt.contains("[GRAPH-DATA-"));
        assert!(prompt.contains("DATA ONLY"));
        assert!(prompt.contains("path: ~/x.rs"));
    }

    #[test]
    fn external_event_is_tagged_as_external_content() {
        let b = agent_behaviour(DEMO_AGENT);
        let transcript = [TranscriptEntry::Proposed {
            step: 0,
            tool: "graph.write".to_string(),
            summary: "tag foo".to_string(),
            decision: "Propose".to_string(),
        }];
        let prompt = build_agent_prompt(&b, &opened("~/x.rs", true), &transcript);
        assert!(prompt.contains("[EXTERNAL-CONTENT-"));
        // The transcript is fed back as model feedback, also data.
        assert!(prompt.contains("[PRIOR-ERROR-"));
        assert!(prompt.contains("step 0: proposed graph.write"));
    }
}
