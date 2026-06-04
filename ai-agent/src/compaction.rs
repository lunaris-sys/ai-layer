//! Context compaction for the bounded agent loop's working memory.
//!
//! As a long-running agent behaviour accumulates loop steps, the per-step
//! prompt grows. Compaction keeps it inside the model's input context window
//! with a **fixed-buffer threshold** (fire when the prompt would exceed the
//! window minus a reserved headroom) and the convergent harness's
//! cheap-pruning-first policy, done **deterministically**:
//!
//! 1. *Prune* collapses runs of redundant correction feedback (the loop's own
//!    re-prompts after an invalid step) with no model call.
//! 2. *Tighten* drops the human-readable rationale from older proposed-action
//!    entries while keeping every load-bearing fact (the tool, the gate
//!    decision, and every refusal) verbatim. No model output ever replaces an
//!    authoritative loop fact, so a degraded or adversarial model cannot erase
//!    a refusal or invert a decision through compaction.
//! 3. If the prompt still will not fit, the caller terminates the loop closed
//!    rather than send an over-window prompt or silently truncate.
//!
//! This is the design's *cheap, model-free tier* (Foundation P8: prune before
//! summarise). The *expensive* tier, an LLM summary of bulk history (long tool
//! results) with a structured, validated summary contract, lands with the real
//! provider in a later increment; it is deliberately out of this tier because
//! replacing authoritative working memory with unvalidated model prose is
//! unsafe. Because this tier makes no model call it spends no tokens and no
//! wall-clock, so it cannot fail on a provider or a budget.
//!
//! Today the transcript is bounded by the manifest step budget, so the
//! threshold is rarely reached; this is defensive infrastructure for long
//! behaviours and (later) large tool results. The real per-model window is a
//! provider property; until it is wired the policy carries a deliberately low
//! default so an unset policy fails safe (compacts early, never overflows).
//!
//! The token-window check uses the loop's coarse 4-bytes-per-token estimate,
//! which can under-count token-dense input; a model-accurate tokenizer is a
//! provider property that lands with the provider. The low default window
//! keeps the estimate conservative in the meantime.

/// How the loop's working memory is kept inside the model's context window:
/// a fixed-buffer threshold (no adaptive/ML sizing), then deterministic prune
/// and tighten passes.
#[derive(Debug, Clone)]
pub struct CompactionPolicy {
    /// Tokens reserved below the model's window for the response and any growth
    /// between the check and the call, so compaction fires before the window
    /// is actually full. The window itself is not stored here: it is a
    /// property of the wired model, read from the provider per run, so the
    /// bound always tracks the real backend rather than a guess.
    pub headroom: u32,
    /// The most-recent transcript entries kept in full detail (never
    /// tightened), so the model always sees its latest moves with their
    /// rationale intact.
    pub keep_recent: usize,
}

impl Default for CompactionPolicy {
    fn default() -> Self {
        Self {
            headroom: 2_048,
            keep_recent: 4,
        }
    }
}

impl CompactionPolicy {
    /// The estimated token count at or below which a prompt fits the given
    /// model window (the window minus the reserved headroom).
    pub fn threshold(&self, context_window: u32) -> u32 {
        context_window.saturating_sub(self.headroom)
    }

    /// Whether a prompt of this estimated token size needs compaction against
    /// the given model window.
    pub fn over(&self, context_window: u32, prompt_tokens: u32) -> bool {
        prompt_tokens > self.threshold(context_window)
    }
}

/// One entry in the bounded loop's working memory. Structured rather than a
/// pre-formatted string, so compaction operates on the *facts* (which tool a
/// step proposed, the gate's decision, a refusal and its reason) and never has
/// to recover them from, or be fooled by, model-controlled text. The facts are
/// preserved verbatim through compaction; only the rationale prose of an older
/// proposal is dropped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TranscriptEntry {
    /// A proposed action and the gate's decision on it. `summary` is the
    /// model's one-line rationale; it is the only part compaction may drop.
    Proposed {
        /// The loop step that produced this entry.
        step: u32,
        /// The proposed tool (a load-bearing fact, never dropped).
        tool: String,
        /// The model's one-line rationale (dropped when the entry is tightened).
        summary: String,
        /// The gate decision, debug-rendered (a load-bearing fact, never dropped).
        decision: String,
    },
    /// The gate refused a proposed action. Never tightened or dropped, so the
    /// model cannot lose the fact that it must not retry the refused action.
    Refused {
        /// The loop step that produced this entry.
        step: u32,
        /// Why the gate refused (kept verbatim).
        reason: String,
    },
    /// A correction re-prompt the loop fed back (an invalid step, or a stop on
    /// an undeclared condition). Carries no decision and repeats while the
    /// model is confused, so a run of them is safe to collapse.
    Nag {
        /// The loop step that produced this entry.
        step: u32,
        /// The re-prompt text after the `step {n}: ` prefix.
        detail: String,
    },
}

impl TranscriptEntry {
    fn is_nag(&self) -> bool {
        matches!(self, TranscriptEntry::Nag { .. })
    }

    /// Render this entry to its transcript line. A `Proposed` whose `summary`
    /// is empty (originally empty, or tightened) omits the rationale clause.
    fn render(&self) -> String {
        match self {
            TranscriptEntry::Proposed {
                step,
                tool,
                summary,
                decision,
            } => {
                if summary.is_empty() {
                    format!("step {step}: proposed {tool}; gate decision: {decision}")
                } else {
                    format!("step {step}: proposed {tool} ({summary}); gate decision: {decision}")
                }
            }
            TranscriptEntry::Refused { step, reason } => {
                format!("step {step}: action refused ({reason})")
            }
            TranscriptEntry::Nag { step, detail } => format!("step {step}: {detail}"),
        }
    }
}

/// Render the transcript to the newline-joined text the prompt's data block
/// carries. The caller wraps it in a content-origin-tagged block (S18-A); this
/// is plain rendering.
pub fn render(transcript: &[TranscriptEntry]) -> String {
    transcript
        .iter()
        .map(TranscriptEntry::render)
        .collect::<Vec<_>>()
        .join("\n")
}

/// Cheap, model-free prune: collapse each run of consecutive correction-nag
/// entries to its most recent one. Proposals, decisions, and refusals are
/// never touched, so no load-bearing fact is dropped. Returns whether the
/// transcript changed.
pub fn prune(transcript: &mut Vec<TranscriptEntry>) -> bool {
    let before = transcript.len();
    let mut out: Vec<TranscriptEntry> = Vec::with_capacity(before);
    for entry in transcript.drain(..) {
        if entry.is_nag() && out.last().is_some_and(TranscriptEntry::is_nag) {
            // Replace the previous nag with this newer one (keep the most
            // recent feedback, drop the stale duplicate).
            *out.last_mut().expect("checked non-empty above") = entry;
        } else {
            out.push(entry);
        }
    }
    *transcript = out;
    transcript.len() != before
}

/// Model-free tighten: drop the rationale prose from every `Proposed` entry
/// older than the kept tail, keeping the tool and the gate decision verbatim.
/// Refusals and the recent tail are untouched. This shrinks the prompt without
/// losing any load-bearing fact. Returns whether anything was tightened.
pub fn tighten(transcript: &mut [TranscriptEntry], keep_recent: usize) -> bool {
    let len = transcript.len();
    if len <= keep_recent {
        return false;
    }
    let mut changed = false;
    for entry in &mut transcript[..len - keep_recent] {
        if let TranscriptEntry::Proposed { summary, .. } = entry {
            if !summary.is_empty() {
                summary.clear();
                changed = true;
            }
        }
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;

    fn proposed(step: u32) -> TranscriptEntry {
        TranscriptEntry::Proposed {
            step,
            tool: "graph.write".to_string(),
            summary: format!("tag file {step}"),
            decision: "RequireConfirmation".to_string(),
        }
    }
    fn refused(step: u32) -> TranscriptEntry {
        TranscriptEntry::Refused {
            step,
            reason: "tool fs.delete out of scope".to_string(),
        }
    }
    fn nag(step: u32) -> TranscriptEntry {
        TranscriptEntry::Nag {
            step,
            detail: "your response was not a valid step (no JSON object)".to_string(),
        }
    }

    #[test]
    fn threshold_is_window_minus_headroom() {
        let p = CompactionPolicy {
            headroom: 200,
            keep_recent: 4,
        };
        assert_eq!(p.threshold(1000), 800);
        assert!(!p.over(1000, 800));
        assert!(p.over(1000, 801));
    }

    #[test]
    fn threshold_saturates_when_headroom_exceeds_window() {
        let p = CompactionPolicy {
            headroom: 500,
            keep_recent: 4,
        };
        // Degenerate config: threshold floors at 0, so everything is "over"
        // (fail toward compacting/closing, never toward an oversized prompt).
        assert_eq!(p.threshold(100), 0);
        assert!(p.over(100, 1));
    }

    #[test]
    fn render_omits_the_rationale_only_when_the_summary_is_empty() {
        let full = proposed(3).render();
        assert_eq!(
            full,
            "step 3: proposed graph.write (tag file 3); gate decision: RequireConfirmation"
        );
        let tight = TranscriptEntry::Proposed {
            step: 3,
            tool: "graph.write".to_string(),
            summary: String::new(),
            decision: "RequireConfirmation".to_string(),
        }
        .render();
        assert_eq!(
            tight,
            "step 3: proposed graph.write; gate decision: RequireConfirmation"
        );
    }

    #[test]
    fn prune_collapses_runs_of_nags_keeping_the_most_recent() {
        let mut t = vec![proposed(0), nag(1), nag(2), nag(3), proposed(4)];
        assert!(prune(&mut t));
        assert_eq!(t, vec![proposed(0), nag(3), proposed(4)]);
    }

    #[test]
    fn prune_never_touches_proposals_or_refusals() {
        let mut t = vec![proposed(0), refused(1), proposed(2), refused(3)];
        assert!(!prune(&mut t));
        assert_eq!(t.len(), 4);
    }

    #[test]
    fn prune_keeps_non_consecutive_nags_separated_by_substance() {
        let mut t = vec![nag(0), proposed(1), nag(2)];
        assert!(!prune(&mut t)); // not a run: a proposal sits between them
        assert_eq!(t.len(), 3);
    }

    #[test]
    fn tighten_drops_old_rationale_but_keeps_tool_decision_and_refusals() {
        let mut t = vec![proposed(0), refused(1), proposed(2), proposed(3), proposed(4)];
        assert!(tighten(&mut t, 2)); // keep the last two in full
        // Old proposals lose their rationale prose but keep tool + decision.
        match &t[0] {
            TranscriptEntry::Proposed { summary, tool, decision, .. } => {
                assert!(summary.is_empty());
                assert_eq!(tool, "graph.write");
                assert_eq!(decision, "RequireConfirmation");
            }
            other => panic!("expected a tightened proposal, got {other:?}"),
        }
        // The refusal is untouched, and the recent tail keeps full detail.
        assert_eq!(t[1], refused(1));
        assert_eq!(t[3], proposed(3));
        assert_eq!(t[4], proposed(4));
    }

    #[test]
    fn tighten_is_a_noop_within_the_kept_tail() {
        let mut t = vec![proposed(0), proposed(1)];
        assert!(!tighten(&mut t, 4)); // keep_recent >= len
        assert_eq!(t, vec![proposed(0), proposed(1)]);
    }

    #[test]
    fn tighten_actually_shrinks_the_rendered_transcript() {
        let mut t: Vec<TranscriptEntry> = (0..8).map(proposed).collect();
        let before = render(&t).len();
        assert!(tighten(&mut t, 2));
        assert!(render(&t).len() < before, "tightening must shrink the render");
    }
}
