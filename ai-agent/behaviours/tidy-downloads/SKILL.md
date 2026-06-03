---
name: tidy-downloads
description: Sort files in ~/Downloads into project folders by inferred topic.
kind: agent
reads: full
mode: supervised
trigger:
  type: schedule
  every_secs: 604800
tools:
  graph.query: []
  fs.list: [~/Downloads]
  fs.move: [~/Downloads, ~/Documents/Projects]
budget:
  max_steps: 40
  max_tokens: 30000
  max_wall_ms: 120000
terminal:
  downloads_empty_or_unsortable: store
  no_confident_moves: silent
---

# tidy-downloads

A bounded agentic loop. For each file in `~/Downloads`, infer the best
destination project folder from the filename plus Knowledge-Graph context;
move only high-confidence files and leave the rest.

Safety notes (from the dry-run):

- Take a Snapper snapshot before the batch so the whole move set is
  reversible at once (gap B1); a `move` onto an occupied destination is an
  irreversible overwrite, so its precondition is "destination empty" — if
  not, treat as high-impact and confirm, never overwrite silently (gap F4).
- The cadence above is weekly; actual firing is gated to an idle window by
  the idle scheduler (B3). The B0 schema only expresses the interval; the
  "and idle" qualifier is the scheduler's concern, not the manifest's.
- Default mode is `supervised`, not `autonomous`: silently rearranging a
  user's files, even reversibly, must be a deliberate per-app opt-in, and
  even then should emit a post-hoc "tidied N files → Undo" summary (gap F7).
