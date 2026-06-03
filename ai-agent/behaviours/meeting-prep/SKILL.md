---
name: meeting-prep
description: Before a calendar meeting, gather related notes and files and offer them.
kind: agent
reads: project
mode: suggest
trigger:
  type: event
  event: calendar.event.upcoming
tools:
  graph.query: []
  calendar.read: [triggering-event]
budget:
  max_steps: 10
  max_tokens: 12000
  max_wall_ms: 15000
terminal:
  suggestion_ready: push
  nothing_relevant_found: silent
---

# meeting-prep

A read-only, Suggest-only behaviour. When a calendar event is upcoming
(the calendar source emits `calendar.event.upcoming` shortly before), read
the event and find related files, notes, and past meetings in the
Knowledge Graph; assemble a compact prep suggestion.

Security note (validated in the dry-run): the event's title/description is
**external content** — anyone can send a calendar invite, so it is a prompt-
injection vector. It enters tagged as `EXTERNAL-CONTENT` (S18-A), is
screened by the S17 classifier, and — because this behaviour is Suggest-
only — it can never act on injected instructions; any future variant that
could act would hit the hardcoded external-content confirmation rule.

Surfacing: `nothing_relevant_found` is `silent` (the P3 value floor — do not
announce having found nothing); a real result pushes, subject to timing and
an expiry (a meeting-prep suggestion is worthless once the meeting starts,
gap F10). Needs `project`-scoped read; if the global read level is lower
the behaviour is disabled with an explanation (gap G3).
