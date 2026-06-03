---
name: auto-tag-by-project
description: Tag a newly opened file with the project it belongs to.
kind: workflow
reads: project
mode: supervised
handler: auto_tag_by_project
trigger:
  type: event
  event: file.opened
  filter: "path not_startswith ~/.cache"
tools:
  graph.query: []
  graph.write: [Project, FILE_PART_OF]
terminal:
  file_tagged: store
  no_matching_project: silent
  already_tagged: silent
---

# auto-tag-by-project

A deterministic workflow (almost no LLM). When a file is opened, resolve
the project it belongs to from its path prefix against existing `Project`
nodes. If exactly one project matches and no `FILE_PART_OF` edge exists,
propose creating that edge.

Ambiguity rule (design-doc gap G2): if the path prefix-matches more than
one project, the most specific (longest prefix) wins; if two candidates are
equally specific, escalate rather than guess.

Terminal `already_tagged` is also a precondition: if the edge already
exists the run is an immediate no-op (gap F2).
