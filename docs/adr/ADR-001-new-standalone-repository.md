# ADR-001: New standalone repository, thesis repo frozen

Status: Accepted · Date: 2026-07-03

## Context
The thesis implementation must remain permanently untouched as the historical
reference. Options: (a) branch `production-v2` in the thesis repo, (b) new folder in
the thesis workspace, (c) new independent repository.

## Decision
New independent git repository, `edge-voice-assistant`, created as a sibling folder
in the current workspace (own `.git`, own GitHub remote). The thesis repo is never
branched or modified.

## Rationale
- The new implementation shares essentially **no code** with the prototype (different runtimes, structure,
  UI, packaging) — a branch would carry misleading history and invite accidental
  merges into the thesis repo.
- An independent repo gets its own CI, releases, issues, and license without ever
  risking the frozen reference.
- The workspace keeps both side by side for easy comparison during development.

## Consequences
- Thesis knowledge is carried over via a written analysis (internal notes), not git history.
- The folder can be relocated anywhere later without breaking anything.
