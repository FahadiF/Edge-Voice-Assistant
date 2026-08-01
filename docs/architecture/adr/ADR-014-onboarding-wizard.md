# ADR-014: Guided onboarding wizard on first run

Status: Accepted · Date: 2026-07-04

## Context
After ADR-013 made the app installable, first use still required a new user to
know three commands (`eva setup`, `eva models download …`, `eva run`). For a
product aiming at non-technical users (LM Studio / Docker Desktop / Home
Assistant class), that is too much documentation to read before hearing a first
response. The developer commands must remain, but `eva run` should just work.

## Decision
1. **`eva run` gates on readiness, then guides.** It computes readiness; if
   complete it starts immediately, otherwise it launches an interactive wizard
   that explains what will happen, shows detected hardware, the recommended
   runtime, the required models with sizes and time estimates, asks for one
   confirmation, then performs runtime install → model download → verify → start
   — with step-by-step progress.
2. **`eva first-run`** invokes the same flow explicitly (and can stop before
   starting with `--setup-only`).
3. **The wizard owns no installation logic.** It orchestrates the existing
   pieces — hardware detection, `eva.runtime` install, `ModelManager` downloads,
   runtime probing — so there is exactly one implementation of each concern
   (`eva/onboarding.py`). `eva doctor` and the `run`/`bench` preflight share the
   wizard's `check_readiness`.
4. **Readiness is derived from real artifacts**, not a flag: runtimes must be
   importable and model files present. A persisted `SetupState`
   (`config/setup_state.json`) records completion for messaging ("first time"
   vs. repair) and future config migration, but never overrides the artifact
   check — so a deleted model correctly re-triggers setup.
5. **Never show a traceback.** Every step is wrapped; failures print what
   failed, why, and how to fix it, and return a friendly result.
6. **Non-interactive safety.** Without a TTY and without `--yes`, the wizard
   prints the plan and exits non-zero instead of blocking on a prompt, so
   scripts and CI behave predictably.

## Alternatives rejected
- **Keep listing commands and exit** — the M2 behavior; fails the "no docs
  needed" goal.
- **A separate installer binary** — deferred to M8 packaging; the in-app wizard
  serves developers and source installs now and will run inside the packaged
  app later.
- **A flag-file as the readiness source of truth** — brittle; a stale "done"
  flag survives a deleted/corrupted model. Artifact probing is authoritative.

## Consequences
- New steps (additional models, runtime/model updates, config migration,
  post-setup benchmark, a demo conversation, microphone/speaker calibration)
  are added as `_Step`s in `_build_steps` without touching the control flow.
- `eva setup`, `eva doctor`, `eva models`, `eva diagnose` remain first-class and
  unchanged for advanced users.
