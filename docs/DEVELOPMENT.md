# Developer Guide

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux: source .venv/bin/activate)
pip install -e ".[dev]"
```

## Quality gate

Every change must pass all four before it is considered done:

```bash
ruff check .            # lint
ruff format --check .   # formatting
mypy                    # strict type checking
pytest -m "not integration"
```

Integration tests that need real audio devices or downloaded models are marked
`@pytest.mark.integration` and run manually: `pytest -m integration`.

## Architecture rules

- Read `ARCHITECTURE.md` and the ADRs before changing module boundaries.
- Dependency direction: `core` ← subsystems ← `conversation` ← `server` ← UIs.
  `eva/core` imports nothing else from `eva`. Subsystems (`vad`, `asr`, `llm`,
  `tts`, `memory`, `tools`, …) import only `core` and `config` — never each
  other's adapters.
- Components are referenced by registry id, never by class (ADR-009/ADR-010).
  Adding an engine or model means adding an adapter + registry entry; no core edits.
- Anything that can block goes behind an executor; the audio callback must remain
  allocation-free and lock-free.
- Every artifact that crosses the pipeline carries a turn epoch (ADR-006); any
  new stage must drop stale-epoch items.

## Coding standards

- Python 3.12+, full type hints, `mypy --strict` clean.
- Docstrings explain *constraints and intent*, not what the next line does.
- Errors crossing a subsystem boundary are wrapped in the `eva.core.errors`
  hierarchy.
- No `print` outside the CLI presentation layer — use `logging`.
- Settings: never read files directly; go through `eva.config`. New settings are
  added to the schema (bounds + description) so the API and UI inherit them.
- Tests accompany the change; pure-logic modules (core, endpointing, chunking)
  aim for exhaustive branch coverage with fakes, not mocks of internals.

## Layout

See `ARCHITECTURE.md` §6. Short version: one package per subsystem; each owns its
port (`base.py`), its `registry.py`, and its adapters.

## Running the assistant (from M2)

```bash
eva models list                                  # catalog + install state
eva models download qwen3.5-4b-instruct-q4_k_m   # ~2.7 GB
eva models download kokoro-82m-v1.0              # ~340 MB
eva run                                          # interactive voice loop
eva bench                                        # end-to-end pipeline benchmark, no mic needed
```

faster-whisper weights download automatically on first use (~460 MB for `small`).

## Adding an engine adapter

1. Implement the port (`eva/<subsystem>/base.py`) in a new module in that package.
2. Register a factory in the subsystem's `registry.py` `register_builtins()`
   (or via a plugin entry point once the SDK lands).
3. If the engine needs weights, add a catalog entry (`eva/models/catalog.py`) —
   the consistency tests enforce that settings/profiles only reference catalog ids.
4. Engines are synchronous and single-threaded; the orchestrator provides all
   concurrency and cancellation (see ADR-012). LLM engines must honor
   `should_abort` between tokens.

## Release checklist (draft — finalized in M8)

1. Quality gate green on Windows and Linux CI.
2. Integration suite run on reference hardware; latency targets in budget.
3. CHANGELOG section finalized; version bumped (SemVer).
4. Docs regenerated (API reference, user guide) and reviewed.
5. Installers built and smoke-tested on clean VMs.
