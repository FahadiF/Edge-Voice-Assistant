# Development Guide

Working environment, quality gate, and codebase orientation for people changing EVA's
code. For contribution *process* — issues, PRs, review expectations — see
[CONTRIBUTING.md](../../CONTRIBUTING.md).

---

## 1. Environment setup

Requires **Python 3.12+** and, for the web UI, **Node 20+**.

```bash
git clone https://github.com/FahadiF/Edge-Voice-Assistant.git
cd Edge-Voice-Assistant
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux:    source .venv/bin/activate

pip install -e ".[dev]"
pre-commit install
```

On Linux, PortAudio is needed for audio: `sudo apt-get install libportaudio2`.

The LLM runtime is **not** a base dependency — it ships no PyPI wheels and would require
a C++ toolchain (ADR-013). Install it for your hardware:

```bash
eva setup                    # guided: detects hardware, picks CPU or CUDA build
```

You can develop and run most of the test suite without it: the tests use fake engines.

**Web UI:**

```bash
cd web
npm ci
npm run dev                  # Vite dev server, proxies to the API on :8765
```

---

## 2. The quality gate

Everything below must pass. CI runs the identical commands on Windows and Linux.

```bash
ruff check .                 # lint
ruff format --check .        # formatting
mypy                         # strict type checking, whole package
pytest -m "not integration"  # unit + component tests

cd web
npm run lint                 # eslint
npm run build                # tsc -b && vite build
npm test                     # vitest
```

`pre-commit install` wires lint and format to every commit, and mypy to every push
(mypy is slow enough that gating each commit hurts the edit loop).

**Tests marked `integration`** need real audio hardware or model weights, and are
excluded from CI and from the command above. **No test carries the marker today**, so
`-m "not integration"` currently runs the whole suite — the filter stays in CI so that
such a test is excluded the moment someone writes one. Anything the automated suite
needs but cannot assume is installed is reached through `pytest.importorskip`, which
skips per test rather than per category.

Validation that genuinely needs a microphone, speaker, or GPU is not automated at
all: it is the manual protocol in [MANUAL_TESTING.md](MANUAL_TESTING.md).

---

## 3. Codebase orientation

```
src/eva/
  core/          turn epochs, event bus, registry, task manager, errors
                 — imports nothing else in eva; the dependency leaf
  config/        settings schema (pydantic), persistence, app paths
  audio/         duplex stream, WebRTC APM, ring buffers, VAD segmenter,
                 playback queue, capture pipeline
  vad/ asr/ llm/ tts/ embedding/    one port + registry + adapters each
  memory/        MemoryStore + UserProfileStore ports, SQLite adapter,
                 retriever, summarizer, retention
  conversation/  orchestrator (the turn pipeline), context builder,
                 sentence chunker, personas, languages, markdown filter
  models/        catalog, download manager, integrity verification
  hardware/      detection and tier presets
  server/        FastAPI app: REST routers + WebSocket event stream
  desktop/       pywebview shell, tray, window state, server supervision
  metrics/       per-turn latency, diagnostics snapshot
  plugins/       manifest and discovery (ADR-011; capability wiring not yet built)
  cli.py         one subparser group per concern, thin clients of the services
web/             React + TypeScript + Vite UI; talks to the API only
tests/           unit and component tests with fake adapters
docs/            architecture, ADRs, guides
```

**Dependency direction:** `core` ← subsystems ← `conversation` ← `server` ← clients.
Enforced by review, not tooling. One documented exception: `memory` imports `embedding`'s
port (ADR-010 amendment).

### Where the interesting logic lives

| Question | File |
|---|---|
| How is a turn orchestrated and cancelled? | `conversation/orchestrator.py` |
| How does barge-in actually stop audio? | `audio/playback.py` (`stop()`, fade) + `core/turn.py` |
| When does an utterance end? | `audio/segmenter.py` |
| How is the prompt assembled? | `conversation/context_builder.py` |
| How does text stay in step with speech? | `audio/playback.py` markers → `TtsSentenceStarted` (ADR-028) |
| How are models chosen for hardware? | `hardware/presets.py` |

---

## 4. Architecture rules

The five rules in [CONTRIBUTING.md](../../CONTRIBUTING.md#the-five-architecture-rules) are
the contract. The two that catch people most often:

**Never name a concrete implementation in core code.** If you find yourself writing
`if settings.tts.engine == "kokoro"` outside `tts/`, the port is missing a capability —
add it to the port instead.

**The audio callback is real-time.** `audio/duplex.py::_callback` runs on the PortAudio
thread every 10 ms. No allocation beyond frame copies, no logging, no blocking, no locks
beyond the queues' micro-mutexes. Work belongs on the capture consumer thread.

---

## 5. Testing approach

The turn pipeline — including every barge-in race — is tested with **zero models loaded**,
using fake ASR/LLM/TTS adapters. See `tests/test_orchestrator.py` for the pattern, and
`tests/server_fakes.py` for the shared fakes.

Guidelines:

- Test the contract, not the implementation. Tests naming internal attributes break on
  every refactor.
- Cover the cancellation paths. Interrupt during prefill, during synthesis, during
  playback, and twice in rapid succession — these are where real bugs live.
- Anything needing a microphone, speaker, or model weights gets `@pytest.mark.integration`.
- Pure logic (metrics, WER, chunking, segmentation) should be tested directly; it is
  cheap and it is what conclusions get drawn from.

---

## 6. Debugging

**Per-turn pipeline trace.** Set `EVA_CONVERSATION_TRACE=1` to log a `CVTRACE` line at
every streaming hand-off — first token, each sentence available, synthesis start, first
PCM, playback queue depth — all on one turn-relative clock. Off by default, zero cost.

**Diagnostics.** `GET /api/v1/diagnostics`, the Diagnostics page in the web UI, or
`eva diagnose` for a static report. Includes device placement, input level, dropped
frames, barge-in latency, queue depths, and the live event log.

**Audio problems.** `eva devices` lists devices, `eva listen` shows live VAD and
segmentation, `eva echo-test` measures echo cancellation, and `eva capture-test` records
one utterance raw-vs-processed and decodes both. See
[TROUBLESHOOTING.md](TROUBLESHOOTING.md).

**Logs.** Rotating file logs in the directory `eva diagnose` prints. Set
`developer.log_level` to `DEBUG` in settings, or `developer.debug` to see llama.cpp's own
load report (the only way to confirm how many layers actually offloaded to GPU).

---

## 7. Making changes to settings

Settings are one pydantic document (`config/settings.py`) with strict keys. The JSON
schema drives the entire settings UI — adding a field with a good `description` makes it
appear automatically, correctly typed, with no UI work.

Adding a field with a default requires no migration. Renaming, removing, or restructuring
does: bump `SETTINGS_SCHEMA_VERSION` and add a case to `_migrate_raw()`. The v1→v2
permissions regroup is the worked example.

The web UI's `Settings` type is generated, not hand-mirrored — see "Regenerating API
types" below. Regenerate in the same change as a settings-schema edit, or the mirror
drifts (`tests/test_web_types_sync.py` fails CI until you do).

---

## 7a. Regenerating API types

Most of `web/src/api/types.generated.ts` comes from the backend's OpenAPI schema
(ADR-023's schema-generation amendment). After changing any REST-facing pydantic model:

```bash
cd web
npm run generate:types    # dumps openapi.json, regenerates types.generated.ts
```

Commit the regenerated file — it's committed output, not built fresh in CI, so CI's
existing Python/Node job split never needs a cross-language step.

Two files stay hand-maintained on purpose, in `web/src/api/manual/`:

- **`websocket-types.ts`** — WebSocket event payloads. These travel over `/ws`, not as
  an HTTP response, so `/openapi.json` has no knowledge of them regardless of backend
  schema quality.
- **`dict-response-types.ts`** — `ModelCard` and `MemoryExport`. Their endpoints return
  a plain `dict[str, Any]` with no `response_model`, so there's no schema to generate
  from until one is added.

If a response field is always present once served but shows up as optional (`field?:`)
in the generated output, the backend model likely has a `= None`/`default_factory`
default that Pydantic correctly treats as "not required to construct" — which is a
different question from "always present in the response." Add
`json_schema_serialization_defaults_required=True` to that model's `model_config`
(see `MemoryTurn`, `ResourceUsage`, `UserProfile` for examples) rather than adding a
frontend-side type override or a parallel response-only class.

---

## 8. Release process

1. Quality gate green on Windows and Linux.
2. Manual validation for anything touching audio, timing, or cancellation — see
   [MANUAL_TESTING.md](MANUAL_TESTING.md). Automated tests do not cover perceived latency.
3. Clean-environment smoke test: fresh venv, `pip install -e ".[dev]"`, and every command
   either works or fails with actionable guidance — never `ModuleNotFoundError`
   (release gate, ADR-013).
4. `CHANGELOG.md`, `ROADMAP.md` status, and affected docs updated.
5. Version bumped in `pyproject.toml`, `src/eva/__init__.py`, and `web/package.json`.
6. Tag `vX.Y.Z-alpha`.
