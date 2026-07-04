# Changelog

Notable changes to Edge Voice Assistant. The format follows
[Keep a Changelog](https://keepachangelog.com/); versioning follows SemVer from the
first release onward.

## [Unreleased]

### 2026-07-04 — M2.6: Platform API & UI backend

**Added**
- FastAPI platform API (ADR-017): versioned REST under `/api/v1` plus one
  WebSocket event stream. `eva serve` runs it; the CLI is now one client of
  the same engine services the server exposes (Part 10 — no duplicated logic).
- `eva/server/`: app factory (localhost-only CORS, uniform `EvaError` → HTTP
  status mapping, OpenAPI/Swagger UI generated automatically), `ServerState`
  (the single engine-lifecycle owner — explicit `POST /engine/start`, never
  an implicit side effect of the server booting), and one router per concern:
  settings, models, diagnostics, plugins, conversation, engine, system.
- WebSocket (`/api/v1/ws`): forwards every existing engine event
  (transcripts, LLM tokens/sentences, TTS/playback, state transitions, turn
  lifecycle) plus new `ModelDownloadProgress/Completed/Failed` and
  `EngineStarted/Stopped` events; sends an initial `snapshot` so clients never
  need to poll before their first live event. `EventBus` now keeps bounded
  history for reconnects/diagnostics.
- Settings API: GET/PUT/PATCH/validate/reset + JSON Schema, all backed by a
  new shared `eva.config.service` module (also used by the new `eva config
  show|schema|reset` CLI group).
- Model manager API: list/info/download (background, progress via WebSocket)/
  remove/activate — the full `describe()` model card exposed over HTTP.
- Diagnostics API: `RuntimeSnapshot` with and without a running engine
  (`snapshot_idle` for the "server up, engine not started" state).
- Plugin API (`eva/plugins/`, ADR-011 backend): manifest schema + a genuinely
  functional `PluginManager` using standard entry points (group
  `eva.plugins`) — discover/enable/disable/reload, empty by default until a
  plugin package exists.
- Conversation API: history, current turn, interrupt/cancel (new
  `Orchestrator.interrupt()` — barge-in reachable without a microphone; new
  `TurnCancelled` reason `"manual"`), clear, export/import (new
  `ConversationHistory.turns`/`load_turns`).
- `docs/API.md` (endpoint map + WebSocket protocol) and ADR-017.
- 69 new tests (264 total): every router, the WebSocket stream (including
  disconnect/unsubscribe and multi-client fan-out), the plugin manager against
  fake entry points, the settings service, and full engine start/stop/interrupt/
  export/import cycles against a fake engine — plus a real end-to-end check
  against the installed Qwen3.5/faster-whisper/Kokoro models on reference
  hardware (LLM/ASR on CUDA, TTS on CPU, matching the M2.5 startup banner).
- Verified in a clean virtual environment (standing release gate): base
  install includes FastAPI/uvicorn/websockets with no compiler required;
  `eva serve` runs as a real subprocess answering HTTP and OpenAPI requests.

### 2026-07-04 — M2.5: Production hardening

**Fixed (release blockers)**
- **CI failed on every run because `src/eva/models/` was never committed**: the
  unanchored `.gitignore` pattern `models/` (meant for downloaded weights) also
  matched the source package. Runtime-artifact ignores are now anchored to the
  repo root, and a package-integrity test imports every `eva` module so a
  missing package can never pass CI again. GitHub Actions bumped to
  checkout@v5 / setup-python@v6 (Node deprecation warnings).
- **Inconsistent behavior across restarts** (different model selected, changed
  barge-in feel) — root-caused to unpersisted configuration, silent
  order-dependent device placement, and zero startup visibility; fixed
  architecturally (ADR-015), not by tuning thresholds.

**Added**
- Persistent configuration: first run resolves the active preset against the
  detected hardware tier and writes `settings.json`; model selection is stable
  across restarts and releases, with a pinning regression test.
- Model presets (ADR-015): Balanced / Fast / High Accuracy / Low Memory /
  Developer as registry data per hardware tier; `eva profiles list|set`;
  manual `eva models use <id>` persists and flips the profile to `custom`.
- Startup banner: profile, hardware tier, all four active models (LLM/ASR/
  TTS/VAD), the device each engine actually landed on, and the language.
- Deterministic engine load order (LLM → ASR → TTS): the LLM owns the GPU;
  engine ports expose the `device` actually used.
- Multilingual foundation (ADR-016): language registry with per-language ASR
  hints, prompt notes, and voice preferences; wired through the orchestrator;
  English, Finnish, Swedish, Bengali (tested) plus German and Spanish;
  graceful voice fallback when TTS lacks a native voice.
- Model manager as UI backend: `describe()` full model card (name, version,
  provider, license, languages, context length, VRAM/RAM, quantization, disk
  size, install state, update placeholder, active flag, hardware
  compatibility); `eva models info <id>`; provider/version metadata on every
  catalog entry.
- Developer diagnostics API (`eva.metrics.diagnostics`): JSON-serializable
  runtime snapshot — active models and devices, pipeline state, turn epoch,
  playback/VAD levels, queue depths, dropped frames, CPU/RAM/GPU/VRAM usage,
  last-turn latency metrics (TTFT/TTFA/tokens-per-s), and recent events (the
  bus now keeps a bounded history).
- Configuration system audit: every settings field now carries a description
  (schema-enforced by test); previously hidden defaults promoted to settings
  (`tts.model`, `audio.fade_out_ms`, sentence-chunker bounds,
  `conversation.language`).

**Changed**
- `run_probe` made public (shared by hardware detection and diagnostics);
  the hidden TTS-model mapping in the engine assembly was replaced by the
  `tts.model` setting.

**Tests**
- +37 tests (195 total): package integrity, presets (including preset↔catalog
  consistency), configuration persistence and stability pinning, language
  resolution for en/fi/sv/bn, diagnostics snapshots, model cards and
  compatibility flags, settings-documentation enforcement.

### 2026-07-04 — Guided first-run onboarding

**Added**
- Interactive setup wizard (`eva/onboarding.py`): on `eva run`, if the system
  is not fully set up, EVA explains what will happen (detected hardware,
  recommended runtime, required models with sizes and a time estimate), asks for
  one confirmation, then installs the runtime, downloads models, verifies, and
  starts the assistant — with step-by-step progress. No documentation required
  (ADR-014).
- `eva first-run` command: runs the wizard directly; `--setup-only` finishes
  setup without starting; `--yes` auto-confirms.
- `eva run --yes` for non-interactive/automated first runs.
- Persisted `SetupState` (`config/setup_state.json`) for first-time-vs-repair
  messaging and future config migration; the authoritative readiness gate
  remains the real installed artifacts.
- `download_mb_hint` on catalog entries so the wizard shows honest sizes for
  engine-managed models (e.g. Faster Whisper).

**Changed**
- `eva doctor` and the `run`/`bench` preflight now share one `check_readiness`
  implementation with the wizard (no duplicated readiness logic).
- `eva run` no longer just prints commands when setup is incomplete — it guides
  the user through it. Failures are reported in friendly language; tracebacks
  are never shown to end users.

**Preserved**
- `eva setup`, `eva doctor`, `eva models`, `eva diagnose` remain first-class
  developer tools; the wizard reuses them rather than duplicating logic.

**Tests**
- +16 tests (158 total): onboarding readiness, plan + estimates,
  confirm/decline, non-interactive blocking, full-run step execution, friendly
  failure, and state persistence — all hermetic (no network, models, or audio).

### 2026-07-04 — M2 packaging fix: installable from a clean checkout

**Fixed (release blocker)**
- Declared the ML runtimes that were used but missing from `pyproject.toml`.
  `faster-whisper` and `kokoro-onnx` (both ship universal PyPI wheels) are now
  base dependencies, so `pip install -e "."` yields a runnable ASR + TTS + audio
  application with no compiler. Previously a clean checkout failed at runtime
  with `No module named 'faster_whisper'` / `'llama_cpp'`.

**Added**
- `eva setup`: detects hardware and installs the `llama-cpp-python` build
  (CPU or CUDA) from the llama.cpp wheel index — the LLM runtime has no PyPI
  wheels, so it cannot be a plain dependency (ADR-013). Supports `--cpu`,
  `--cuda`, `--dry-run`, `--force`.
- `eva doctor`: readiness report listing every runtime and model as
  `ok`/`MISSING` with the exact remedy command.
- `[cpu]` and `[cuda]` optional-dependency extras for manual/reproducible
  installs; the `[cuda]` extra also pulls the NVIDIA cudart/cuBLAS wheels.
- `eva.runtime` module: runtime probing and install-command construction (pure,
  unit-tested).
- Preflight in `eva run` and `eva bench`: both now report missing runtimes and
  models with actionable guidance instead of raising `ModuleNotFoundError`.
- `docs/INSTALLATION.md` (Windows + Linux) and ADR-013.
- 17 new tests (142 total): runtime probing, variant selection, install-command
  construction, CLI `doctor`/`setup`/graceful-preflight behavior, and download
  truncation/resume.
- Established the clean-environment smoke test as a per-milestone release gate.

### 2026-07-04 — M2: Streaming conversational pipeline

**Added**
- Event system (`eva.core.events`): typed, JSON-serializable engine events
  (turn lifecycle, transcripts, LLM tokens/sentences, TTS, state changes) on an
  asyncio event bus with bounded per-subscriber queues and thread-safe publish.
- Turn management (`eva.core.turn`): monotonic turn epochs as the cancellation
  backbone (ADR-006); every pipeline artifact is epoch-tagged and stale work
  aborts at the next boundary.
- Engine ports + registries: `ASREngine`, `LLMEngine` (streaming + per-token
  abort contract), `TTSEngine` (sentence-granular synthesis, voice discovery),
  mirroring the M1 VAD registry (ADR-010).
- Adapters: faster-whisper (CTranslate2, CUDA→CPU fallback, greedy decode
  tuned for short utterances), llama.cpp (GGUF, streaming chat completion,
  abort per token, Windows CUDA DLL resolution), Kokoro via kokoro-onnx
  (torch-free, 24→16 kHz resampling at the adapter boundary) — ADR-012.
- Turn orchestrator (`eva.conversation.orchestrator`): asyncio pipeline —
  LLM producer thread → token consumer → sentence chunker → speak worker;
  barge-in/supersede/shutdown cancellation; partial transcripts from
  segmenter `UtteranceProgress` snapshots; per-turn metrics.
- Punctuation-aware sentence chunker (abbreviation/decimal guards, clause-
  boundary force-split for run-on generations).
- Conversation history with turn windowing (persistence lands in M4).
- Model manager backend: catalog as data (ids, licenses, sizes, VRAM/RAM
  needs, verified download URLs), atomic downloads with progress, install/
  remove/resolve, disk usage; consistency tests keep settings/profiles/catalog
  aligned.
- Default LLM updated to **Qwen3.5-4B** Q4_K_M (ADR-002 amendment).
- CLI: `eva run` (interactive voice loop with live token streaming),
  `eva models list|download|remove`, `eva bench` (reproducible end-to-end
  pipeline benchmark using TTS-generated speech — no microphone needed).
- Per-turn metrics collection (ASR, TTFT, tokens/s, first-sentence TTS, TTFA,
  total) with session summary.
- Dependencies: faster-whisper, kokoro-onnx, llama-cpp-python (CUDA wheel) +
  nvidia CUDA runtime wheels.
- 56 new unit tests (127 total): event bus, turn controller, chunker,
  history, resampler, model manager (truncation detection, resume, failure
  atomicity), and full orchestrator coverage with fake engines (streaming
  order, barge-in cancellation, superseding, repeated interruptions, failure
  paths, partial transcripts, metrics).

**Fixed**
- Model downloads now verify received bytes against Content-Length and resume
  via HTTP Range on retry — a dropped connection previously produced a
  silently truncated model file that failed at load time.

**Benchmarks** (RTX 3060 Laptop 6 GB, Ryzen 9 5900HX; `eva bench`, warm run)
- ASR (faster-whisper small int8, CUDA): 490 ms for 2.9 s of speech
- Time to first token (ASR + LLM prefill): 535 ms
- LLM (Qwen3.5-4B Q4_K_M, full GPU offload): 65 tok/s
- First reply sentence ready: 140 ms after generation start
- First-sentence TTS (Kokoro, CPU): ~1.3 s (RTF ≈ 0.6)
- Estimated time to first audio: ~2.0 s — dominated by first-sentence TTS;
  identified M3/M7 lever: chunked/streamed synthesis of the first segment
  (kokoro-onnx supports incremental synthesis) and/or shorter first segment.
- Model load time (all three engines): ~16 s cold.

### 2026-07-03 — M1: Full-duplex audio core

**Added**
- Canonical audio format: 16 kHz mono int16 in 10 ms frames (`eva.audio.frames`),
  with level metering and float/int conversion helpers.
- `FrameRing`: bounded, drop-oldest frame queue between the audio callback and
  consumer threads, with overflow diagnostics.
- `PlaybackQueue`: frame-granular playback with click-free fade-out (40 ms) —
  the mechanism barge-in uses to silence the assistant instantly.
- `DuplexAudioStream`: one PortAudio stream for capture + playback (single
  clock), real-time-safe callback, measured loop delay reported to the echo
  canceller, per-callback error containment.
- WebRTC APM integration (`WebRtcAudioProcessor`): echo cancellation, noise
  suppression, AGC, high-pass filter; graceful fallback to passthrough when the
  native module is unavailable.
- VAD subsystem (`eva.vad`): `VADEngine` port, Silero adapter (ONNX, no torch),
  and the platform's first component registry (`eva.core.registry.Registry`).
- `SpeechSegmenter`: pure-logic endpointing state machine — 300 ms pre-roll,
  noise gate, mid-utterance pause tolerance, max-utterance safety stop, and
  single-shot barge-in confirmation that keeps the triggering speech for ASR.
- `CapturePipeline` consumer thread (frames → VAD chunks → segmenter events)
  and `AudioSystem` composition root.
- CLI: `eva devices`, `eva listen` (live VAD monitor), `eva echo-test`
  (speaker/microphone echo-suppression measurement with pass/fail verdict).
- Dependencies: numpy, sounddevice, livekit (WebRTC APM), pysilero-vad (<3).
- 48 new unit tests (71 total), including a device-free APM test proving
  >10 dB attenuation of a synthetic echo.

**Verified**
- Full quality gate green (ruff, mypy strict, pytest) on Windows.
- Live duplex run on reference hardware: WebRTC APM active, 0 callback errors,
  no VAD self-triggers during tone playback.

### 2026-07-03 — Architecture review & project identity

- Renamed the project to **Edge Voice Assistant** across the repository
  (folder `edge-voice-assistant`, docs, architecture, roadmap); release
  versioning now targets 1.0.0.
- ADR-010: subsystem packages (`vad/`, `asr/`, `llm/`, `tts/`, `memory/`,
  `tools/`, …) each owning port + registry + adapters, replacing the
  `ports/`/`adapters/` layering; single registry primitive in `core`;
  dependency-direction rule documented.
- ADR-011: plugin SDK — manifest + entry points, narrow `eva.sdk` facade,
  marketplace-ready lifecycle (install/update/enable/disable/remove).
- Hardware profiles redesigned as two layers: detected capability tier →
  goal-oriented presets (Balanced / Fast / High Accuracy / Low Memory /
  Developer / Custom, user-editable).
- Settings surface expanded to the full section list (General, per-subsystem
  model managers, Conversation, Memory, Prompt Templates, Personalities, Audio,
  Hardware, Performance, Plugins, Developer, Diagnostics, Appearance,
  Accessibility, Privacy, Updates).
- Added `docs/DEVELOPMENT.md` (setup, quality gate, architecture rules, coding
  standards, release checklist).

### 2026-07-03 — M0: Project foundation

**Added**
- Installable `eva` package (src layout, `pyproject.toml`, MIT license, typed).
- Settings system: strict pydantic schema (audio, VAD, ASR, LLM, TTS, conversation,
  server, UI, developer sections) with validation bounds, atomic JSON persistence,
  and partial-file merge. VAD defaults carry over the values tuned in the thesis
  prototype.
- Application paths via platformdirs, with `EVA_HOME` override for portable
  installs and test isolation.
- Hardware detection (psutil + `nvidia-smi`/`rocm-smi` probes; degrades to
  CPU-only, never raises) and hardware-profile recommendation
  (`cpu-only` / `gpu-6gb` / `gpu-12gb`).
- Logging: console + rotating file handler, optional JSON line format.
- CLI: `eva diagnose` (system/hardware/profile/configuration/paths report) and
  `eva version`; UTF-8 output enforced on Windows consoles.
- Tooling: ruff (lint + format), mypy strict with pydantic plugin, pytest
  (23 unit tests), CI workflow for Windows + Linux.

**Verified**
- Full quality gate green (lint, format, types, tests).
- `eva diagnose` on reference hardware (RTX 3060 Laptop, 6 GB VRAM) detects the
  GPU and recommends the `gpu-6gb` profile.

### 2026-07-03 — Project inception

- Analyzed the thesis prototype; findings in `docs/THESIS_ANALYSIS.md`.
- Evaluated the current open-weight model landscape (ASR, LLM, TTS, VAD, AEC).
- Defined the system architecture (`docs/ARCHITECTURE.md`), roadmap
  (`docs/ROADMAP.md`), and ADR-001 … ADR-009.
