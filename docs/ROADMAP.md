# Roadmap

Each milestone ends in a working, tested, demonstrable state, with documentation
and CHANGELOG updated; a milestone is reviewed and signed off before the next one
begins. The order is chosen so the riskiest platform-dependent piece (full-duplex
audio) and the product's core differentiator (barge-in) are validated earliest.

## M0 — Project foundation ✅ (completed 2026-07-03)
Scaffold `src/eva` package, pyproject, ruff + mypy (strict, pydantic plugin) +
pytest, GitHub Actions CI workflow (Windows + Linux), logging (console + rotating
file, optional JSON), pydantic settings schema + atomic JSON persistence, app paths
(platformdirs, `EVA_HOME` override), hardware detection (CPU/RAM/NVIDIA/ROCm probes)
with profile recommendation. **Exit met:** `eva diagnose` prints a hardware/config
report (verified on RTX 3060 Laptop → `gpu-6gb`); ruff/mypy/pytest all green
(23 tests). Note: the CI workflow first runs once the repository is published.

## M1 — Full-duplex audio core ✅ (completed 2026-07-03)
Duplex PortAudio stream (single clock), frame rings, WebRTC APM integration
(AEC/NS/AGC with far-end reference and measured loop-delay reporting), Silero VAD
adapter behind the first ADR-010 registry, utterance segmenter with pre-roll
buffer, noise gate, and barge-in confirmation window, capture pipeline thread,
device enumeration, processor fallback (APM → passthrough). CLI diagnostics:
`eva devices`, `eva listen`, `eva echo-test` (records the user's voice, replays it
over the speakers, reports raw vs cleaned echo level and VAD self-triggers).
**Exit met:** APM attenuates a synthetic pure echo by >10 dB (device-free test in
the default suite); live duplex run on reference hardware with APM active, zero
callback errors, no VAD events during playback; segmenter unit tests prove
barge-in confirmation at the configured window (default 200 ms) with the
triggering speech retained for ASR. Deferred to M7: prefer WASAPI/low-latency
host APIs (MME default reports ≈210 ms loop delay).

## M2 — Streaming pipeline v1 (CLI) ✅ (completed 2026-07-04)
Event system + turn epochs (ADR-006), ASR/LLM/TTS ports + registries + adapters
(faster-whisper, llama.cpp with Qwen3.5-4B Q4 — ADR-002 amendment, Kokoro via
kokoro-onnx — ADR-012), asyncio turn orchestrator with producer/consumer/speaker
pipelining and full cancellation, punctuation-aware sentence chunker, partial
transcripts, conversation history, model manager backend (catalog, downloads,
resolution), per-turn metrics. CLI: `eva run`, `eva models`, `eva bench`.
**Exit met:** streaming spoken conversation end-to-end; orchestrator control
flow (including barge-in cancellation and repeated interruptions) unit-tested
with fake engines; benchmark results recorded in the changelog.

## M2.5 — Production hardening ✅ (completed 2026-07-04)
CI made authoritative (root cause: a `.gitignore` pattern excluded
`src/eva/models/` from the repository — fixed, plus a package-integrity test).
Deterministic runtime configuration (ADR-015): persisted settings, model
presets (Balanced/Fast/High Accuracy/Low Memory/Developer + custom), startup
banner with actual device placement, deterministic LLM→ASR→TTS load order.
Multilingual foundation (ADR-016): language registry (en/fi/sv/bn tested).
Model manager `describe()` cards + `eva models info|use`, `eva profiles`.
Developer diagnostics API (runtime snapshot: models, devices, state, resources,
latency metrics, events). Configuration audit: all fields documented,
hidden defaults promoted to settings. 195 tests.

## M2.6 — Platform API & UI backend ✅ (completed 2026-07-04)
FastAPI platform API (ADR-017): versioned REST (`/api/v1`) + one WebSocket
event stream, so the CLI, desktop app, web UI, and plugins are all thin
clients of the same engine. Routers: settings (get/put/patch/validate/reset +
JSON Schema), models (list/info/download/remove/activate, background
downloads with WebSocket progress), diagnostics (`RuntimeSnapshot`, idle and
running), plugins (ADR-011 backend: manifest + entry-point discovery +
enable/disable/reload), conversation (history/current/interrupt/cancel/clear/
export/import), engine lifecycle (explicit start/stop, never implicit).
`ServerState` is the single lifecycle owner; every router reuses existing
services (no duplicated logic — Part 10). `eva serve` CLI command; `eva
config show|schema|reset` shares the new `eva.config.service` module with the
Settings API. OpenAPI/Swagger UI generated automatically.
**Exit met:** 264 tests total (69 new) covering every router, the WebSocket
stream (multi-client fan-out, disconnect/unsubscribe), the plugin manager
against fake entry points, and full engine start/stop/interrupt/export/import
cycles; verified against the real installed models on reference hardware
(LLM/ASR on CUDA, TTS on CPU via the API, matching the M2.5 startup banner);
clean-environment smoke test passed (FastAPI/uvicorn/websockets are base
dependencies with universal wheels — no clean-install regression); `eva
serve` verified as a real subprocess answering HTTP + OpenAPI requests.

## M3 — Natural Voice Conversation ✅ (completed 2026-07-04)
Not a feature milestone — a latency and interruption-quality milestone. Pipeline
inspection found the dominant TTFA cost was Kokoro synthesizing an entire sentence
before any audio reached the speaker, which was also the largest gap in barge-in
responsiveness (no cancellation checkpoint mid-synthesis). ADR-018 adds streaming
TTS synthesis (`TTSEngine.synthesize_stream()`, additive/optional, default falls
back to one chunk via `synthesize()`); `KokoroTTS` implements it via kokoro-onnx's
native `create_stream()`. The orchestrator now plays audio chunk-by-chunk with an
epoch check between chunks, closing the TTS-blocking gap in barge-in. Also: a
lower first-sentence chunking threshold (earlier first sound), bounded/backpressured
token and sentence queues (no unnecessary buffering), a measured barge-in
audible-stop-latency metric (`BargeInLatencyMeasured`), the previously-defined-
but-never-emitted `SpeechFinished` event now published, richer runtime diagnostics
(queue depths, playback buffer seconds, barge-in count/latency — all additive to
`RuntimeSnapshot`, no new API endpoints per ADR-017), and Ctrl+C now exits cleanly
at every stage of `eva run` (model loading, audio startup, active conversation) and
every other CLI command via a top-level backstop.
Speculative LLM generation on unconfirmed partial transcripts was considered for
further TTFA reduction and explicitly deferred to M4+: it would add a second
speculative-cancellation path in the same milestone hardening the existing one —
worse risk/reward during a hardening pass.
**Exit met (automated):** 291 tests total (+27), including a 20-consecutive-
rapid-interruption stress test, double-barge-in and zero-delay-burst race tests,
bounded-queue backpressure tests (including a tight-bounds/short-timeout crash
guard), and a chunk-boundary playback-smoothness test proving streamed chunks
join without audible gaps. Full quality gate (ruff, mypy strict, pytest) green.
**Not yet exit-tested (needs the reference machine, not reproducible in this
environment):** the literal "<150 ms audible stop" and "20 consecutive real-mic
interruptions" targets, which need a real microphone/speaker and a stopwatch or
audio-level probe — the automated stress tests validate the *mechanism* (epoch
correctness, no leaks, no crashes) under adversarial timing with fake engines,
not the physical audio latency. Run `eva bench --rounds 3` and the manual
interruption protocol on the RTX 3060 Laptop / Ryzen 9 5900HX reference machine
before treating M3's product-facing exit criteria as fully met.

## M4 — Memory, Personalization & Intelligence ✅ (completed 2026-07-05)
New `eva/memory/` subsystem (ADR-019): `MemoryStore` + `UserProfileStore`
ports over one SQLite database (`conversations_dir/memory.db`, WAL mode,
numbered migrations, FTS5 text search with a LIKE fallback). New
`eva/embedding/` subsystem (ADR-020, ADR-010 amendment): `all-MiniLM-L6-v2`
via ONNX Runtime + `tokenizers` (no PyTorch), a new `kind="embedding"`
catalog entry, brute-force numpy cosine retrieval with recency decay +
pinned/favorite boosting — no vector database, real-measured and bounded
independent of history size (`retrieval_scan_limit`, default 2000
candidates). `ContextBuilder` (ADR-021): deterministic prompt composition
(persona + language + profile → relevant memories → summary → recent window
→ current utterance), every build inspectable via a `ContextTrace`.
Personas (ADR-022): registry-backed, mirroring the language-profile pattern,
6 built-ins + settings-persisted custom ones. User profiles: SQLite-backed,
separate from app `Settings` (multi-user-ready). Voices: `eva/tts/voices.py`
registry over existing TTS capability discovery. `LLMSummarizer` reuses the
existing `LLMEngine` port — no new ML dependency. Retention policy
(age + per-conversation cap, pinned-exempt). `RuntimeSnapshot` gains memory
diagnostics. Four new FastAPI routers (`/memory`, `/personas`, `/users`,
`/voices`), all ADR-017-compliant, additive to the existing API.
**Exit met:** 462 tests total (+171 since M3); conversation memory persists
across restarts via `eva serve`'s `/api/v1/memory/*` and `/conversation/*`
endpoints; semantic + keyword search, personas, and user profiles all
verified through both the port layer (SQLite adapter) and the API layer
(FastAPI `TestClient`); a real measured benchmark
(`eva.benchmark.memory.run_memory_benchmark`) shows retrieval + context
composition latency plateauing at ~60 ms regardless of total history size
once bounded by `retrieval_scan_limit` — not estimated, measured, and one
real N+1-query performance bug was found and fixed by that measurement
before it shipped. Deferred to M5+ (documented, not silently dropped):
`eva memory`/`eva user` CLI commands, real encryption-at-rest.

## M5 — Web UI
The platform API and WebSocket protocol are already built (M2.6); this
milestone is the React UI consuming them — no new backend surface expected,
only whatever small gaps using it in anger reveals. Conversation view with
live partial transcripts and token streaming, mic state, push-to-talk &
always-listening toggle. Full settings surface per ADR-009 (LM Studio / Open
WebUI / Home Assistant class), driven by the settings schema and component
registries. Sections:
- **General**: startup behavior, language, default interaction mode
- **Language Models / Speech Recognition / Speech Synthesis / Voice Detection**:
  per-subsystem model manager pages — installed & active models, size, quantization,
  context length, license, RAM/VRAM requirements, disk usage, compatibility,
  download/remove/set-default/benchmark; voice picker, speech speed/pitch,
  streaming toggle; VAD sensitivity, silence timeout, confidence threshold
- **Conversation**: history browser, context window, sampling (temperature, top-p,
  max tokens, stop sequences), export/import
- **Memory**: retention policy, summarization, clearing
- **Prompt Templates** and **Personalities**: registry-backed editors
- **Audio**: devices, gain/volume, AEC/NS/AGC, barge-in, push-to-talk, always-listening
- **Hardware**: detected CPU/GPU/RAM/VRAM/backend, capability tier, profile presets
  (Balanced / Fast / High Accuracy / Low Memory / Developer / Custom), thread count /
  GPU layers / batch / context / memory-limit overrides
- **Performance**: live graphs (CPU/GPU/VRAM/RAM, per-stage latencies, TTFT, TTFA,
  full-response time)
- **Plugins**: manager (list/enable/disable/install/remove) per ADR-011
- **Developer**: logs viewer, debug mode, benchmark suite, export logs, reload
  models/plugins, config viewer
- **Diagnostics**: hardware report, audio pipeline health, self-tests
- **Appearance**: dark/light/system theme, UI scaling
- **Accessibility**: reduced motion, keyboard navigation, captions-first mode
- **Privacy**: data locations, retention, offline guarantee statement
- **Updates**: manual engine/model update checks (offline-friendly)
**Exit:** full product usable and fully configurable from a browser at localhost;
API documented (OpenAPI + WebSocket protocol reference in docs/).

## M6 — Desktop app
pywebview shell, engine process supervision, tray + global PTT hotkey, first-run
setup wizard (profile pick + model download).
**Exit:** double-click launch to a working assistant on Windows and Linux dev boxes.

## M7 — Benchmarking & performance engineering
Benchmark harness: ASR (WER + latency on recorded fixtures), LLM (tokens/s, TTFT),
TTS (RTF, TTFA), end-to-end turn latency, memory/VRAM/CPU sampling; HTML/Markdown
report generator; profile-based optimization pass; re-validate default model picks
(Parakeet/Moonshine adapters land here if data justifies them).
**Exit:** reproducible benchmark reports; documented per-profile defaults; startup
time and interaction latency targets met or consciously re-set.

## M8 — Packaging & release
PyInstaller bundles, Inno Setup installer, AppImage, docs set (Installation, User,
Developer, Architecture, Contribution, Troubleshooting guides), 1.0.0 release.
**Exit:** a non-developer installs and talks to the assistant without touching Python.

## Deferred (architecture supports, not scheduled)
Plugin marketplace features (RAG, vision/OCR, filesystem, calendar, IoT/home
automation, optional web search), macOS builds (Metal via llama.cpp), voice cloning
profile (Chatterbox), wake-word activation, multilingual UI.

## Standing rules
- **Clean-environment smoke test is a per-milestone release gate:** a fresh venv +
  `pip install -e ".[dev]"` must yield a runnable app (commands either work or fail
  with actionable guidance — never `ModuleNotFoundError`). See ADR-013.
- Correctness, modularity, testability before micro-optimization (optimization has a
  dedicated milestone: M7).
- Every significant design decision gets a new or updated ADR.
- CHANGELOG.md, roadmap status, and affected docs are updated with every batch of
  changes, so the project state is always readable from the repository alone.
