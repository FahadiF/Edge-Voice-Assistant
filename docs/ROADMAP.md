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

## M4.5 Integration & validation pass ✅ (completed 2026-07-05)
Real-hardware testing exposed integration gaps that unit tests with fakes
could not: a release-blocking multi-system-message crash (fixed by
refactoring `ContextBuilder` to emit exactly one system message with strict
user/assistant alternation, enforced by `validate_chat_messages` — works
across all supported chat templates with no model-specific hacks), assistant
identity (introduces itself as "Edge Voice Assistant"), personas actually
influencing replies, full CLI parity (`eva personas/users/voices/memory/
profile`), persona/profile/voice visibility in the startup banner and
diagnostics, and the first version of `docs/MANUAL_TESTING.md`.

## M5 — Web UI & Desktop Shell ✅ shipped

The platform API and WebSocket protocol were already built (M2.6); this
milestone is the React + TypeScript UI consuming them — no new backend
surface beyond the one anticipated gap (serving the built UI, ADR-023).
Delivered:
- **Dashboard**: live assistant/engine state, mic level, active models/
  persona/profile/voice, memory stats, latency, resources — all pushed over
  the WebSocket, zero polling
- **Conversation**: streaming transcript (partials → finals → tokens),
  interruption markers, timestamps, search, export/import/clear
- **Memory**: search, pin/favorite/forget, archive/restore/merge/delete,
  summarize, context-inspector (exact composed prompt + retrieval trace),
  delete-all with typed confirmation
- **Personas**: list/activate/create/edit/duplicate/delete + prompt preview
- **User Profiles**: create/switch/edit/delete + client-side import/export
- **Models**: full model cards (provider/license/languages/VRAM/RAM/size),
  download with live WebSocket progress, activate/remove
- **Voices**: search/filter by language+style, in-browser PCM preview
  (Web Audio API, no container format), select + persists
- **Settings**: fully schema-driven (ADR-009) from `GET /settings/schema` —
  every section, field, bound, and description comes from the schema
- **Diagnostics**: live resource meters + sparklines, queue depths, event
  log, hardware summary
- **Plugins**: list/enable/disable/reload (ADR-011)
- **Desktop shell** (`eva-desktop`, optional `[desktop]` extra): a minimal
  `pywebview` window hosting the same UI — landed a milestone early, scoped
  down (window only; see M6 for the rest)

**Exit:** full product usable and fully configurable from a browser at
localhost, and from a native desktop window; API documented (OpenAPI +
WebSocket protocol reference in docs/). ✅ Both met.

## M5.1 — Presentation layer ✅ (completed 2026-07-05)
Markdown is the canonical text form everywhere (ADR-024): the web UI renders
it (react-markdown + remark-gfm, GFM tables/lists/code), and a stateful
`MarkdownSpeechFilter` at the TTS boundary converts it to natural speech —
no asterisks or backticks are ever spoken, including markers split across
sentence-chunker segments. Plus a frontend review pass: WebSocket reconnect
cache invalidation, error surfacing, accessibility, autoscroll, race fixes.

## M5.2 — Conversational intelligence ✅ (completed 2026-07-06)
Prompt composition rewritten as an explicit hierarchy (ADR-021 amendment):
identity → conversation guidance → capability honesty → persona →
language/profile → summary → memories → technical facts. Fixes validated
live: continuity across short follow-ups, honest "not enabled in this
build" for image requests, helpfulness over literalness, stronger persona
differentiation (Teacher persona added), natural identity answers, quality
maintained in long conversations. Conversational-quality regression tests
guard the composition.

## M5.3 — Capability polish ✅ (completed 2026-07-06)
Markdown-to-speech hardened (entities, nesting, unpaired-marker scrub).
Chat composer: typed text turns flow through the same orchestrator event
queue as voice (`POST /conversation/say`), Enter/Shift+Enter, attachment
placeholders. Permissions settings page with named toggles wired to real
behavior, and a permission-gated local system-information provider
(date/time, hardware) so the assistant can answer "what time is it?"
offline (ADR-025).

## M5.4 — Production readiness ✅ (completed 2026-07-08)
Long-term memory actually used in replies: embeddings computed at write
time, keyword fallback when the embedding model is absent, end-to-end
recall verified ("my nickname is …" → recalled in a later conversation).
Permissions regrouped (General/Files/Devices/Tools/Privacy) with a
settings-schema migration (v1→2). Conversation titles: auto-generated,
editable, persisted, exported/imported. Memory page UX, sticky composer,
streaming polish, markdown regression cases, full UI + backend review.

## M5.5 — Stability, lifecycle & performance ✅ (completed 2026-07-10)
Engine lifecycle and supervision (ADR-026): parallel model preload with
per-component progress events (GPU order LLM→ASR preserved; ~18 s cold
start vs ~22 s serial), optional lazy TTS loading, graceful shutdown
(ordered, exception-proof), a cancellation-architecture fix (single-owner-
thread stream driving — eliminates "generator already executing"), a named
background-task manager, supervised ASR/TTS crash recovery (one turn lost,
never the assistant; cooldown-guarded), and background server process
management: `eva start`, `eva stop`, `eva restart`, `eva status`,
`eva logs`. Composer gains a dedicated Stop control beside mic/send.
636 tests total.

## v0.5 — Documentation synchronization ✅ (completed 2026-07-11)
Maintenance pass, not a feature milestone: all documentation synchronized
with the shipped M5.x state, private team notes moved out of `docs/` into
an untracked `.dev/` folder, version bumped to `0.5.0a1`.

## M5.6 — Final hardening, UX & production readiness ✅ (completed 2026-07-12)

The last M5 milestone — no new capabilities, everything M5 promised made
production-quality before M6 begins:

- Continue stored conversations (`POST /conversation/resume` + Memory-page
  "Continue"), preserving id, context, summaries, and title.
- Bounded, clean shutdown everywhere: `eva serve` Ctrl+C ≤ ~5 s with the
  web UI connected; `eva stop` goes through `POST /system/shutdown`
  (graceful) before falling back to terminate.
- Mic-permission-off now runs a playback-only audio stream — typed chats
  speak, and the input device is never opened.
- Time-to-first-audio: Kokoro warm-up at load + first-segment clause
  splitting in the chunker.
- Non-English pronunciation: conversation language wired through the TTS
  port to Kokoro's espeak phonemizer.
- Hardening: SQLite shared-connection locking, download size/SHA-256
  verification, WebSocket Origin validation, documentation sync.

## M6 — Desktop polish (in progress)
Tray icon, global push-to-talk hotkey, engine process supervision (auto-
restart on crash, single-instance lock), first-run setup wizard as a native
window (profile pick + model download), and installers (PyInstaller +
Inno Setup / AppImage, per ADR-008). The window-hosting mechanics
themselves (`eva-desktop`, `pywebview`) already shipped in M5 — M6 is the
remaining desktop-specific polish around that shell. Delivered as gated
sub-milestones (ADR-027):
- **M6.1 ✅** server supervision (attach-or-spawn + bounded-backoff restart),
  window-state persistence, `DesktopSettings` section.
- **M6.2 ✅** system tray (`DesktopPlatform` port + pystray adapter;
  supervisor-state icon/menu; push-not-poll status), window lifecycle
  (close/minimize-to-tray, restore, left-click activation), and
  **background-safe minimize**: the WebView2 renderer is kept unthrottled while
  hidden (measured — Chromium otherwise clamps hidden-tab timers to ~1 Hz and
  eventually freezes the page, stalling the live UI/WebSocket in the tray) so
  minimizing is "hidden only" — engine, streaming, and WS continue and restore
  is instant. Belt-and-suspenders: the web client force-reconnects the WS on
  becoming visible.
- **M6.3** global hotkey · **M6.4** first-run wizard
  (React `/welcome`, ADR-028) · **M6.5** notifications + crash recovery +
  autostart/single-instance · **M6.6** Windows installer.
**Exit:** double-click launch to a working assistant on Windows and Linux dev boxes.

### M6 UX backlog — investigated 2026-07-21 (see CHANGELOG for the measurements)

- **A/B. Sentence-level speak-while-generating / inter-sentence gaps —
  investigated, mostly already correct.** Measured with real per-sentence
  timing against the running engine: the orchestrator already runs LLM
  generation, sentence chunking, and TTS synthesis concurrently, and the
  playback buffer's lead *grows* through a reply (proof there's no queueing
  stall). The remaining, real cost is Kokoro's per-sentence CPU synthesis time
  (~2.5-3.2s for a typical sentence, linear in text length — not a fixed
  per-call overhead, so coalescing sentences would not help and would delay
  first audio). A small, safe win (trimming ~40-100ms of measured genuine
  edge-silence per sentence boundary) was implemented. **True TTS-call
  parallelism (synthesizing N+1 while N is still being computed) was tested
  and crashes** — `KokoroTTS` holds one shared, non-thread-safe
  kokoro-onnx/phonemizer instance; two concurrent `synthesize_stream()` calls
  corrupt it. Any future work to overlap synthesis calls needs either (a) a
  small pool of independent `KokoroTTS`/model instances (real memory/load-time
  cost, needs its own design) or (b) revisiting the CPU-only TTS decision
  (ADR-004/012/018) — neither undertaken here without that trade-off being
  explicitly chosen. Remaining latency is CPU-inference-speed-bound, the same
  class of environmental constraint as the earlier LLM GPU-throttling finding
  — a candidate for M7's benchmarking/optimization pass, not a code defect.
- **C. Event Log copy/export — done.** Copy all / Export .txt / Export .log /
  Clear log added to the Diagnostics page.

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
