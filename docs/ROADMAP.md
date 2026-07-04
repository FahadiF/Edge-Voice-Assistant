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

## M3 — Barge-in complete (product priority #1)
Epoch cancellation wired through every stage (LLM abort, TTS cancel, playback ramp,
buffer retention of the interrupting speech). Repeated-interruption stress tests,
race-condition test suite, half-duplex + push-to-talk fallbacks.
**Exit:** "No, stop" mid-reply audibly stops playback < 150 ms, the utterance is
transcribed without repetition, and 20 consecutive rapid interruptions leave a clean state.

## M4 — Conversation engine
SQLite conversation memory, history windowing + summarization, personas/system-prompt
profiles, settings manager surfaces, conversation export/import (JSON), text
normalization pre-TTS, multiple voices.
**Exit:** persistent multi-session memory; switchable personalities and voices.

## M5 — Server API + Web UI
FastAPI + WebSocket protocol (versioned), React UI. Conversation view with live
partial transcripts and token streaming, mic state, push-to-talk & always-listening
toggle. Full settings surface per ADR-009 (LM Studio / Open WebUI / Home Assistant class),
driven by the settings schema and component registries. Sections:
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
- Correctness, modularity, testability before micro-optimization (optimization has a
  dedicated milestone: M7).
- Every significant design decision gets a new or updated ADR.
- CHANGELOG.md, roadmap status, and affected docs are updated with every batch of
  changes, so the project state is always readable from the repository alone.
