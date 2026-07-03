# Changelog

Notable changes to Edge Voice Assistant. The format follows
[Keep a Changelog](https://keepachangelog.com/); versioning follows SemVer from the
first release onward.

## [Unreleased]

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
