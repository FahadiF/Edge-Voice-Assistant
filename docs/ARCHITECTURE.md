# Architecture — Edge Voice Assistant

Status: Accepted. Decisions are recorded in [adr/](adr/).

## 1. Design principles

1. **Barge-in first.** Interruption is not a feature bolted on top — the whole runtime
   is organized around *cancellable turns*. Anything that cannot be cancelled mid-flight
   is a design bug.
2. **Streaming everywhere.** No stage waits for the previous stage to finish completely.
3. **Ports and adapters.** The core engine depends only on abstract interfaces
   (`ASREngine`, `LLMEngine`, `TTSEngine`, `VADEngine`, `MemoryStore`, `Tool`,
   `AudioDevice`). Models are adapters; swapping one is a config change.
4. **One headless engine, many frontends.** CLI, web UI, and desktop app are thin
   clients over the same engine API (WebSocket + REST on localhost).
5. **Offline by construction.** The only network code lives in the model downloader.

## 2. System overview

```
┌────────────────────────────────────────────────────────────────────┐
│                        Frontends (thin clients)                    │
│   CLI (dev)      Web UI (React, localhost)     Desktop (pywebview) │
└───────────────┬────────────────────────────────────────────────────┘
                │ WebSocket (events/audio state) + REST (config/CRUD)
┌───────────────▼────────────────────────────────────────────────────┐
│                    Engine Server (FastAPI, asyncio)                │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │              Conversation Orchestrator (turn FSM)            │  │
│  │   turn epochs · cancellation · dialogue policy · memory      │  │
│  └──┬─────────┬──────────┬──────────┬──────────┬───────────────┘  │
│     │ports    │          │          │          │                   │
│  ┌──▼──┐  ┌──▼───┐  ┌───▼───┐  ┌───▼───┐  ┌──▼─────┐             │
│  │ VAD │  │ ASR  │  │  LLM  │  │  TTS  │  │ Memory │  Tools/     │
│  │port │  │ port │  │ port  │  │ port  │  │  port  │  Plugins    │
│  └──┬──┘  └──┬───┘  └───┬───┘  └───┬───┘  └──┬─────┘             │
│  Silero   faster-    llama.cpp   Kokoro    SQLite                 │
│  (ONNX)   whisper    (GGUF)      (+Piper)  (JSON export)          │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │        Audio Subsystem — full-duplex, single clock           │  │
│  │  duplex PortAudio stream (10 ms frames) → WebRTC APM         │  │
│  │  (AEC + NS + AGC, playback frames fed as far-end reference)  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│  Cross-cutting: settings · profiles · model manager · hardware     │
│  detection · metrics · structured logging · diagnostics            │
└────────────────────────────────────────────────────────────────────┘
```

## 3. The turn state machine and barge-in (the core mechanism)

Every user interaction is a **turn** with a monotonically increasing **epoch number**.
All artifacts flowing through the pipeline (audio frames, partial transcripts, LLM
token streams, synthesized sentences, playback buffers) are tagged with their epoch.

States: `IDLE → LISTENING → THINKING → SPEAKING → (LISTENING | IDLE)`

**Barge-in path** (target: audible stop < 150 ms after speech onset):

1. The mic is *never* muted. The duplex audio callback runs WebRTC APM: playback
   frames are fed as the far-end reference, so the echo of the assistant's own voice
   is subtracted from the mic signal before VAD ever sees it.
2. Silero VAD runs continuously on the echo-cancelled stream. During `SPEAKING`,
   a short speech-onset confirmation window (~200 ms of speech frames, tunable)
   triggers `barge_in()`.
3. `barge_in()` bumps the epoch. This single atomic action:
   - ramps playback down over ~40 ms (no click) and flushes the playback queue,
   - cancels the LLM token stream (asyncio cancellation → llama.cpp abort callback),
   - cancels pending TTS synthesis,
   - transitions to `LISTENING` **retaining the audio already captured** (ring buffer
     includes the pre-trigger frames, so "No, stop" is not lost — unlike the thesis).
4. Every consumer drops any item whose epoch < current. No stale replies can ever
   be spoken, no matter how fast the user interrupts repeatedly.

Fallback ladder (config): full-duplex AEC (default) → half-duplex mute-while-speaking
(if AEC unavailable/poor) → push-to-talk (always available).

## 4. Streaming pipeline (perceived-latency budget)

```
user stops speaking ──► endpoint detected (VAD, ~300–500 ms adaptive)
   ASR finalize (faster-whisper, partials already computed)      ~150–300 ms
   LLM prefill + first sentence tokens (llama.cpp, streaming)    ~300–600 ms
   TTS first sentence (Kokoro)                                   ~150–300 ms
──► FIRST AUDIO OUT                                       target ≤ 1.2 s
   ...while remaining sentences generate + synthesize in parallel
```

- **ASR**: audio is transcribed incrementally during the utterance (partials shown in
  UI); on endpoint only a small finalization pass remains.
- **LLM → TTS**: a *sentence chunker* consumes the token stream and emits speakable
  segments (sentence or clause boundaries, with a min/max length policy) to TTS.
- **TTS → playback**: synthesized segments queue into the playback ring; segment N+1
  synthesizes while N plays.
- Adaptive endpointing: the fixed ~1 s silence wait is replaced by a shorter base
  window that lengthens when the partial transcript looks incomplete (trailing
  conjunction/comma heuristic) — natural pauses without premature cut-offs.

## 5. Concurrency model

- **asyncio** event loop owns orchestration, the turn FSM, and the server API.
- Blocking inference (ASR decode, llama.cpp decode, TTS synth) runs in dedicated
  worker threads via `asyncio.to_thread` / executors, streaming results back through
  `asyncio.Queue`s. Each worker checks the epoch between chunks → prompt cancellation.
- The audio callback (PortAudio thread) is real-time-safe: no allocation, no locks —
  it only moves frames between lock-free ring buffers. APM/VAD run on a consumer
  thread, not in the callback.
- GPU discipline on 6 GB: LLM owns the GPU; ASR runs int8 (GPU when idle VRAM allows,
  else CPU); TTS and VAD run on CPU by default (Kokoro is faster than real-time on CPU).

## 6. Module layout (src layout, installable package)

Packages follow ADR-010: one package per subsystem, each owning its port
(abstract interface), its registry, and its built-in adapters — the tree itself
communicates the pipeline.

```
edge-voice-assistant/
├── pyproject.toml            # installable package; ruff, mypy, pytest config
├── src/eva/                  # "Edge Voice Assistant" engine
│   ├── core/                 # pure domain: turn FSM, epochs, events, errors,
│   │                         #   registry primitive — imports nothing else in eva
│   ├── audio/                # duplex stream, APM (AEC/NS/AGC), ring buffers,
│   │                         #   playback, device enumeration
│   ├── vad/                  # VADEngine port + registry + adapters (silero)
│   ├── asr/                  # ASREngine port + registry + adapters (faster-whisper)
│   ├── llm/                  # LLMEngine port + registry + adapters (llama.cpp)
│   ├── tts/                  # TTSEngine port + registry + adapters (kokoro, piper)
│   ├── conversation/         # orchestrator, history, sentence chunker,
│   │                         #   prompt-template + personality registries
│   ├── memory/               # MemoryStore port + registry + SQLite adapter,
│   │                         #   conversation import/export
│   ├── tools/                # Tool port + registry (function-calling tools)
│   ├── plugins/              # plugin SDK: manifest, discovery, lifecycle (ADR-011)
│   ├── models/               # model manager: catalog, download, verify, licenses,
│   │                         #   disk usage, compatibility, hot-swap
│   ├── hardware/             # detection + profile presets
│   ├── config/               # settings schema, persistence, app paths
│   ├── benchmark/            # benchmark suite + report generation
│   ├── metrics/              # per-stage latency, resource sampling, diagnostics
│   ├── server/               # FastAPI app: REST + WebSocket (the API-first boundary)
│   └── cli.py                # headless/dev interface
├── web/                      # React + Vite web UI (consumes the API only)
├── desktop/                  # desktop shell + tray/launcher (consumes the API only)
├── tests/                    # unit + integration (fake adapters, recorded audio)
├── packaging/                # PyInstaller specs, Inno Setup, AppImage recipe
└── docs/                     # architecture, ADRs, guides, API reference
```

**Dependency direction:** `core` ← subsystems ← `conversation` ← `server` ← UIs.
Subsystems may import `core` and `config` only — never each other's adapters.
Business logic stays in engine services; `web/`, `desktop/`, and the CLI are pure
API consumers, so future clients (mobile app, third-party integrations) require
no engine changes.

## 7. Default model stack (6 GB VRAM profile)

| Stage | Default | Why | Footprint |
|---|---|---|---|
| VAD | Silero VAD v5 (ONNX, CPU) | Still SOTA for size; proven in thesis | ~2 MB |
| AEC/NS | WebRTC APM (livekit `rtc.apm` or webrtc-audio-processing) | Battle-tested full-duplex AEC | CPU, negligible |
| ASR | faster-whisper `small` int8 (GPU) / `base` (CPU fallback) | 4× whisper speed, mature, multilingual | ~0.5 GB |
| LLM | Qwen3-4B-Instruct GGUF Q4_K_M via llama.cpp | Best quality/VRAM at 4B; native streaming + abort | ~2.8 GB + KV |
| TTS | Kokoro-82M (CPU) | Apache-2.0, faster than real-time on CPU, strong quality | ~0.4 GB RAM |
| TTS alt | Piper (low-end CPU) / Chatterbox (voice cloning, GPU) | Profile options | — |

**Profiles** have two layers. Hardware detection produces a *capability tier*
(`cpu-only`, `gpu-6gb`, `gpu-12gb`); each tier maps to goal-oriented *presets* —
**Balanced** (default), **Fast**, **High Accuracy**, **Low Memory**, **Developer** —
that select a concrete model combination. Presets are registry entries (ADR-010):
users can create, edit, export, and share **Custom** profiles from the UI.
Alternates (Parakeet, Moonshine, SenseVoice ASR; other GGUF LLMs; Chatterbox TTS)
plug in as adapters.

See ADR-002…ADR-005 for full rationale and rejected alternatives.

## 8. Engine API (frontend contract)

- **WebSocket** `/ws`: bidirectional event stream. Server → client events are the
  engine's typed event set (authoritative definitions in `eva/core/events.py`,
  implemented M2): `StateChanged`, `SpeechStarted`, `SpeechFinished`,
  `BargeInDetected`, `PartialTranscript`, `FinalTranscript`, `LlmStarted`,
  `LlmToken`, `LlmSentence`, `LlmFinished`, `TtsStarted`, `TtsAudioReady`,
  `TtsFinished`, `TurnStarted`, `TurnFinished`, `TurnCancelled`; client →
  `set_mode(ptt|always)`, `ptt_down/up`, `interrupt`, `text_message` (typed input).
- **REST**: settings CRUD, profiles/personas, model list/download/switch,
  conversations (list/export/import), diagnostics, logs.
- Audio I/O stays in the engine process (server owns the sound devices); frontends
  only render state. This keeps the web UI trivial and audio latency out of the browser.

## 9. Quality & testing strategy

- Unit tests with fake adapters (scripted ASR/LLM/TTS) — the turn FSM and barge-in
  logic are tested with zero models loaded, including the nasty races
  (interrupt during prefill / during synth / during playback / double interrupt).
- Integration tests with recorded WAV fixtures driving the pipeline offline.
- ruff (lint+format), mypy (strict on `core`/`ports`), pytest, GitHub Actions CI
  (lint + tests on Windows and Linux runners; model-free).
- Structured logging (`structlog`-style JSON option), per-stage latency metrics
  persisted for the benchmark reports.
