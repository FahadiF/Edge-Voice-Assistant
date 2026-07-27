# Architecture — Edge Voice Assistant

Status: Accepted. Decisions are recorded in [adr/](adr/).

## 1. Design principles

1. **Barge-in first.** Interruption is not a feature bolted on top — the whole runtime
   is organized around *cancellable turns*. Anything that cannot be cancelled mid-flight
   is a design bug.
2. **Streaming everywhere.** No stage waits for the previous stage to finish completely.
3. **Ports and adapters.** The core engine depends only on abstract interfaces
   (`ASREngine`, `LLMEngine`, `TTSEngine`, `VADEngine`, `MemoryStore`,
   `UserProfileStore`, `MemoryRetriever`, `EmbeddingProvider`, `Summarizer`,
   `AudioOutput`). Models are adapters; swapping one is a config change.
   *A `Tool`/capability port is designed but not implemented — see §10.*
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
│  (ONNX)   whisper    (GGUF)      (ONNX)    (JSON export)          │
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
   - cancels pending TTS synthesis at the next streamed chunk boundary (M3/ADR-018:
     `TTSEngine.synthesize_stream()` yields sub-sentence chunks, so a stale turn is
     dropped mid-sentence rather than only between sentences),
   - transitions to `LISTENING` **retaining the audio already captured** (ring buffer
     includes the pre-trigger frames, so "No, stop" is not lost — unlike the thesis).
4. Every consumer drops any item whose epoch < current. No stale replies can ever
   be spoken, no matter how fast the user interrupts repeatedly.

Fallback ladder (config): full-duplex AEC (default) → half-duplex mute-while-speaking
(if AEC unavailable/poor) → push-to-talk (always available).

## 4. Streaming pipeline (perceived-latency budget)

**Measured** on the reference laptop (RTX 3060 6 GB, Windows 11), not estimated:

```
user stops speaking
   endpoint detected (VAD, fixed 800 ms silence window)           800 ms
   ASR finalize (faster-whisper small, int8 on GPU)            ~250 ms
   LLM prefill + first clause (llama.cpp, streaming)          ~1650 ms
   TTS first clause (Kokoro, CPU)                             ~1630 ms
──► FIRST AUDIO OUT                                     measured ~3.5 s
   ...while remaining sentences generate + synthesize in parallel
```

Two costs dominate and both are addressable: speech synthesis runs on **CPU** because the
CPU build of ONNX Runtime is a base dependency, and prompt prefill re-processes the whole
context each turn because volatile content sits at the head of the prompt, defeating
llama.cpp's KV prefix cache. Both are scheduled performance work; see ROADMAP.

- **ASR**: audio is transcribed incrementally during the utterance (partials shown in
  UI); on endpoint only a small finalization pass remains.
- **LLM → TTS**: a *sentence chunker* consumes the token stream and emits speakable
  segments (sentence or clause boundaries, with a min/max length policy) to TTS.
- **TTS → playback**: synthesized segments queue into the playback ring; segment N+1
  synthesizes while N plays. Since M3 (ADR-018), synthesis itself is chunked below
  the sentence level where the engine supports it (Kokoro via kokoro-onnx's native
  phoneme-batch streaming) — the first audio for a sentence reaches the speaker
  after the first chunk, not the whole sentence.
- **Endpointing is currently a fixed silence window** (`vad.silence_timeout_ms`,
  default 800 ms). Adaptive endpointing — a shorter base window that lengthens when the
  partial transcript looks incomplete — is designed but **not implemented**.

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
│   ├── tts/                  # TTSEngine port + registry + adapters (kokoro);
│   │                         #   voices.py: voice registry over engine capability
│   │                         #   discovery (M4, ADR-022)
│   ├── embedding/            # EmbeddingProvider port + registry + ONNX adapter
│   │                         #   (M4, ADR-020) — a memory building block, not
│   │                         #   memory-specific (ADR-010 amendment)
│   ├── conversation/         # orchestrator, history (turn pairing only — M4
│   │                         #   moved storage/composition to memory/context_builder),
│   │                         #   sentence chunker, language + persona registries,
│   │                         #   context_builder.py (deterministic prompt composition)
│   ├── memory/               # MemoryStore + UserProfileStore ports + registry +
│   │                         #   SQLite adapter (one db, WAL, FTS5), NumpyMemoryRetriever,
│   │                         #   LLMSummarizer, retention policy (M4, ADR-019/020)
│   ├── plugins/              # plugin SDK: manifest, discovery, lifecycle (ADR-011).
│   │                         #   NOTE: capability wiring is not yet implemented —
│   │                         #   plugins can be listed/toggled but cannot register
│   ├── models/               # model manager: catalog, download, verify, licenses,
│   │                         #   disk usage, compatibility, hot-swap
│   ├── hardware/             # detection + profile presets
│   ├── config/               # settings schema, persistence, app paths
│   ├── benchmark/            # benchmark suite + report generation
│   ├── metrics/              # per-stage latency, resource sampling, diagnostics
│   ├── server/               # FastAPI app: REST + WebSocket (the API-first boundary);
│   │                         #   server/static.py mounts the built web UI when present (M5, ADR-023)
│   ├── desktop/              # pywebview shell, tray, window state, server
│   │                         #   same FastAPI app on a thread, opens one native window at it
│   └── cli.py                # headless/dev interface — one file, one subparser group
│                             #   per concern (models, profiles, config, personas, users,
│                             #   voices, memory, profile — M4 integration pass), each a
│                             #   thin client of the same services the API routers call
├── web/                      # React + TypeScript + Vite web UI (M5, ADR-023) — talks to
│   │                         #   /api/v1/* and the WebSocket only, never an eva.* import
│   └── src/
│       ├── api/              # typed REST client (client.ts, endpoints.ts, types.ts —
│       │                     #   a hand-maintained mirror of the pydantic schemas)
│       ├── ws/                # WebSocket connection + zustand live-state store
│       ├── theme/             # design tokens + dark/light/system ThemeProvider
│       ├── components/        # shared UI (Layout, SchemaForm, dialogs, toasts)
│       └── pages/              # one page per M5 part: Dashboard, Conversation, Memory,
│                               #   Personas, Users, Models, Voices, Settings, Diagnostics, Plugins
├── tests/                    # unit + integration (fake adapters, recorded audio)
├── packaging/                # PyInstaller specs, Inno Setup, AppImage recipe
└── docs/                     # architecture, ADRs, guides, API reference
```

**Dependency direction:** `core` ← subsystems ← `conversation` ← `server` ← UIs.
Subsystems may import `core` and `config` only — never each other's adapters —
with one documented exception (ADR-010 amendment, M4): `memory` imports
`embedding`'s port and registry, a genuine building-block relationship (turning
text into a vector is not memory-specific), never the reverse. Business logic
stays in engine services; `web/`, `desktop/`, and the CLI are pure API
consumers, so future clients (mobile app, third-party integrations) require no
engine changes. The FastAPI server itself gains one narrow rendering
responsibility in M5 (ADR-023): serving the *built* `web/` output as a static
SPA when one exists — this is not business logic, and the app is byte-for-byte
the old API-only app when no build is present.

## 7. Default model stack (6 GB VRAM profile)

| Stage | Default | Why | Footprint |
|---|---|---|---|
| VAD | Silero VAD v5 (ONNX, CPU) | Still SOTA for size; proven in thesis | ~2 MB |
| AEC/NS | WebRTC APM (livekit `rtc.apm` or webrtc-audio-processing) | Battle-tested full-duplex AEC | CPU, negligible |
| ASR | faster-whisper `small` int8 (GPU) / `base` (CPU fallback); `large-v3-turbo` selectable | 4× whisper speed, mature, multilingual | ~0.5 GB (turbo: 1.1 GB measured) |
| LLM | Qwen3.5-4B-Instruct GGUF Q4_K_M via llama.cpp | Best quality/VRAM at 4B; native streaming + abort | ~3.5 GB incl. KV (measured) |
| TTS | Kokoro-82M (CPU) | Apache-2.0, faster than real-time on CPU, strong quality | ~0.4 GB RAM |
| Embeddings | all-MiniLM-L6-v2 (ONNX, CPU) | Semantic memory retrieval (ADR-020) | ~90 MB |

**Profiles** have two layers. Hardware detection produces a *capability tier*
(`cpu-only`, `gpu-6gb`, `gpu-12gb`); each tier maps to goal-oriented *presets* —
**Balanced** (default), **Fast**, **High Accuracy**, **Low Memory**, **Developer** —
that select a concrete model combination. Presets are registry entries (ADR-010):
users can create, edit, export, and share **Custom** profiles from the UI.
Additional engines (Parakeet/Moonshine ASR, other GGUF LLMs, alternative TTS) would
plug in as adapters; none ship today. Only the stack in the table above is implemented.

See ADR-002…ADR-005 for full rationale and rejected alternatives.

## 8. Engine API (frontend contract) — implemented M2.6 (ADR-017)

- **WebSocket** `/api/v1/ws`: server → client event stream — the same typed
  events the orchestrator has always published (`eva/core/events.py`):
  `StateChanged`, `SpeechStarted`, `PartialTranscript`, `FinalTranscript`,
  `LlmStarted/Token/Sentence/Finished`,
  `TtsStarted/AudioReady/SentenceStarted/Finished`,
  `TurnStarted/Finished/Cancelled`, `BargeInDetected`, plus
  `ModelDownloadProgress/Completed/Failed` and `EngineStarted/Stopped`. An
  initial `snapshot` message on connect means clients never poll for state.
  Client → server control is REST, not WebSocket messages (see `docs/API.md`).
- **REST** (`/api/v1`): settings (get/put/patch/validate/reset + JSON Schema),
  models (list/info/download/remove/activate), diagnostics (`RuntimeSnapshot`),
  plugins (list/enable/disable/reload), conversation (history/current/
  interrupt/cancel/clear/export/import), engine (status/readiness/start/stop),
  system/health. Full endpoint map in `docs/API.md`; OpenAPI/Swagger UI is
  generated automatically at `/docs`.
- Audio I/O stays in the engine process (server owns the sound devices); frontends
  only render state. This keeps the web UI trivial and audio latency out of the browser.
- The engine does not start when the server process starts — `POST
  /api/v1/engine/start` is explicit, so `eva serve` never opens a microphone
  or loads models as a side effect of being run.

## 9. Quality & testing strategy

- Unit tests with fake adapters (scripted ASR/LLM/TTS) — the turn FSM and barge-in
  logic are tested with zero models loaded, including the nasty races
  (interrupt during prefill / during synth / during playback / double interrupt).
- Integration tests with recorded WAV fixtures driving the pipeline offline.
- ruff (lint+format), mypy (**strict, whole package**), pytest, GitHub Actions CI
  (lint + type check + tests on Windows and Linux runners; model-free).
  837 tests as of v0.6.0-alpha; hardware/model-dependent tests are marked
  `integration` and excluded from CI.
- Structured logging (optional JSON output), per-stage latency metrics exposed through
  the diagnostics snapshot. An opt-in per-sentence pipeline trace
  (`EVA_CONVERSATION_TRACE=1`) logs every streaming hand-off on one turn-relative clock.

## 10. Known architectural gaps

Recorded here so the difference between the design and the implementation is never
something a contributor has to discover by reading source. Each is scheduled; see
[ROADMAP.md](ROADMAP.md).

| Gap | Current state | Consequence |
|---|---|---|
| **No capability/tool port** | The `Tool` port in §1 is designed, not implemented | `permissions.tools.*`, `files.*`, and `general.internet` gate nothing, because nothing exists to gate. Blocks tools, online mode, and vision. |
| **Plugin capability wiring** | Discovery, manifests, and enable/disable work (ADR-011); registering a plugin's `contributes` into the subsystem registries does not | Plugins can be listed and toggled but cannot add capabilities |
| **`managed_by="engine"` model lifecycle** | Engine-managed weights (faster-whisper) have no install detection, prefetch, removal, or integrity verification | ASR models always display as "not installed"; first-load downloads are silent and unbounded |
| **LLM port assumes local weights** | `LLMEngine` exposes `load`/`unload`/`device` | A remote or server-backed provider cannot implement the port honestly. Blocks the provider abstraction. |
| **Auto-summarization not wired** | `LLMSummarizer` and `summarize_after_turns` exist; nothing invokes them during a live conversation | Long conversations grow the prompt until the recent-turn window truncates it |
| **Settings is one flat document** | Single pydantic model, strict keys | No structure for per-provider configuration, credentials, or fallback chains |
| **No secret storage** | None anywhere in the project | Prerequisite for any authenticated provider |

**Offline-by-construction is currently enforced by convention, not by test.** The only
network code lives in the model downloader, but nothing verifies that. Before any online
capability is introduced, that invariant needs an automated guard.
