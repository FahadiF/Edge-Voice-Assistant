# Architecture Diagrams

Visual companions to `docs/ARCHITECTURE.md`.
These diagrams describe the system **as implemented through M5.5 (v0.5)**
unless a diagram is explicitly labeled "Future" — future diagrams describe
intended, not-yet-built shapes and should not be mistaken for current
behavior.

All diagrams are Mermaid; they render natively in GitHub, GitLab, and most
modern Markdown viewers.

## 1. Overall System Architecture

```mermaid
flowchart TB
    subgraph Clients["Clients (thin, no business logic)"]
        CLI["eva CLI\n(eva run / eva start / eva serve / ...)"]
        WebUI["Web UI — React + TS (web/)\nbuilt, served statically by the API\n(SPA fallback, ADR-023)"]
        Desktop["Desktop shell (eva-desktop)\nminimal pywebview window\n(tray/hotkey — M6)"]
        Plugin["Third-party Plugin\n(none exist yet)"]
    end

    subgraph API["Platform API (eva.server) — ADR-017"]
        FastAPI["FastAPI app\n/api/v1/*"]
        WS["WebSocket\n/api/v1/ws"]
        State["ServerState\n(engine lifecycle owner)"]
    end

    subgraph Engine["Engine (eva.engine, eva.conversation)"]
        Orchestrator["Turn Orchestrator\n(asyncio, turn epochs)"]
        Bus["EventBus"]
    end

    subgraph Subsystems["Subsystem Packages (ports + registries)"]
        VAD["eva.vad"]
        ASR["eva.asr"]
        LLM["eva.llm"]
        TTS["eva.tts"]
        Audio["eva.audio"]
        Memory["eva.memory\n(SQLite store, retriever,\nsummarizer — M4)"]
        Embedding["eva.embedding\n(ONNX MiniLM — M4)"]
        Models["eva.models\n(ModelManager)"]
        Hardware["eva.hardware"]
        Config["eva.config"]
    end

    CLI -->|direct calls| Engine
    CLI -->|direct calls| Subsystems
    CLI -->|"eva start/stop:\nPID-file process mgmt\n(eva.service, M5.5)"| API
    WebUI -->|HTTP + WS| API
    Desktop -->|HTTP + WS, hosts the built UI| API
    Plugin -.->|future: eva.sdk facade| Engine

    FastAPI --> State
    WS --> Bus
    State --> Orchestrator
    State --> Models
    State -->|"plugin discovery"| Plugin

    Orchestrator --> VAD
    Orchestrator --> ASR
    Orchestrator --> LLM
    Orchestrator --> TTS
    Orchestrator --> Audio
    Orchestrator --> Memory
    Orchestrator --> Bus
    Memory --> Embedding

    VAD --> Config
    ASR --> Config
    LLM --> Config
    TTS --> Config
    Audio --> Hardware
    Models --> Hardware
```

## 2. Module Dependency Graph

```mermaid
flowchart BT
    Core["eva.core\n(errors, registry, events, turn)\nDEPENDS ON NOTHING ELSE IN eva"]

    Config["eva.config"]
    Hardware["eva.hardware"]
    Audio["eva.audio"]
    VAD["eva.vad"]
    ASR["eva.asr"]
    LLM["eva.llm"]
    TTS["eva.tts"]
    Models["eva.models"]
    Runtime["eva.runtime"]
    Plugins["eva.plugins"]
    Metrics["eva.metrics"]
    MemoryPkg["eva.memory"]
    EmbeddingPkg["eva.embedding"]

    Conversation["eva.conversation"]

    Engine["eva.engine"]
    Onboarding["eva.onboarding"]
    Benchmark["eva.benchmark"]

    Server["eva.server"]
    CliMod["eva.cli"]
    VoiceLoop["eva.voice_loop"]
    Service["eva.service\n(PID-file process mgmt)"]
    DesktopMod["eva.desktop"]

    Config --> Core
    Hardware --> Core
    Audio --> Core
    Audio --> Config
    VAD --> Core
    ASR --> Core
    ASR --> Config
    LLM --> Core
    TTS --> Core
    Models --> Core
    Models --> Hardware
    Runtime --> Core
    Runtime --> Hardware
    Plugins --> Core
    Metrics --> Core
    Metrics --> Hardware
    MemoryPkg --> Core
    MemoryPkg --> Config
    EmbeddingPkg --> Core

    Conversation --> Core
    Conversation --> Config
    Conversation --> VAD
    Conversation --> ASR
    Conversation --> LLM
    Conversation --> TTS
    Conversation --> Audio
    Conversation --> MemoryPkg
    MemoryPkg --> EmbeddingPkg

    Engine --> Conversation
    Engine --> Models
    Onboarding --> Runtime
    Onboarding --> Models
    Onboarding --> Hardware
    Benchmark --> ASR
    Benchmark --> LLM
    Benchmark --> TTS

    Server --> Engine
    Server --> Onboarding
    Server --> Models
    Server --> Plugins
    Server --> Metrics

    CliMod --> Engine
    CliMod --> Onboarding
    CliMod --> Server
    CliMod --> Models
    CliMod --> Hardware
    CliMod --> Service

    VoiceLoop --> Engine
    DesktopMod --> Server

    style Core fill:#2d5,stroke:#333
```

**Rule visualized:** nothing points sideways within the "one port +
registry" row (`VAD`/`ASR`/`LLM`/`TTS` never point at each other) — all
cross-subsystem coordination happens one layer up, in `eva.conversation`.

## 3. Voice / Audio Pipeline (steady state, no interruption)

```mermaid
flowchart LR
    Mic(["Microphone"]) --> Duplex["Duplex PortAudio Stream\n(one device clock)"]
    Speaker(["Speakers"]) --- Duplex
    Duplex -->|mic frame, 10ms| APM["WebRTC APM\n(AEC + NS + AGC)"]
    Duplex -.->|far-end reference, 10ms| APM
    APM -->|cleaned frame| Chunker["Frame Chunker\n(10ms → 512-sample)"]
    Chunker --> VADEngine["Silero VAD"]
    VADEngine -->|speech probability| Segmenter["Speech Segmenter\n(pure logic, pre-roll ring buffer)"]
    Segmenter -->|SpeechStart / UtteranceEnd / BargeIn / UtteranceProgress| Orchestrator["Turn Orchestrator"]

    Orchestrator -->|utterance audio| ASREngine["faster-whisper"]
    ASREngine -->|transcript| Orchestrator
    Orchestrator -->|messages| LLMEngine["llama.cpp\n(streaming tokens)"]
    LLMEngine -->|tokens| SentenceChunker["Sentence Chunker"]
    SentenceChunker -->|speakable segment| TTSEngine["Kokoro"]
    TTSEngine -->|PCM| PlaybackQueue["Playback Queue\n(fade-out on interrupt)"]
    PlaybackQueue --> Duplex
```

## 4. Conversation Sequence (one normal turn)

```mermaid
sequenceDiagram
    participant User
    participant Segmenter as Speech Segmenter
    participant Orch as Turn Orchestrator
    participant ASR
    participant LLM
    participant Chunker as Sentence Chunker
    participant TTS
    participant Audio as Playback Queue
    participant Bus as EventBus

    User->>Segmenter: speaks
    Segmenter->>Orch: SpeechStart
    Orch->>Bus: publish SpeechStarted
    User->>Segmenter: (stops speaking, silence timeout)
    Segmenter->>Orch: UtteranceEnd(audio)
    Orch->>Orch: advance epoch → N
    Orch->>Bus: publish TurnStarted(N)

    Orch->>ASR: transcribe(audio)  [worker thread]
    ASR-->>Orch: text
    Orch->>Bus: publish FinalTranscript(N, text)

    Orch->>LLM: stream(messages, should_abort)  [producer thread]
    loop per token
        LLM-->>Orch: token
        Orch->>Bus: publish LlmToken(N, token)
        Orch->>Chunker: feed(token)
        alt sentence boundary reached
            Chunker-->>Orch: segment
            Orch->>Bus: publish LlmSentence(N, segment)
            Orch->>TTS: synthesize(segment)  [worker thread]
            TTS-->>Orch: pcm
            Orch->>Audio: say(pcm)
            Orch->>Bus: publish TtsAudioReady(N) [first segment only]
        end
    end
    Orch->>Bus: publish LlmFinished(N)
    Audio-->>User: hears reply (streaming, overlapped with generation)
    Orch->>Bus: publish TtsFinished(N), TurnFinished(N)
```

## 5. Barge-in Sequence (interruption)

```mermaid
sequenceDiagram
    participant User
    participant Segmenter as Speech Segmenter
    participant Orch as Turn Orchestrator
    participant LLM
    participant TTS
    participant Audio as Playback Queue
    participant Bus as EventBus

    Note over Orch,Audio: Turn epoch N is active; assistant is speaking
    User->>Segmenter: starts speaking over playback
    Segmenter->>Segmenter: accumulate speech during playback
    Note over Segmenter: barge_in_confirm_ms reached (default 200ms)
    Segmenter->>Orch: BargeIn(speech_ms)
    Orch->>Bus: publish BargeInDetected(epoch=N)

    Orch->>Orch: advance epoch → N+1 (ALL epoch-N work is now stale)
    Orch->>Audio: stop_speaking() → fade out ~40ms, flush queue
    Orch->>LLM: cancel (should_abort() now true for epoch N)
    Orch->>TTS: in-flight synthesis result discarded on stale epoch check
    Orch->>Bus: publish TurnCancelled(epoch=N, reason="barge-in")
    Orch->>Bus: publish StateChanged(state="listening")

    Note over Segmenter: pre-roll ring buffer already retained the\ninterrupting speech — it is NOT lost
    Segmenter->>Orch: UtteranceEnd(interrupting_audio) [epoch N+1]
    Orch->>Orch: start new turn normally (see diagram 4)
```

**Key invariant:** no artifact tagged with epoch N is ever spoken or acted on
after the epoch advances to N+1 — every producer/consumer checks staleness at
its next natural checkpoint (per LLM token, per TTS segment, per playback
poll). This is the single mechanism behind `"barge-in"`, `"superseded"`,
`"shutdown"`, and `"manual"` (API-triggered) cancellation — see
`TurnCancelled.reason`.

## 6. Engine Lifecycle

```mermaid
stateDiagram-v2
    [*] --> NotBuilt: eva serve starts\n(no engine yet — explicit start required)
    NotBuilt --> Building: POST /api/v1/engine/start\n(or `eva run`)
    Building --> Building: readiness check\n(eva.onboarding.check_readiness)
    Building --> NotBuilt: readiness failed\n→ 409 with problems list
    Building --> Loading: build_assistant()\n(resolve settings → model files → engines)
    Loading --> Loading: preload() — parallel (ADR-026)\nLLM strictly before ASR (GPU order,\nADR-015); TTS/embedding concurrent;\nComponentLoadStarted/Finished progress events
    Loading --> AudioStarting: start_audio()\n(opens duplex stream)
    AudioStarting --> Running: orchestrator.run() task scheduled
    Running --> Running: turns processed\n(see diagrams 4 & 5)
    Running --> Running: supervised recovery (ADR-026)\nASR/TTS crash → unload+reload in background\n(cooldown-guarded; one turn lost, not the engine)
    Running --> Stopping: POST /api/v1/engine/stop\n(or Ctrl+C in `eva run`, or `eva stop`)
    Stopping --> Stopping: ordered shutdown —\ncancel owned tasks (TaskManager),\norchestrator shutdown (TurnCancelled reason="shutdown"),\naudio stop, engines unloaded; exception-proof
    Stopping --> NotBuilt: assistant discarded
    NotBuilt --> [*]: process exits
```

## 7. Model Manager Workflow

```mermaid
flowchart TD
    Start(["eva models download <id>\nor POST /models/{id}/download"]) --> Lookup["ModelManager.info(id)\n(catalog lookup)"]
    Lookup -->|unknown id| Err404["RegistryError → 404"]
    Lookup -->|managed_by=bundled| Skip["nothing to do\n(ships with a dependency, e.g. Silero)"]
    Lookup -->|managed_by=engine| SkipEngine["nothing to do here —\nengine downloads its own\nweights on first load()"]
    Lookup -->|managed_by=manager| Check["already installed?"]
    Check -->|yes| Done1(["done, no-op"])
    Check -->|no| Download["for each ModelFile:\nHTTP Range resume from .part\nverify bytes == Content-Length"]
    Download -->|success| Rename["atomic rename .part → final filename"]
    Download -->|failure/incomplete| ErrIncomplete["ModelError\n('incomplete', keeps .part for resume)"]
    Rename --> Done2(["installed"])

    subgraph describe["ModelManager.describe(id, settings) — the full model card"]
        direction LR
        Meta["catalog metadata\n(name, version, provider,\nlicense, VRAM/RAM, quantization)"]
        InstallState["is_installed(), disk_usage_mb()"]
        Compat["compatibility check vs.\ndetected hardware VRAM/RAM"]
        Active["is this id currently\nactive in settings?"]
    end
    Done2 -.-> describe
```

## 8. First-Run Onboarding Flow

```mermaid
flowchart TD
    Entry(["eva run (no --yes)\nor eva first-run"]) --> Persist{"settings.json\nexists?"}
    Persist -->|no| Resolve["resolve_and_persist_settings()\n→ recommend_profile(detect_hardware())\n→ apply_preset() → save"]
    Persist -->|yes| BuildPlan
    Resolve --> BuildPlan["build_plan(settings, paths)\n→ SetupPlan: variant, missing models,\n  size/time estimate"]
    BuildPlan --> Complete{"plan.is_complete?"}
    Complete -->|yes| StartNow(["start immediately,\nno wizard shown"])
    Complete -->|no| Welcome["print welcome screen:\nhardware, runtime, models,\nestimated size/time"]
    Welcome --> Confirm{"interactive TTY\nand user confirms?"}
    Confirm -->|no, non-interactive & no --yes| Block(["exit non-zero,\nprint plan + remedy commands"])
    Confirm -->|declined by user| Cancelled(["exit 0,\n'run eva first-run when ready'"])
    Confirm -->|yes| Steps["execute steps in order:\n1. install LLM runtime (eva.runtime)\n2. download each missing model\n3. verify (re-check readiness)"]
    Steps -->|any step fails| Friendly["friendly error:\nwhat failed / why / how to fix\n(NEVER a raw traceback)"]
    Steps -->|all succeed| SaveState["SetupState(completed=True).save()"]
    SaveState --> StartNow
```

## 9. Plugin Architecture (current: backend only, no loading)

```mermaid
flowchart TB
    subgraph Installed["Installed Python Packages"]
        PkgA["some-plugin-package\n(hypothetical — none exist today)"]
    end

    PkgA -->|declares entry point| EntryPoints["importlib.metadata.entry_points\n(group='eva.plugins')"]
    EntryPoints --> Manager["PluginManager.discover()"]
    Manager -->|ep.load()| Factory["zero-arg callable"]
    Factory -->|returns| Manifest["PluginManifest\n(id, name, version, license,\ncontributes, permissions)"]
    Manifest --> State["PluginState\n(manifest, enabled, healthy, error)"]
    State --> API["GET /api/v1/plugins\nenable/disable/reload"]

    State -.->|"NOT YET IMPLEMENTED:\nactually loading contributions\ninto engine registries"| Registries["eva.vad / eva.asr / eva.llm / eva.tts\nregistries, eva.conversation\npersona/template registries"]
    Factory -.->|"NOT YET DESIGNED:\nnarrow eva.sdk facade\n(open design question, ADR-011)"| EngineInternals["Engine internals\n(orchestrator, ports)"]

    style Registries stroke-dasharray: 5 5
    style EngineInternals stroke-dasharray: 5 5
```

## 10. FastAPI / API Architecture

```mermaid
flowchart TB
    Uvicorn["uvicorn\n(eva serve)"] --> App["FastAPI app\n(eva.server.app.create_app)"]
    App --> CORS["CORSMiddleware\n(localhost/127.0.0.1 only)"]
    App --> ErrorHandlers["exception handlers\n(EvaError → status code,\nValidationError → 422)"]
    App --> StateInit["app.state.eva = ServerState(paths)"]

    App --> RouterSystem["/health, /system/hardware"]
    App --> RouterSettings["/settings\nGET/PUT/PATCH/validate/reset/schema"]
    App --> RouterModels["/models\nlist/get/download/remove/activate"]
    App --> RouterDiagnostics["/diagnostics"]
    App --> RouterPlugins["/plugins\nlist/get/enable/disable/reload"]
    App --> RouterEngine["/engine\nstatus/readiness/start/stop"]
    App --> RouterConversation["/conversation\nhistory/current/say/interrupt/\ncancel/clear/export/import"]
    App --> RouterMemory["/memory\nsearch/stats/turns/conversations/\ncontext-preview/export/import (M4)"]
    App --> RouterPersonas["/personas (M4)"]
    App --> RouterUsers["/users (M4)"]
    App --> RouterVoices["/voices + preview (M4)"]
    App --> RouterWS["/ws (WebSocket)"]
    App --> StaticUI["Static web UI mount at /\n(SPA fallback, ADR-023 —\nonly when web/dist exists)"]

    RouterSettings -->|StateDep| ServerState["ServerState\n(single engine-lifecycle owner)"]
    RouterModels -->|StateDep| ServerState
    RouterDiagnostics -->|StateDep| ServerState
    RouterPlugins -->|StateDep| ServerState
    RouterEngine -->|StateDep| ServerState
    RouterConversation -->|StateDep| ServerState
    RouterMemory -->|StateDep| ServerState
    RouterPersonas -->|StateDep| ServerState
    RouterUsers -->|StateDep| ServerState
    RouterVoices -->|StateDep| ServerState
    RouterWS --> ServerState

    ServerState --> ConfigService["eva.config.service\n(shared with CLI)"]
    ServerState --> ModelManager["eva.models.ModelManager\n(shared with CLI)"]
    ServerState --> PluginManagerRef["eva.plugins.PluginManager"]
    ServerState --> Assistant["eva.engine.Assistant\n(built only on /engine/start)"]
    ServerState --> EventBusRef["eva.core.events.EventBus"]

    App -.->|auto-generated| OpenAPI["/openapi.json, /docs (Swagger UI)"]
```

## 11. Event Bus Flow

```mermaid
flowchart LR
    subgraph Producers
        Orchestrator["Turn Orchestrator\n(most events)"]
        ServerStateProd["ServerState\n(EngineStarted/Stopped,\nModelDownload*)"]
    end

    Producers -->|publish / publish_threadsafe| Bus["EventBus\n- bounded per-subscriber queues (256)\n- drops OLDEST on overflow\n- keeps bounded history (100 events)"]

    Bus --> Sub1["CLI renderer\n(eva run — voice_loop.py)"]
    Bus --> Sub2["WebSocket client 1"]
    Bus --> Sub3["WebSocket client 2\n(fan-out: every client\nsees every event)"]

    NewClient["New WebSocket connection"] -->|on connect| Snapshot["send {type:'snapshot', data: RuntimeSnapshot}"]
    Snapshot --> Sub2
```

## 12. Desktop ↔ Backend ↔ Engine Communication (partially built — M6 completes it)

```mermaid
flowchart LR
    subgraph DesktopProcess["Desktop App Process (pywebview)"]
        NativeShell["Native window shell\n(tray, global hotkey)"]
        WebView["Embedded web view\n(same built UI as the web app)"]
    end

    subgraph BackendProcess["Backend Process (eva serve, supervised by the shell)"]
        FastAPIProc["FastAPI app"]
        EngineProc["Assistant / Orchestrator\n(only after /engine/start)"]
    end

    NativeShell -->|spawns + monitors subprocess| BackendProcess
    WebView -->|HTTP fetch, same-origin| FastAPIProc
    WebView -->|WebSocket| FastAPIProc
    FastAPIProc --> EngineProc
    EngineProc -->|PortAudio| Hardware["Microphone / Speakers"]

    NativeShell -.->|global PTT hotkey| WebView
    WebView -.->|POST /conversation/interrupt| FastAPIProc
```

**Status: partially built.** The backend, the web UI content, and a
minimal window shell (`eva-desktop`: in-process server + pywebview window)
shipped in M5. The tray icon, global push-to-talk hotkey, and
subprocess-supervision shape shown above are M6 scope (the supervision
primitives — PID file, spawn, health poll, terminate — already exist in
`eva.service`).

## 13. Future v1.0 Architecture (aspirational — not current state)

```mermaid
flowchart TB
    subgraph UserFacing["User-Facing Surfaces — Web UI EXISTS (M5); installers are M8"]
        Web["Web UI\n(browser, localhost) — built"]
        Desk["Desktop App\n(window shell built; tray/hotkey/\ninstaller — M6/M8)"]
    end

    subgraph PlatformAPI["Platform API — EXISTS (M2.6–M5.5)"]
        API["FastAPI /api/v1\n+ WebSocket"]
    end

    subgraph EngineCore["Engine Core — EXISTS (M2–M5.5)"]
        Orch["Turn Orchestrator\n+ persistent memory, personas (M4)\n+ lifecycle supervision (M5.5)"]
    end

    subgraph PluginEcosystem["Plugin Ecosystem — FUTURE (post-v1.0)"]
        SDK["eva.sdk facade\n(designed, not built)"]
        Vision["Vision / OCR plugin"]
        RAG["RAG / documents plugin"]
        IoT["IoT / home automation plugin"]
        Marketplace["Plugin marketplace UI"]
    end

    subgraph Packaging["Packaging — FUTURE (M8)"]
        Installer["Windows Installer / AppImage"]
    end

    UserFacing --> PlatformAPI --> EngineCore
    EngineCore -.-> SDK
    SDK -.-> Vision
    SDK -.-> RAG
    SDK -.-> IoT
    Marketplace -.-> SDK
    Installer -.-> UserFacing

    style PluginEcosystem stroke-dasharray: 5 5
    style Packaging stroke-dasharray: 3 3
```
