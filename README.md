![Version](https://img.shields.io/github/v/release/FahadiF/Edge-Voice-Assistant?include_prereleases)
![Python](https://img.shields.io/badge/python-3.12-blue)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-success)
![License](https://img.shields.io/badge/License-Apache%202.0-blue)

# Edge Voice Assistant

An offline AI voice assistant designed for natural, low-latency conversations on local hardware.

Edge Voice Assistant runs entirely on your computer after the required AI models are installed. It combines local speech recognition, language models, and speech synthesis into a modular AI platform that prioritizes **privacy**, **responsiveness**, and **extensibility**.

Once the required models are installed, the assistant operates completely offline without requiring cloud APIs or Internet connectivity.

---

## Highlights

- Fully offline after initial model installation
- Natural voice conversations with streaming speech recognition and response generation
- Real-time voice interruption (barge-in)
- Text and voice input through a full web UI (chat composer + microphone)
- Persistent conversation memory with semantic recall, personas, and user profiles
- Modular architecture with interchangeable AI models
- Local REST + WebSocket API
- Fine-grained permission controls (microphone, memory, system information)
- Background server lifecycle (`eva start/stop/restart/status/logs`)
- Cross-platform support for Windows and Linux
- Automatic hardware detection and model recommendations
- Built for developers, researchers, and edge AI applications

---

## Project Vision

Edge Voice Assistant is designed as a long-term open-source platform for local conversational AI.

The architecture is intentionally modular so that language models, speech recognition engines, text-to-speech engines, memory systems, plugins, desktop applications, web applications, and future multimodal capabilities can evolve independently.

The long-term goal is to provide an extensible foundation for private, local-first conversational AI that can adapt to future AI models and hardware.

---

## Current Status

The project is under active development.

### Completed milestones

- **M0 – Project Foundation**

- **M1 – Audio Foundation**
  - Full-duplex audio
  - Echo cancellation
  - Voice activity detection
  - Audio diagnostics

- **M2 – Streaming Pipeline**
  - Streaming speech recognition
  - Local language model
  - Streaming speech synthesis
  - Turn management
  - Cancellation
  - Offline conversation (`eva run`)

- **M2.5 – Production Hardening**
  - Persistent configuration
  - Hardware profiles
  - Model presets
  - Multilingual foundation
  - Developer diagnostics
  - Guided first-run experience

- **M2.6 – Platform API**
  - FastAPI backend
  - REST API
  - WebSocket events
  - Plugin framework
  - Shared engine architecture

- **M3 – Natural Voice Conversation**
  - Streaming TTS
  - Lower time-to-first-audio
  - Faster interruption
  - Rich runtime diagnostics
  - Graceful shutdown
  
  **M4 – Conversation Engine**

  - Persistent memory
  - Personas
  - Multiple voices
  - Context management

- **M4 Integration & Validation Pass**
  - Fixed assistant identity (introduces itself as "Edge Voice Assistant",
    reveals the underlying model only on an explicit technical question)
  - Full CLI parity: `eva personas`, `eva users`, `eva voices`, `eva
    memory`, `eva profile`
  - Persona/user-profile/voice state visible in the startup banner and
    diagnostics
  - [Manual testing guide](docs/MANUAL_TESTING.md) for the whole milestone

- **M5 – Web UI & Desktop Shell**
  - React + TypeScript web UI (`web/`) covering the dashboard, conversation,
    memory, personas, user profiles, models, voices, settings, diagnostics,
    and plugins — every capability M4 shipped, now reachable without the
    CLI or raw HTTP
  - Fully schema-driven settings UI (ADR-009) — no hardcoded field lists
  - Live updates over the existing WebSocket event stream — no polling
  - Minimal `pywebview` desktop shell (`eva-desktop`) hosting the same UI

- **M5.1 – M5.4 – Experience & Production Readiness**
  - Markdown rendered in the UI and converted to natural speech at the
    TTS boundary
  - Rewritten prompt composition: stronger personas, capability honesty,
    conversational continuity
  - Chat composer with typed text turns sharing the voice pipeline
  - Grouped permission settings (General, Files, Devices, Tools, Privacy)
    that actually gate behavior
  - Long-term memory integrated into replies (semantic recall with
    keyword fallback), conversation titles, memory manager improvements

- **M5.5 – Stability, Lifecycle & Performance**
  - Parallel model loading with per-component startup progress (~18 s cold
    start on the reference machine)
  - Graceful shutdown and a hardened cancellation architecture
  - Automatic recovery when speech recognition or synthesis crashes
  - Background server commands: `eva start`, `eva stop`, `eva restart`,
    `eva status`, `eva logs`

### Next milestone

**M6 – Desktop polish** (tray icon, global push-to-talk hotkey, engine
process supervision, first-run wizard window, installers)

See the [Roadmap](docs/ROADMAP.md) for implementation progress.

## Starting, stopping, and exiting

Two ways to run the assistant — pick the one that matches what you're doing.

**Development — foreground, stop with Ctrl+C:**

```bash
cd web && npm install && npm run build   # once: builds web/dist/
eva serve --open   # API + web UI in the foreground; opens a browser
eva run            # or: voice-only console loop, no server
```

Stop with **Ctrl+C**. Shutdown is bounded (≤ ~5 s even with browser tabs
still connected) and always ends cleanly: the turn in flight is cancelled,
components stop in order, no tracebacks, no orphan processes. For UI work
with live reload, run `eva serve` in one terminal and `cd web && npm run
dev` in another (it proxies `/api` to the backend).

**Background / production — start, stop, restart, status:**

```bash
eva start          # background server at http://127.0.0.1:8765/ (no window)
eva status         # process, API, and engine state
eva restart        # stop + start (e.g. after changing models)
eva logs           # tail the newest log file
eva stop           # graceful: engine stops, audio released, DB flushed,
                   # then the process exits
```

`eva stop` asks the background server to shut down over the API first (the
clean path) and only falls back to terminating the process if the API
doesn't answer.

For the desktop shell: `pip install -e ".[desktop]"` then `eva-desktop`.

---

## Hardware Targets

Primary development platform

- NVIDIA RTX 3060 Laptop GPU (6 GB VRAM)
- AMD Ryzen 9 5900HX
- 16 GB RAM

The application is designed to scale across different hardware profiles and automatically recommend suitable AI models for each system.

---

## Documentation

| Document | Description |
|----------|-------------|
| [Installation](docs/INSTALLATION.md) | Installation and first-time setup |
| [Architecture](docs/ARCHITECTURE.md) | Overall system architecture |
| [API Reference](docs/API.md) | REST & WebSocket API |
| [Roadmap](docs/ROADMAP.md) | Development milestones |
| [Architecture Decision Records](docs/adr/) | Architecture Decision Records (ADRs) |

---

## Research Background

Edge Voice Assistant originated from my Master's thesis research in **Sustainable and Autonomous Systems** at the **University of Vaasa**.

The original thesis implementation has been preserved separately as a historical research artifact, while this repository represents the long-term open-source continuation of that work.

**Original thesis repository**

https://github.com/FahadiF/Modular-Software-Implementation-Edge-Voice-Chatbot

---

## Acknowledgements

This project began during my Master's thesis at the **University of Vaasa**.

I would like to express my sincere gratitude to my thesis supervisor,

**Jani Boutellier**  
https://github.com/jboutell

for his guidance, valuable feedback, and support throughout the research that inspired this project.

I am also grateful to the open-source community and the developers behind projects such as **llama.cpp**, **faster-whisper**, **Kokoro ONNX**, **Silero VAD**, **ONNX Runtime**, **FastAPI**, and **CTranslate2**, whose work makes modern local AI accessible to everyone.

---

## Contributing

Contributions, bug reports, feature requests, and discussions are welcome.

If you plan to make significant architectural changes, please open an issue first so we can discuss the proposed design before implementation.

---

## License

Licensed under the **Apache License 2.0**.

See the [LICENSE](LICENSE) file for details.
