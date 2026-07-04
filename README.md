![Version](https://img.shields.io/github/v/release/FahadiF/Edge-Voice-Assistant?include_prereleases)
![Python](https://img.shields.io/badge/python-3.12-blue)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-success)
![License](https://img.shields.io/github/license/FahadiF/Edge-Voice-Assistant)

# Edge Voice Assistant

An offline AI voice assistant designed for natural, low-latency conversations entirely on local hardware.

Edge Voice Assistant combines local speech recognition, language models, and speech synthesis into a modular AI platform that prioritizes **privacy**, **responsiveness**, **extensibility**, and **full local execution**.

Once the required AI models are installed, the assistant operates completely offline without requiring cloud APIs or Internet connectivity.

---

## Highlights

- Fully offline after initial model installation
- Natural real-time voice conversations
- Streaming speech recognition and response generation
- Voice interruption (barge-in) with fast cancellation
- Modular AI architecture with interchangeable engines
- Local REST + WebSocket API
- Cross-platform support (Windows & Linux)
- Automatic hardware detection and model recommendations
- Designed for developers, researchers, and edge AI applications

---

## Project Vision

Edge Voice Assistant is designed as a long-term open-source platform for conversational AI on edge devices.

Rather than being tied to a single AI model or technology stack, the architecture is intentionally modular so that every subsystem can evolve independently.

Current and future interchangeable components include:

- Speech Recognition (ASR)
- Large Language Models (LLMs)
- Text-to-Speech (TTS)
- Voice Activity Detection (VAD)
- Memory systems
- Plugins
- Personas
- Desktop UI
- Web UI
- Future multimodal capabilities

The long-term goal is to provide a production-quality foundation for private, local-first conversational AI.

---

# Current Status

The project is under active development.

## Completed Milestones

- ✅ **M0 – Project Foundation**
- ✅ **M1 – Audio Foundation**
  - Full-duplex audio
  - Echo cancellation
  - Voice activity detection
  - Audio diagnostics

- ✅ **M2 – Streaming AI Pipeline**
  - Streaming ASR
  - Local LLM
  - Streaming TTS
  - Turn management
  - Cancellation
  - Offline conversation (`eva run`)

- ✅ **M2.5 – Production Hardening**
  - Persistent configuration
  - Hardware profiles
  - Model presets
  - Multilingual foundation
  - Diagnostics
  - Guided first-run experience

- ✅ **M2.6 – Platform API**
  - FastAPI backend
  - REST API
  - WebSocket events
  - Plugin framework
  - Shared engine architecture

- ✅ **M3 – Natural Voice Conversation**
  - Faster interruption
  - Lower time-to-first-audio
  - Streaming speech synthesis
  - Improved runtime diagnostics
  - Graceful shutdown

---

## Current Development

🚧 **M4 – Conversation Engine**

Planned features include:

- Persistent conversation memory
- Personas
- Multiple assistant voices
- Long-term memory
- Memory retrieval
- Context management

See the [Roadmap](docs/ROADMAP.md) for future milestones.

---

# Hardware Targets

Primary development platform

- NVIDIA RTX 3060 Laptop GPU (6 GB VRAM)
- AMD Ryzen 9 5900HX
- 16 GB RAM

The assistant automatically detects available hardware and recommends suitable AI models and runtime configurations.

Designed to scale from CPU-only systems to modern consumer GPUs.

---

# Documentation

| Document | Description |
|----------|-------------|
| [Installation Guide](docs/INSTALLATION.md) | Installation and first-time setup |
| [Architecture](docs/ARCHITECTURE.md) | System architecture overview |
| [Roadmap](docs/ROADMAP.md) | Development roadmap |
| [API Reference](docs/API.md) | REST & WebSocket API |
| [Architecture Decision Records](docs/adr/) | Design decisions |
| [GitHub Releases](../../releases) | Release history |

---

# Research Background

Edge Voice Assistant originated from my Master's thesis research in **Sustainable and Autonomous Systems** at the **University of Vaasa**.

The original research prototype has been preserved separately as a historical implementation, while this repository represents the long-term open-source continuation of that work.

**Original thesis repository**

https://github.com/FahadiF/Modular-Software-Implementation-Edge-Voice-Chatbot

The goal of this repository is to evolve the research prototype into a modular, extensible, production-oriented platform for offline conversational AI.

---

# Acknowledgements

This project began during my Master's thesis at the **University of Vaasa**.

I would like to express my sincere gratitude to my thesis supervisor,

**Jani Boutellier**  
https://github.com/jboutell

for his guidance, valuable feedback, and support throughout the research that inspired this project.

I am also grateful to the open-source community and the developers of the projects that make modern local AI possible, including:

- llama.cpp
- faster-whisper
- Kokoro ONNX
- Silero VAD
- ONNX Runtime
- FastAPI
- CTranslate2

---

# Contributing

Contributions, discussions, bug reports, and feature requests are welcome.

As the project continues to evolve, contributions that improve performance, modularity, documentation, accessibility, and hardware compatibility are especially appreciated.

Please open an Issue before submitting large architectural changes.

---

# License

This project is licensed under the **MIT License**.

See the [LICENSE](LICENSE) file for details.
