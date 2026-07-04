![Version](https://img.shields.io/github/v/release/FahadiF/Edge-Voice-Assistant?include_prereleases)
![Python](https://img.shields.io/badge/python-3.12-blue)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-success)
![License](https://img.shields.io/badge/License-Apache%202.0-blue).

# Edge Voice Assistant

An offline AI voice assistant designed for natural, low-latency conversations on local hardware.

Edge Voice Assistant runs entirely on your computer after the required AI models are installed. It combines local speech recognition, language models, and speech synthesis into a modular voice platform that prioritizes privacy, responsiveness, and extensibility.

## Highlights

- Fully offline after initial model installation
- Natural voice conversations with streaming speech recognition and response generation
- Real-time voice interruption (barge-in)
- Modular architecture with interchangeable AI models
- Desktop and local web interfaces powered by the same backend
- Cross-platform support for Windows and Linux
- Built for developers, researchers, and edge AI applications

## Project Vision

Edge Voice Assistant is designed as a long-term open-source platform for local AI interaction.

The architecture is intentionally modular so that language models, speech recognition engines, text-to-speech engines, memory systems, plugins, and future multimodal capabilities can evolve independently.

## Current Status

The project is under active development.

Completed milestones:

- **M0 – Project Foundation**
- **M1 – Audio Foundation** (full-duplex audio, echo cancellation, voice activity detection)
- **M2 – Streaming Pipeline** (streaming speech recognition, language model, and speech
  synthesis with turn management and cancellation; `eva run` provides a spoken
  conversation from the command line)
- **M2.5 – Production Hardening** (persisted configuration, model presets,
  multilingual foundation, developer diagnostics)
- **M2.6 – Platform API** (FastAPI + WebSocket backend at `eva serve` — the
  CLI, and eventually the desktop and web apps, are all clients of the same
  engine; see [docs/API.md](docs/API.md))
- **M3 – Natural Voice Conversation** (streaming TTS synthesis for lower
  time-to-first-audio and faster interruption, richer runtime diagnostics,
  clean shutdown on Ctrl+C at every stage)

Next milestone:

**M4 – Conversation Engine** (persistent memory, personas, multiple voices)

See the project roadmap for implementation progress.

## Hardware Targets

Primary development platform

- NVIDIA RTX 3060 Laptop GPU (6 GB VRAM)
- 16 GB RAM

The application is designed to scale across different hardware profiles and automatically recommend suitable AI models.

## Documentation

| Document | Description |
|----------|-------------|
| Architecture | Overall system architecture |
| Roadmap | Development milestones |
| ADRs | Architecture Decision Records |
| Developer Guide | Development workflow |
| API Reference | Backend APIs |
| Plugin SDK | Extension development |
| Benchmark Guide | Performance evaluation |

## Project Background

Edge Voice Assistant is the production successor to the research prototype developed during my Master's thesis at the University of Vaasa.

The original thesis implementation is preserved separately as a historical research artifact while this repository focuses on long-term product development.
