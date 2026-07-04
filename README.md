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

Next milestone:

**M3 – Barge-In** (end-to-end voice interruption)

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


## How to Run:

Previously, our thesis project was just Python scripts. This new project is being built as a **Python package**, where `eva` is a console command created when the package is installed.

---

# Step 1 — Are we in the project folder?


Run:

```powershell
pwd
```

(or `Get-Location` in PowerShell)

---

# Step 2 — Is the virtual environment activated?

Do you see something like this?

```text
(.venv) PS C:\Downloads\edge-voice-assistant>
```

If **NOT**, activate it.

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Command Prompt:

```cmd
.venv\Scripts\activate.bat
```

After activation your prompt should become:

```text
(.venv) PS ...
```

---

# Step 3 — Install the package

Claude's README says:

```bash
pip install -e ".[dev]"
```

Run:

```powershell
pip install -e ".[dev]"
```

or

```powershell
python -m pip install -e ".[dev]"
```

---

# Step 4 — Verify

Run:

```powershell
eva --help
```

or

```powershell
eva version
```

If that works, then

```powershell
eva diagnose
```

---

# Step 4.5 — Download Models if running First Time!

Download the models

Run these commands:

eva models download qwen3.5-4b-instruct-q4_k_m

After that finishes:

eva models download kokoro-82m-v1.0


