# Installation Guide

Edge Voice Assistant runs fully offline after a one-time setup.

## Quickest start (guided)

```bash
git clone <your-repo-url> edge-voice-assistant
cd edge-voice-assistant
python -m venv .venv && .venv\Scripts\Activate.ps1   # Linux: source .venv/bin/activate
pip install -e .
eva run
```

On first run, `eva run` detects that setup is incomplete and launches a guided
wizard: it shows your hardware, the recommended runtime, and the models to
download (with sizes and a time estimate), asks once to continue, then installs
everything and starts the assistant. No further commands or documentation
needed. `eva first-run` launches the same wizard on demand.

The manual, stage-by-stage flow below is for developers and advanced users who
want to control each step.

`eva run` is the terminal voice loop. For the full experience — chat with
text or voice, manage memory, personas, models, and settings — use the
**web UI** (see [Web UI](#web-ui-and-background-server) below).

---

## Manual installation (developer flow)

Installation has three stages:

1. Install the application and its base dependencies (ASR, TTS, audio, VAD).
2. Install the LLM runtime for your hardware (`eva setup`).
3. Download the models (`eva models download`).

`eva doctor` verifies all three at any time.

> **Why three stages?** The language-model runtime (`llama-cpp-python`) publishes
> no packages on PyPI — only a source archive that would require a C++ compiler.
> Prebuilt CPU and CUDA wheels live on the llama.cpp wheel index instead, and
> `eva setup` selects the correct one for your machine automatically (see
> [ADR-013](adr/ADR-013-llm-runtime-installation.md)). The base install therefore
> stays compiler-free and cross-platform.

---

## Requirements

- **Python 3.12** (64-bit)
- **OS:** Windows 10/11 or Linux (Ubuntu 22.04+/equivalent)
- **RAM:** 16 GB recommended
- **GPU (optional):** NVIDIA GPU with 6 GB+ VRAM for GPU-accelerated inference.
  CPU-only works and is fully supported.
- **Disk:** ~4 GB for the default model set.

---

## Windows

```powershell
# 1. Clone and enter the project
git clone <your-repo-url> edge-voice-assistant
cd edge-voice-assistant

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Install the application (base dependencies: ASR, TTS, audio, VAD)
python -m pip install --upgrade pip
pip install -e ".[dev]"          # omit [dev] for a user (non-developer) install

# 4. Install the LLM runtime for your hardware
eva setup                        # auto-detects CPU vs NVIDIA CUDA

# 5. Download the models (~3 GB)
eva models download qwen3.5-4b-instruct-q4_k_m
eva models download kokoro-82m-v1.0

# 6. Verify and run
eva doctor
eva run
```

If PowerShell blocks the activation script, run once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

---

## Linux

```bash
# System audio library (PortAudio) — required by sounddevice
sudo apt-get update && sudo apt-get install -y libportaudio2

git clone <your-repo-url> edge-voice-assistant
cd edge-voice-assistant

python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e ".[dev]"

eva setup
eva models download qwen3.5-4b-instruct-q4_k_m
eva models download kokoro-82m-v1.0

eva doctor
eva run
```

---

## The LLM runtime (`eva setup`)

`eva setup` detects your hardware and installs the matching `llama-cpp-python`
build from the llama.cpp wheel index:

| Command | Installs |
|---|---|
| `eva setup` | Auto: CUDA build if an NVIDIA GPU is present, otherwise CPU |
| `eva setup --cpu` | Force the CPU build |
| `eva setup --cuda` | Force the CUDA build (+ NVIDIA cudart/cuBLAS wheels) |
| `eva setup --dry-run` | Print the exact pip command without running it |

### Manual alternative

If you prefer to install it yourself:

```bash
# CPU
pip install -e ".[cpu]"  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# CUDA (NVIDIA)
pip install -e ".[cuda]" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

---

## Verifying the installation

```
eva doctor
```

Example of a fully set-up system:

```
Runtime dependencies
--------------------
[ok     ] numpy            numerics
[ok     ] sounddevice      audio input/output
[ok     ] livekit          echo cancellation / noise suppression
[ok     ] pysilero_vad     voice activity detection
[ok     ] faster_whisper   speech recognition (ASR)
[ok     ] kokoro_onnx      speech synthesis (TTS)
[ok     ] llama_cpp        language model (LLM)

Models
------
[ok     ] qwen3.5-4b-instruct-q4_k_m
[ok     ] kokoro-82m-v1.0

All checks passed. `eva run` is ready.
```

Any `MISSING` line is printed with the exact command to fix it.

---

## Web UI and background server

The web UI is a React app in `web/` served statically by the API server.
Building it requires Node.js 20+ (a one-time step; end-user packages will
ship it prebuilt in a later release):

```bash
cd web
npm install
npm run build        # outputs web/dist
cd ..

eva start            # start the server in the background
# open http://127.0.0.1:8765/
eva status           # process, API, and engine status
eva logs             # tail the newest log file
eva stop             # graceful shutdown
```

`eva serve --open` runs the server in the foreground and opens the browser
automatically. The optional desktop window:

```bash
pip install -e ".[desktop]"
eva-desktop
```

---

## Command reference

| Command | Purpose | Needs |
|---|---|---|
| `eva version` | Print version | base |
| `eva diagnose` | Hardware, configuration, and paths | base |
| `eva doctor` | Dependency and model readiness | base |
| `eva first-run` | Guided setup wizard | base |
| `eva setup` | Install the LLM runtime | base |
| `eva devices` | List audio devices | base |
| `eva listen` | Live VAD/segmentation monitor | base + microphone |
| `eva echo-test` | Measure echo cancellation | base + mic/speakers |
| `eva models list/download/remove` | Manage models | base |
| `eva bench` | End-to-end pipeline benchmark (no mic) | full setup |
| `eva run` | Interactive voice assistant (terminal) | full setup + mic/speakers |
| `eva serve` | Platform API server (foreground; serves the web UI) | base |
| `eva start/stop/restart` | Background server lifecycle | base |
| `eva status` / `eva logs` | Server status / newest log tail | base |
| `eva personas/users/voices/memory` | Manage personas, profiles, voices, memory | base |
| `eva config show/schema/reset` | Inspect or reset settings | base |

---

## Troubleshooting

- **The setup wizard was cancelled or interrupted** — run `eva first-run` to
  resume; it re-detects what is still missing and continues.
- **`eva run` in a script/CI exits without starting** — a non-interactive shell
  cannot answer the wizard prompt; run `eva run --yes` (or `eva first-run --yes`)
  to auto-confirm, or complete setup manually first.
- **Force setup to run again** — `eva first-run` always shows the wizard; it
  re-verifies and only installs what is missing.
- **CUDA build fails to load (`llama.dll` dependency error)** — the NVIDIA
  cudart/cuBLAS wheels are missing; re-run `eva setup --cuda`, or fall back to
  `eva setup --cpu`.
- **No audio devices / stream fails to open** — check `eva devices`; on Linux
  ensure `libportaudio2` is installed.
- **Slow first response on CPU** — expected for a 4B model without a GPU; use a
  smaller LLM (`eva models download qwen3-1.7b-instruct-q4_k_m`, then set it in
  the configuration) or a CUDA GPU.
