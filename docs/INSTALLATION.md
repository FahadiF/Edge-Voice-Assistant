# Installation Guide

Edge Voice Assistant runs fully offline after a one-time setup. Installation has
three stages:

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

## Command reference

| Command | Purpose | Needs |
|---|---|---|
| `eva version` | Print version | base |
| `eva diagnose` | Hardware, configuration, and paths | base |
| `eva doctor` | Dependency and model readiness | base |
| `eva setup` | Install the LLM runtime | base |
| `eva devices` | List audio devices | base |
| `eva listen` | Live VAD/segmentation monitor | base + microphone |
| `eva echo-test` | Measure echo cancellation | base + mic/speakers |
| `eva models list/download/remove` | Manage models | base |
| `eva bench` | End-to-end pipeline benchmark (no mic) | full setup |
| `eva run` | Interactive voice assistant | full setup + mic/speakers |

---

## Troubleshooting

- **`eva run` says "setup is incomplete"** — run `eva doctor`; it lists each
  missing runtime or model with the command to fix it.
- **CUDA build fails to load (`llama.dll` dependency error)** — the NVIDIA
  cudart/cuBLAS wheels are missing; re-run `eva setup --cuda`, or fall back to
  `eva setup --cpu`.
- **No audio devices / stream fails to open** — check `eva devices`; on Linux
  ensure `libportaudio2` is installed.
- **Slow first response on CPU** — expected for a 4B model without a GPU; use a
  smaller LLM (`eva models download qwen3-1.7b-instruct-q4_k_m`, then set it in
  the configuration) or a CUDA GPU.
