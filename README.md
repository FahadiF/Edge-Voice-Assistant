![Version](https://img.shields.io/github/v/release/FahadiF/Edge-Voice-Assistant?include_prereleases)
![Python](https://img.shields.io/badge/python-3.12+-blue)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-success)
![License](https://img.shields.io/badge/License-Apache%202.0-blue)
![Tests](https://img.shields.io/badge/tests-982%20passing-brightgreen)

# Edge Voice Assistant (EVA) <img width="25" height="25" alt="icon" src="https://github.com/user-attachments/assets/71fe0186-e329-4410-98e9-b1287b17b382" />

**A private voice assistant that runs entirely on your own computer.**

Talk to EVA and it listens, thinks, and answers out loud — with no cloud service involved.
Speech recognition, the language model, and speech synthesis all run locally. Once the
models are downloaded you can unplug the network and nothing changes.

<img width="1683" height="726" alt="image" src="https://github.com/user-attachments/assets/95ef46bc-7dd7-483f-a935-39e7e8896d6e" />

> **Status: alpha (v0.7.0-alpha.1).** Everything described below is implemented and tested.
> Known limitations are listed [plainly, further down](#limitations). Interfaces may still
> change between releases.

**Jump to:** [Why EVA](#why-eva) · [Quick start](#quick-start) ·
[What it does](#what-it-does-today) · [Architecture](#architecture) ·
[Hardware](#hardware-and-platform-support) · [Limitations](#limitations) ·
[Contributing](#for-contributors) · [Documentation](#documentation)

---

## Why EVA

Most voice assistants send your voice to someone else's computer. EVA doesn't — and it
tries to do that without feeling like a compromise:

- **You can interrupt it.** Talk over EVA mid-sentence and it stops in 40 ms, keeps the
  words you just said, and answers them. Interruption is the mechanism the entire runtime
  is organized around, not a feature bolted on afterwards.
- **It speaks while it thinks.** Generation, synthesis, and playback overlap, so a reply
  begins before it has finished being written.
- **Nothing is hard-wired.** Every model, engine, and runtime is referenced by id and
  resolved through a registry. Changing the language model is a setting, not a patch.
- **It is honest about itself.** Asked to do something it cannot do, EVA says so instead
  of inventing an answer.

---

## Quick start

Requires **Python 3.12+** on **Windows 10/11** or **Linux**.

```bash
git clone https://github.com/FahadiF/Edge-Voice-Assistant.git
cd Edge-Voice-Assistant
python -m venv .venv && .venv\Scripts\Activate.ps1   # Linux: source .venv/bin/activate
pip install -e .
eva run
```

On first run, `eva run` detects that setup is incomplete and launches a guided wizard —
it shows your hardware, the recommended runtime, and the models to download (roughly
3–5 GB), asks once to confirm, then installs everything and starts the assistant.
No further commands needed.

> **Linux users:** install PortAudio first: `sudo apt-get install -y libportaudio2`

### Running EVA

Once set up, you can run EVA in your browser or as a native desktop app:

| Command | Purpose |
|---|---|
| `eva start` | Start the server in the background; open `http://127.0.0.1:8765` in your browser |
| `eva stop` | Stop the background server |
| `eva desktop` | Launch the native desktop app with a system tray icon |

> **Desktop app prerequisite:** `pip install -e ".[desktop]"` (one-time, adds the `pywebview` and `pystray` extras)

**Additional Utilities:**

| Command | Purpose |
|---|---|
| `eva serve` | Start the server in the **foreground** (useful for development; Ctrl+C to stop) |
| `eva status` | Show whether the background server is running |
| `eva diagnose` | Hardware, configuration, and paths report |
| `eva devices` | List audio input/output devices |
| `eva listen` | Live voice-activity monitor — helpful for microphone troubleshooting |

Full instructions: **[docs/INSTALLATION.md](docs/INSTALLATION.md)** ·
Something not working? **[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)**

---

## What it does today

**Conversation**
- Streaming spoken conversation — speech in, speech out
- Real-time barge-in: interrupt at any point and it stops immediately
- Speech-synchronized text: on-screen text appears as it is spoken, never ahead of it
- Typed input through the identical pipeline, minus speech recognition
- Markdown replies rendered on screen and spoken cleanly (no "asterisk asterisk")

**Memory and personalization**
- Conversations persist in SQLite with semantic recall across past sessions
- Personas, user profiles, and per-language voice selection
- Auto-titled conversations; resume any stored conversation

**Interfaces** — four clients, one engine, one API
- Command line, local REST + WebSocket API, React web UI, native desktop shell

**Control and transparency**
- Permission toggles for microphone, memory storage, and system information
- Hardware detection with model presets (Balanced / Fast / High Accuracy / Low Memory / Developer)
- Deep runtime diagnostics: per-stage latency, barge-in timing, queue depths, live event log

---

## Architecture

```
Microphone ─► WebRTC APM ─► Silero VAD ─► Segmenter ─► ASR ─► Context Builder
               (AEC/NS/AGC)                                          │
                                                                     ▼
Speaker  ◄── Playback ◄── TTS ◄── Sentence chunker ◄── LLM (streaming)
```

Every stage runs concurrently — the language model streams tokens while earlier sentences
are already being spoken. A turn-epoch system makes each stage cancellable mid-flight,
which is what makes interruption feel instant.

**Default local stack**

| Stage | Implementation | Runtime |
|---|---|---|
| Voice activity | Silero VAD | ONNX Runtime |
| Speech recognition | faster-whisper (`small` / `base`; `large-v3-turbo` optional) | CTranslate2 |
| Language model | Qwen3.5-4B-Instruct Q4_K_M | llama.cpp |
| Speech synthesis | Kokoro-82M | ONNX Runtime |
| Embeddings | all-MiniLM-L6-v2 | ONNX Runtime |

None of these names appear in the conversation engine. Each sits behind a port
(`ASREngine`, `LLMEngine`, `TTSEngine`, `VADEngine`, `MemoryStore`, `EmbeddingProvider`)
and is resolved from a registry at runtime.

Deeper detail: **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** ·
**[diagrams](docs/ARCHITECTURE_DIAGRAMS.md)** ·
**[architecture decision records](docs/adr/README.md)**

---

## Hardware and platform support

| Tier | Detected by | Models | Status |
|---|---|---|---|
| `gpu-12gb` | ≥ 11 GB VRAM | Qwen3.5-9B + Whisper small | Supported |
| `gpu-6gb` | ≥ 5.5 GB VRAM | Qwen3.5-4B + Whisper small | **Reference platform** |
| `cpu-only` | no / small GPU | Qwen3-1.7B + Whisper base | Supported, slower |

**Measured on the reference platform** — RTX 3060 Laptop (6 GB), Ryzen 9 5900HX, 16 GB RAM,
Windows 11:

| | |
|---|---|
| VRAM with language model + speech recognition resident | 3.9 GB of 6.1 GB |
| Barge-in, detection to silence | 40 ms |
| Speech recognition, 4 s utterance | ~250 ms |
| Time to first audio | ~3.5 s ([why](#limitations)) |

**Operating systems.** Windows 10/11 and Linux are supported, and CI runs the test suite
on both. macOS is not supported: llama.cpp has Metal support, but EVA has no macOS
packaging or testing. Contributions welcome.

**Acceleration.** NVIDIA / CUDA is the tested path. AMD / ROCm is detected but untested.
Everything runs CPU-only without a GPU, more slowly.

---

## Limitations

Knowing these up front is more useful than discovering them:

- **Speech synthesis runs on CPU.** The CPU build of ONNX Runtime is a base dependency,
  so Kokoro cannot use the GPU. First-clause synthesis takes ~1.6 s and is the largest
  single part of the ~3.5 s time-to-first-audio. Moving it to a GPU execution provider is
  the highest-value performance work outstanding.
- **Speech recognition accuracy is bounded by the default model.** Whisper `small`
  confuses acoustically similar consonants (`fox` / `box`) on far-field laptop
  microphones. `large-v3-turbo` is catalogued and selectable — it recovered most of
  those cases for roughly +2% of time-to-first-audio — but it is not yet a tier
  default, pending a wider benchmark.
- **No internet access, tools, file access, or vision.** Permission toggles for these
  exist as the contract that future capabilities must respect — the capabilities
  themselves are not implemented, and EVA says so when asked.
- **Long conversations grow the prompt.** Automatic summarization is implemented but not
  yet wired into the live conversation loop.
- **Windows audio defaults to the MME host API**, which reports ~210 ms of loop latency.
  Lower-latency host APIs are not selected automatically yet.
- **Single user, one conversation at a time.** No multi-user or concurrent sessions.

---

## Roadmap

Milestones M0–M6 are complete: full-duplex audio core, streaming pipeline, platform API,
memory and personalization, web UI, and desktop shell. M7 (conversation experience) is in
progress — M7.1 (speech-synchronized text display) is done; M7.2 (ASR accuracy
investigation) is complete with implementation pending.

Planned next, in order: architecture stabilization → ASR model upgrade → performance work
(GPU speech synthesis, prompt-cache reuse) → a unified provider abstraction making local
and remote models interchangeable → an **optional, off-by-default** online mode with
search, retrieval, and citations.

Online features will never be required, and enabling them will never reduce what works
offline. See **[ROADMAP.md](docs/ROADMAP.md)** for detail and **[CHANGELOG.md](CHANGELOG.md)**
for what has already shipped.

---

## For contributors

Contributions are genuinely welcome. **[CONTRIBUTING.md](CONTRIBUTING.md)** covers the
process; **[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)** covers the working environment.

```bash
pip install -e ".[dev]"
pre-commit install
```

The quality gate, which CI runs on Windows and Linux:

```bash
ruff check . && ruff format --check . && mypy && pytest -m "not integration"
cd web && npm ci && npm run lint && npm run build && npm test
```

Five rules shape every change:

1. Core code never names a concrete implementation — register it, resolve it by id.
2. Dependencies point inward: subsystems depend on `core` and `config`, never the reverse.
3. Every significant design decision gets an [ADR](docs/adr/README.md).
4. New behavior needs tests. Hardware- and model-dependent tests are marked `integration`
   and excluded from CI.
5. Documentation is part of the change, not a follow-up.

Good places to start: additional speech engine adapters, new language registry entries,
non-NVIDIA acceleration, macOS packaging.

---

## Documentation

| Document | For |
|---|---|
| [INSTALLATION.md](docs/INSTALLATION.md) | Installing and running EVA |
| [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | When something doesn't work |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | How the system is built |
| [ARCHITECTURE_DIAGRAMS.md](docs/ARCHITECTURE_DIAGRAMS.md) | Sequence and component diagrams |
| [API.md](docs/API.md) | REST + WebSocket reference |
| [adr/README.md](docs/adr/README.md) | Architecture decision records |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to contribute |
| [DEVELOPMENT.md](docs/DEVELOPMENT.md) | Developer environment and workflow |
| [ROADMAP.md](docs/ROADMAP.md) | Milestones and status |
| [BACKLOG.md](docs/BACKLOG.md) | Unscheduled ideas, with rationale |
| [MANUAL_TESTING.md](docs/MANUAL_TESTING.md) | Validation scenarios |
| [CHANGELOG.md](CHANGELOG.md) | Release history |

---

## Research background

Edge Voice Assistant originated from my Master's thesis research in **Sustainable and
Autonomous Systems** at the **University of Vaasa**, investigating whether a genuinely
conversational voice assistant can run entirely on consumer edge hardware without cloud
inference.

The original thesis implementation is preserved separately as a historical research
artifact; this repository is the long-term open-source continuation of that work.

**Original thesis repository:**
https://github.com/FahadiF/Modular-Software-Implementation-Edge-Voice-Chatbot

---

## Acknowledgements

I would like to express my sincere gratitude to my thesis supervisor,

**Jani Boutellier** — https://github.com/jboutell

for his guidance, valuable feedback, and support throughout the research that inspired
this project.

I am also grateful to the open-source community and the developers behind **llama.cpp**,
**faster-whisper**, **Kokoro ONNX**, **Silero VAD**, **ONNX Runtime**, **CTranslate2**,
and **FastAPI**, whose work makes modern local AI accessible to everyone.

---

## License

Licensed under the **Apache License 2.0** — see [LICENSE](LICENSE).

Model weights carry their own licenses; run `eva models list` to see each. `pystray`
(system tray) is LGPL-3.0 and dynamically imported, per
[ADR-027](docs/adr/ADR-027-native-desktop-shell.md).
