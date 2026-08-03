# Troubleshooting

Symptom → cause → fix. Start with `eva diagnose`, which prints your hardware, active
models, device placement, and the log directory.

**Jump to:** [Microphone](#microphone-and-voice-input) · [Audio output](#audio-output) ·
[Models](#models-and-downloads) · [GPU](#gpu-and-performance) ·
[Recognition quality](#recognition-quality) · [Server and UI](#server-and-web-ui) ·
[Desktop app](#desktop-app) · [Installation](#installation)

---

## Microphone and voice input

### EVA never responds to speech, but typing works

Typing works while speech does not, which means the engine is healthy and the audio
capture path is not. Work down this list:

1. **Is the microphone muted in Windows?** The most common cause by far, and invisible
   from inside EVA. Check the mic-mute key on your laptop (often F4 or F8, sometimes with
   an LED) and Settings → System → Sound → your microphone → make sure it isn't muted.
2. **Run `eva listen`.** Speak. You should see the level meter move well above −60 dBFS
   and `speech started` / `utterance ended` events appear.
   - Level stays near −96 dBFS → the operating system is delivering silence. It is a
     device or OS problem, not EVA. Back to step 1.
   - Level moves but no events → voice-activity detection is not triggering; see below.
3. **Is the microphone permission on?** Settings → Permissions → Devices → Microphone.
   With it off, EVA never opens an input device (typed chat and speech output still work).
4. **Is the right device selected?** `eva devices` lists everything. EVA follows the
   system default unless `audio.input_device` is set.

### Level moves but speech is never detected

The voice-activity threshold may be too high for a quiet microphone. Raise the input
level in the OS first — that is better than lowering the threshold. If you must, adjust
`vad.threshold` (default `0.5`) or `vad.min_speech_ms` (default `380` — utterances
shorter than this are discarded as noise).

### EVA interrupts itself, or reacts to its own voice

Echo cancellation is not working. Check `eva echo-test`, which plays a sample and reports
raw versus cleaned echo level plus any self-triggers. Then:

- Confirm `audio.echo_cancellation` is on.
- Use headphones to eliminate the problem entirely.
- Lower `audio.speaker_volume`.
- If echo cancellation is unavailable on your system, switch `audio.duplex_mode` to
  `half-duplex` or `push-to-talk`.

### Barge-in sometimes doesn't trigger

Speaking over the assistant needs `vad.barge_in_confirm_ms` (default 200 ms) of confirmed
speech while playback is active. Residual echo makes this harder — headphones help most.
`eva capture-test` records one utterance and shows exactly what the recognizer received.

---

## Audio output

### No sound, but the UI shows the assistant speaking

- `eva devices` — confirm the output device, and set `audio.output_device` if the system
  default is wrong.
- Check `audio.speaker_volume` in settings and the OS mixer.
- Look for `Duplex stream started` in the log. If it says `Playback-only stream started`,
  the microphone permission is off (output still works, input does not).

### Audio is choppy or clicks

Check the Diagnostics page for **dropped frames** and **callback errors**, both of which
should be zero. Non-zero usually means CPU saturation — speech synthesis runs on CPU and
is the heaviest consumer. Try a smaller model preset.

---

## Models and downloads

### First run downloads for a long time with no visible progress

Speech-recognition weights are fetched by their engine on first load, and that download
currently reports no progress — the UI shows "Loading speech recognition…" throughout.
It is working; `small` is ~460 MB. This is a known gap scheduled for the Architecture
Stabilization milestone.

Language-model and speech-synthesis downloads **do** report progress, via `eva models
download` or the Models page.

### `Model 'X' is not installed`

```bash
eva models list                    # what is installed
eva models download <model-id>     # fetch one
eva setup                          # guided: everything for your hardware
```

### The Models page says a speech-recognition model is not installed, but it works

A known reporting bug: install detection does not recognize the layout the engine
downloads into, so recognition models always display as not installed. Harmless — it
affects display only. Fix scheduled for Architecture Stabilization.

### Download fails with a checksum or size mismatch

The file is corrupt or truncated. Remove and retry:

```bash
eva models remove <model-id> && eva models download <model-id>
```

Integrity failures are deliberately hard errors, never warnings.

### `... failed integrity verification (checksum mismatch ...)`

An engine-managed model (a Whisper/CTranslate2 speech-recognition model) has weight
files on disk that no longer match the checksum recorded the first time it was
installed — most likely disk-level corruption (a bad sector, an interrupted copy, an
antivirus quarantine action) rather than anything EVA did. Caught here, at install
time, deliberately: the alternative is CTranslate2 crashing with an opaque native error
partway through a turn. Remove and re-download:

```bash
eva models remove <model-id> && eva models download <model-id>
```

A model installed before this check existed (Batch 10 / M6) has no recorded checksum
to compare against; `eva models list`/`eva models info <model-id>` reports its
integrity as unverified rather than corrupt — this is expected and not itself a
problem. It becomes verifiable the next time it is reinstalled.

---

## GPU and performance

### `Library cublas64_12.dll is not found`

CTranslate2 needs the CUDA runtime libraries on `PATH`. EVA registers them when the
language model loads, so this appears in paths that load speech recognition **without**
the language model. If you hit it in normal use, reinstall the CUDA runtime:

```bash
pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12
```

Note that `load()` may report `cuda` successfully and fail later during inference — that
is this same problem.

### Everything runs on CPU despite having a GPU

- `eva diagnose` shows the detected hardware and tier.
- `nvidia-smi` should list your GPU. If it doesn't, the driver is the issue.
- The CUDA build of the LLM runtime must be installed: `eva setup`, or manually per
  [INSTALLATION.md](INSTALLATION.md).
- Set `developer.debug` to see llama.cpp's load report, which prints the number of layers
  actually offloaded. `device: cuda` alone only means the build *supports* offload.

### Replies are slow to start

Roughly 3.5 s to first audio is expected on the 6 GB reference platform. About 1.6 s of
that is speech synthesis on CPU and about 1.65 s is language-model prefill — see
[Limitations](../../README.md#limitations). Options today:

- Use the **Fast** preset (smaller models).
- Shorten replies — the system prompt asks for brevity, but small models drift; a persona
  with `verbosity: concise` helps.
- Reduce `llm.context_length` if long conversations have made prefill slow.

### Out of memory on the GPU

Lower `llm.context_length` (the KV cache scales with it), switch to a smaller preset, or
set `asr.device` to `cpu` to keep recognition off the GPU.

---

## Recognition quality

### Common words are misrecognized

Known and measured: Whisper `small` confuses acoustically similar consonants (`fox`/`box`)
on far-field laptop microphones. Investigated in depth — see
[Limitations](../../README.md#limitations). What helps today:

- **Get closer to the microphone**, or use a headset. Distance is the strongest factor.
- **Disable Windows "Audio Enhancements"** on the microphone (Sound settings → device
  properties). Driver-side processing runs before EVA sees the signal.
- Diagnose your own audio: `eva capture-test --reference "what you will say"` records one
  utterance, saves the raw and processed audio, decodes both, and reports levels and
  word error rate. It will tell you whether information is being lost before or inside EVA.

### The assistant answers something you did not say

Check the transcript in the UI — if the transcript is wrong, this is recognition (above).
If the transcript is right and the answer is wrong, that is the language model; try a
larger preset.

---

## Server and web UI

### The web UI shows "engine not running"

The engine starts explicitly, never as a side effect of starting the server. Press
**Start** in the header, or `POST /api/v1/engine/start`. This is deliberate: `eva serve`
should never open your microphone just because it was run.

### Port 8765 already in use

Another EVA instance is probably running: `eva status`, then `eva stop`. Or change
`server.port` in settings.

### WebSocket connection rejected

The endpoint accepts only localhost origins. Browsing from another machine is not
supported by design — EVA binds to `127.0.0.1` and does not authenticate.

---

## Desktop app

### `eva desktop` says the desktop extra is missing

```bash
pip install -e ".[desktop]"
```

### The window is blank

The web UI has not been built. Either build it (`cd web && npm ci && npm run build`) or
use a release artifact, which ships it prebuilt.

### The window closed but EVA is still running

That is `desktop.close_to_tray` (on by default). Use the tray icon to reopen or quit, or
turn the setting off.

---

## Installation

### `ModuleNotFoundError` for `llama_cpp`

The LLM runtime is not a base dependency — it ships no PyPI wheels (ADR-013). Run
`eva setup`, or install manually per [INSTALLATION.md](INSTALLATION.md).

### `OSError: PortAudio library not found` (Linux)

```bash
sudo apt-get install libportaudio2
```

### Nothing here matches

Open an issue with your `eva diagnose` output, what you expected, what happened, and the
relevant portion of the log from the directory `eva diagnose` prints. See
[CONTRIBUTING.md](../../CONTRIBUTING.md).
