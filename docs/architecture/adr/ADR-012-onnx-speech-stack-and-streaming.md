# ADR-012: ONNX-first speech stack; pipeline-level streaming strategy

Status: Accepted · Date: 2026-07-04

## Context
M2 required concrete runtimes for TTS and partial-transcript streaming. The
reference Kokoro package depends on PyTorch (~2.5 GB installed, slow startup);
whisper-family ASR has no true native streaming; the product must stay light,
offline, and packageable.

## Decision
1. **No PyTorch in the product.** The speech stack runs on ONNX Runtime and
   CTranslate2: Silero VAD (`pysilero-vad<3`, pure-Python + ONNX), Kokoro via
   **kokoro-onnx** (perceptually identical to the PyTorch build), and
   faster-whisper (CTranslate2). The only GPU-native runtime is llama.cpp.
2. **Streaming is a pipeline property, not an engine property.**
   - *ASR partials:* the segmenter emits periodic `UtteranceProgress`
     snapshots (default every 1.2 s); the orchestrator opportunistically
     transcribes the accumulated audio and publishes `PartialTranscript`.
     Engine-native streaming (e.g. Moonshine v2) can later slot behind the
     same events without orchestrator changes.
   - *TTS streaming:* sentence-granular synthesis via the chunker; segment N
     plays while N+1 synthesizes. Engines stay simple blocking functions.
3. **Audio format conversion is an adapter concern**: Kokoro's 24 kHz output
   is resampled to the 16 kHz pipeline format inside the adapter (linear
   interpolation now; polyphase revisit scheduled for M7).

## Rationale
Torch-free keeps the installed footprint in the hundreds of MB instead of GB,
shortens startup, and removes the most fragile packaging dependency. Putting
streaming in the pipeline keeps every engine port a simple blocking call —
easy to implement, easy to fake in tests — while the user-visible streaming
behavior is uniform across engines.

## Consequences
- The turn orchestrator owns all concurrency (token queue, speak worker,
  epoch checks); engines remain synchronous and trivially swappable.
- Partial transcripts cost extra ASR passes; they are settings-gated
  (`asr.partial_transcripts`) and skipped whenever a pass is already running.
