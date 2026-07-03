# ADR-004: Kokoro-82M as default TTS, Piper fallback, Chatterbox optional

Status: Accepted · Date: 2026-07-03

## Context
Thesis used Coqui `your_tts` — the company shut down, the model is dated, digits had
to be hand-expanded, synthesis is batch-only and slow. Requirements: streaming
(per-sentence) synthesis, multiple voices, permissive license, runs on CPU so the
GPU stays free for the LLM.

## Decision
- **Default: Kokoro-82M** (Apache-2.0) on CPU — faster than real-time, 50+ voices,
  8 languages, strong perceived quality for its size.
- **Fallback: Piper** for very weak CPUs.
- **Optional profile: Chatterbox** (MIT, 0.5B) for expressive speech / voice cloning
  on GPUs with headroom.
- Coqui is dropped entirely.

## Rationale
- Sentence-chunked synthesis with Kokoro on CPU keeps time-to-first-audio low while
  the GPU is fully dedicated to LLM decode — the right resource split for 6 GB VRAM.
- Apache-2.0 is the only clean commercial-grade license among the top candidates
  (XTTS is non-commercial CPML; Piper's active fork is GPL).
- Built-in voice packs directly satisfy the "multiple voices" feature.

## Consequences
- `TTSEngine` port contract: synthesize(segment) → audio chunks (streamed), voice
  selection, speed control, sample-rate metadata. Text normalization (numbers,
  units, abbreviations) is a shared pre-TTS component, not per-engine hacks.
