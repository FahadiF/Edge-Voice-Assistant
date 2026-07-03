# ADR-003: faster-whisper as default ASR

Status: Accepted · Date: 2026-07-03

## Context
Thesis used the reference `openai-whisper` package ("base"), batch-only. 2026
landscape: NVIDIA Parakeet TDT 0.6B v3 (best English WER, extremely fast),
Moonshine (edge-optimized English), SenseVoice (CJK strength), Whisper family
(most mature, 99 languages).

## Decision
Default ASR: **faster-whisper** (CTranslate2) — `small` int8 on GPU, `base` on CPU
profile. Incremental/partial transcription implemented at the pipeline level
(rolling windowed decode on the VAD-segmented buffer). Parakeet and Moonshine are
planned adapters behind the same `ASREngine` port.

## Rationale
- ~4× faster and lighter than reference whisper with identical accuracy — a pure win
  over the thesis baseline with minimal risk.
- Multilingual out of the box (product requirement not to hard-lock English).
- Mature, widely deployed, simple pip install on Windows and Linux — Parakeet's
  NeMo/onnx toolchain is heavier and English/EU-language-limited; better as an
  opt-in "fast English" profile than as the default.
- int8 `small` fits ~0.5 GB, coexisting with the 4B LLM in 6 GB VRAM.

## Consequences
- The `ASREngine` port must support: streaming partials, final decode, language hint,
  and word timestamps (used by endpointing heuristics and the UI).
- Benchmark harness (M7) compares faster-whisper vs Parakeet vs Moonshine on our own
  recorded fixtures; the default can change per-profile based on data.
