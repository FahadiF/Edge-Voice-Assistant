# Thesis Prototype Analysis

Reference implementation: `Github/Modular-Software-Implementation-Edge-Voice-Chatbot`
(four scripts: `chatbot-dialogpt.py`, `chatbot-moderate-qwen.py` [Phase 1],
`chatbot-vad-qwen.py` [Phase 2], `chatbot-threaded-qwen.py` [Phase 3]).
Analyzed 2026-07-03. The prototype is treated as read-only historical reference.

## What the prototype is

A single-process, four-thread pipeline (Phase 3, 566 lines):

```
Thread A (Capture: sounddevice + Silero VAD)
   → audio_queue (maxsize=1, np.ndarray)
Thread B (Inference: openai-whisper "base" + Qwen2.5-1.5B fp16 via transformers)
   → tts_queue (maxsize=1, str)
Thread C (Output: Coqui your_tts + sd.play)
Thread D (Keyboard: Space = stop speech, q = quit)
```

Stack: Python 3.12, torch 2.8, openai-whisper, transformers, coqui-tts, silero-vad,
sounddevice. Target: RTX 3060 6 GB.

## Strengths worth preserving

1. **Pipeline decomposition is fundamentally right.** Capture / inference / output as
   concurrent stages with queue handoff is the correct skeleton; Edge Voice Assistant keeps this shape
   (generalized to more stages and proper cancellation).
2. **Backpressure thinking.** `maxsize=1` queues to avoid stale-utterance pile-up shows
   the right instinct — the new design replaces it with turn-epoch invalidation, but the problem
   was correctly identified.
3. **Cascading sentinel shutdown** is a clean, deadlock-free termination pattern.
4. **VAD-gated capture** with tuned thresholds (speech threshold, ~960 ms end-of-utterance
   silence, minimum-speech noise gate, hard timeout). These tuned constants encode real
   experimental knowledge and carry over as defaults.
5. **Per-stage timing instrumentation** — the habit carries into the new implementation as a metrics subsystem.
6. **Cross-platform awareness** (msvcrt vs termios, CUDA/CPU fallback).
7. **Prompt discipline**: system prompt constraining reply length for voice, bounded
   history window, `apply_chat_template` with fallback.

## Technical debt

| # | Debt | Consequence |
|---|---|---|
| D1 | Monolithic single-file scripts; module-level globals for queues/events/config | Untestable, unmaintainable, not extensible |
| D2 | No package structure, tests, linting, CI, or logging framework (raw `print`) | Not production software |
| D3 | Config as constants in source | No profiles, no user settings, edits require code changes |
| D4 | Virtualenv (`edge_env/`) committed inside the repo | Repo bloat; requirements not the source of truth |
| D5 | Coqui TTS (`your_tts`) — the company shut down; model is dated | Dead dependency; digit-expansion hack (`_sanitize_for_tts`) works around missing phoneme coverage |
| D6 | `openai-whisper` reference implementation instead of CTranslate2 (`faster-whisper`) | ~4× slower, more VRAM than necessary |
| D7 | Qwen2.5-1.5B fp16 via `transformers` (~3 GB VRAM) | 4-bit GGUF of a *larger, better* model fits the same budget |
| D8 | Whisper hallucination filter is a substring check (`"<|" in text`) | Fragile; misses common hallucinations ("Thank you.", repeated phrases) |
| D9 | Mic `InputStream` reopened per utterance; playback via `sd.play` + 50 ms sleep-polling | Latency jitter; no shared clock between capture and playback (kills any AEC option) |

## Performance bottlenecks (why it can't feel like ChatGPT Voice)

**B1 — Nothing streams.** Every stage is batch: full utterance → full transcription →
full LLM generation (`lm.generate`, blocking) → full TTS synthesis → playback.
Perceived latency is the **sum** of all stage latencies. Streaming LLM decode with
sentence-chunked TTS collapses time-to-first-audio to roughly
`ASR + first-sentence-generation + first-sentence-TTS`.

**B2 — Barge-in is a heuristic, not an architecture.**
- Only works in `HEADSET_MODE`; on speakers the mic is muted during playback
  (half-duplex) because there is no echo cancellation.
- Requires ~640 ms of continuous speech before triggering, then **discards** that
  speech (`recorded_chunks.clear()`), so the user must say the interruption again.
- Stopping playback does not cancel in-flight LLM generation or TTS synthesis;
  with `maxsize=1` queues the *new* utterance blocks behind the *stale* reply.
  Repeated interruptions queue up stale turns.

**B3 — No echo cancellation.** The mute-during-playback workaround makes true
full-duplex impossible and is the root cause of B2.

**B4 — Single inference thread serializes ASR and LLM**, and TTS synthesis is
serialized with playback in Thread C. On-GPU stages can't overlap with playback of
the previous sentence.

**B5 — VAD end-of-utterance wait (~960 ms) is a fixed cost** added to every turn
before ASR even starts. Streaming/partial ASR during the utterance hides most of it.

## What must be redesigned vs. reused

- **Reuse (as defaults/knowledge):** Silero VAD + tuned thresholds; pipeline stage
  decomposition; sentinel shutdown idea; voice-optimized system prompt; timing metrics.
- **Redesign:** everything else — streaming pipeline with cancellation, full-duplex
  audio with AEC, model runtimes (llama.cpp, faster-whisper, Kokoro), configuration,
  packaging, UI, tests. See [ARCHITECTURE.md](ARCHITECTURE.md) and the ADRs.
