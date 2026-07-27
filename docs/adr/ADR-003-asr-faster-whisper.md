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

---

## Model Selection History

The engine choice above (faster-whisper / CTranslate2) is unchanged. This section
tracks *which model* each tier gets, so the reasoning stays traceable.

### 2026-07-03 — original decision
`small` int8 on GPU, `base` on CPU. The alternatives weighed were Parakeet TDT,
Moonshine, SenseVoice, and the Whisper family generally.

### 2026-07-04 — `distil-large-v3` added (M2.5)
Added in "Added model presets and diagnostics" to give the new `gpu-12gb` tier a
larger option. **No comparison was recorded**, and it carried guidance —
"High-accuracy English ASR for 12 GB+ GPUs" — that was never measured.

### Why `large-v3-turbo` was absent
It simply was not in the option set on 2026-07-03; it appears nowhere in this ADR,
the CHANGELOG, or any commit message before M7.2. It was **not evaluated and
rejected** — it was never considered. Recording that distinction is the point of
this section.

### 2026-07-26 — `large-v3-turbo` added to the catalog (M7.2)

**Evidence gathered** (RTX 3060 Laptop 6 GB, Ryzen 9 5900HX, Windows 11):

| | `small` | `large-v3-turbo` |
|---|---|---|
| VRAM resident, alongside the 4B LLM | 370 MiB | **1121 MiB** (1.5 GB headroom; 1.27 GB at ctx 16384) |
| Decode, 4 s utterance, GPU idle | 255 ms | 326 ms |
| Decode, while the LLM is generating | 348 ms | **451 ms** (+103 ms on the critical path, ≈2 % of a 3.5 s TTFA) |
| LLM throughput during concurrent ASR | 55.8 tok/s | 42.1 tok/s (−25 %, and only during barge-in overlap) |
| Fricative recovery on 11 real recordings | 1 of 4 | **3 of 4** |

The decoder was swept separately and left alone: `initial_prompt` kept (removing it
changed nothing and wider beams *raise* prompt-copy hallucination), `beam_size=1`
kept, and `compute_type="auto"` already resolves to `int8_float16` on this GPU.

**Why it supersedes `distil-large-v3` as the multilingual high-accuracy option:**
distil is English-only, which conflicts with this ADR's own "not to hard-lock
English" requirement and with the six-language registry. Turbo is multilingual and
measurably fits the 6 GB tier.

**What changed here:** catalog entry added; `distil-large-v3` retained as an
English-only option with its unsupported "12 GB+" guidance removed. **No tier
default changed.**

### Still requires benchmarking

- **`gpu-12gb` default.** Currently `small` (balanced). Whether `large-v3`,
  `large-v3-turbo`, or a newer multilingual model is right there is **unmeasured** —
  no 12 GB card has been tested. Deliberately not guessed at.
- **`gpu-6gb` default.** Turbo is the candidate, but the evidence is n=11, one
  speaker, one room, one session. Gated on live validation plus a fixture benchmark.
- **`distil-large-v3` footprint.** Its `vram_mb`/`ram_mb` remain original estimates.
- **`distil-large-v3.5`** exists in faster-whisper's resolver map and is not
  catalogued; English-only, so it inherits the same disqualification.

Tracked in `docs/BACKLOG.md`.
