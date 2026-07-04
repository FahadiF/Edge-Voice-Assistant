# ADR-018: TTS streaming synthesis

Status: Accepted · Date: 2026-07-04

## Context

M3's top priority is reducing time-to-first-audio (TTFA), measured at roughly
0.9-2.0 s end-to-end. Code inspection of the turn pipeline
(`eva/conversation/orchestrator.py`) shows `speak_worker` calling
`TTSEngine.synthesize()` once per sentence as a single blocking call — no audio
reaches the speaker until Kokoro has rendered the *entire* sentence. This is
also the largest gap in barge-in responsiveness (ADR-006): a turn cannot be
audibly cancelled while a `synthesize()` call is in flight, because the engine
gives the pipeline no earlier point to check the turn epoch.

ADR-012 states streaming is a pipeline-level property, not an engine property,
because no available engine supported anything else at the time. Direct
inspection of the installed `kokoro-onnx` package shows this is no longer
true for TTS: `Kokoro.create_stream()` splits text into phoneme batches at
punctuation boundaries (bounded by `MAX_PHONEME_LENGTH`), synthesizes each
batch in a background thread, and yields audio chunks via an async generator
as soon as each batch is ready, while later batches continue processing.
Reimplementing this phoneme-aware batching ourselves at the orchestrator level
would duplicate logic Kokoro already does correctly, so this ADR takes it
directly.

## Decision

1. `TTSEngine` gains an **additive, non-abstract** method:

   ```python
   def synthesize_stream(self, text: str, *, voice: str, speed: float = 1.0) -> Iterator[Frame]:
       """Yield PCM chunks as they become available. Default: one chunk via synthesize()."""
       yield self.synthesize(text, voice=voice, speed=speed)
   ```

   Every existing and future adapter that does not override it behaves exactly
   as before (single chunk, same call). This preserves ADR-010's registry/
   adapter pattern and requires zero changes to any adapter other than Kokoro.

2. `KokoroTTS.synthesize_stream()` overrides the default, bridging
   `kokoro_onnx`'s `create_stream()` async generator to a synchronous
   generator by driving a dedicated `asyncio` event loop one `__anext__()` at
   a time. The event loop and background synthesis task persist across
   `next()` calls (they are simply idle between calls), so synthesis of batch
   N+1 genuinely continues while the orchestrator consumes and plays batch N.

3. The orchestrator's `speak_worker` consumes `synthesize_stream()` instead of
   `synthesize()`, pulling each blocking `next()` call off the event loop
   thread via `loop.run_in_executor`, and checking turn-epoch staleness
   **between chunks** — a cancellation checkpoint that did not previously
   exist mid-sentence. `TtsAudioReady` now fires on the first chunk of the
   first sentence, not the whole sentence.

4. Playback framing changes to match: `AudioOutput.say()` no longer flushes
   the trailing partial frame on every call (that would insert a small
   silence pad at every chunk boundary — audible as clicks/gaps). A new
   `finish_utterance()` method flushes once, after a sentence's chunk stream
   is exhausted. The non-streaming fallback path (one chunk per sentence via
   the default `synthesize_stream`) is unaffected because it calls
   `finish_utterance()` immediately after its single chunk.

## Rationale

This is the smallest change that removes the single largest TTFA and
barge-in-latency bottleneck identified by pipeline inspection, without
touching ADR-010 (registries stay untouched, adapters are unaffected unless
they choose to override) or ADR-017 (no API surface changes — this is
internal to the orchestrator/TTS boundary). Keeping the method non-abstract
with a working default means the port contract itself remains
capability-optional rather than a breaking change.

Speculative LLM generation on unconfirmed (partial) transcripts was
considered as a further TTFA reduction and explicitly deferred to M4+: it
would add a second speculative-cancellation path at the same time M3 is
hardening the existing one, which is a worse risk/reward trade during a
hardening milestone.

## Consequences

- `KokoroTTS` owns one extra `asyncio` event loop per `synthesize_stream()`
  call, created and closed within that call — no event loop leaks into the
  orchestrator's own loop.
- Playback callers using `AudioOutput.say()` must call `finish_utterance()`
  once per utterance; this is enforced by the orchestrator, not by
  `PlaybackQueue` itself (which stays a dumb frame queue, per its existing
  design).
- ASR remains fully blocking per utterance (CTranslate2 has no per-token abort
  hook) — this is an accepted, bounded limitation (typically 200-400 ms),
  unchanged by this ADR.
