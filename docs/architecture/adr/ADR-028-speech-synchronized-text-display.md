# ADR-028: Speech-synchronized text display

Status: Accepted · Date: 2026-07-26

## Context

Every display surface renders the assistant's reply at *generation* pace while
the speaker delivers it at *synthesis* pace, and those differ by roughly an
order of magnitude. The M6.2 pipeline trace measured the gap directly: the LLM
produced all seven sentences of a reply within ~1.8 s while sentence 1 was
still being synthesized, and playback stayed 2–10 s behind, its buffer growing
throughout. The web UI appends every `LlmToken` to the live bubble as it
arrives ([`web/src/ws/store.ts`](../../../web/src/ws/store.ts)) and `eva run`
prints tokens inline, so in practice:

- the user reads the entire answer,
- then waits while EVA reads it back to them,
- so the voice carries no information and becomes an obstacle,
- so users interrupt — stressing barge-in, the weakest subsystem.

This is the highest-priority conversation-experience defect (M7.1). It was
previously recorded as a perception artifact of fast text streaming; it is not.
It is the literal implemented behavior of the display layer.

The pipeline itself is not at fault and is not changed here. One generation,
one synthesis pass, nothing delayed, nothing generated twice — the only thing
that changes is *when text is allowed on screen*.

### Why "sentence queued" is not the synchronization point

The obvious cheap fix — reveal a sentence when it is handed to TTS, or when
`say()` enqueues its audio — does not work. `PlaybackQueue` deliberately holds
a large synthesized lead (measured 2.1 s → 9.9 s across one reply), which is
what lets sentence N keep sounding while N+1 synthesizes (ADR-018). Revealing
on enqueue would therefore still run ahead of the voice by the entire buffer
depth. "Queued" says nothing about "heard".

Timer-based pacing was also rejected: any estimate of "when will this sentence
finish" drifts against the real audio clock, and the drift is unbounded across
a long reply.

## Decision

**The playback clock is the synchronization source, and the display policy
lives in the display layer.**

### 1. Playback markers (`eva.audio.playback`)

`PlaybackQueue.enqueue()` accepts an opaque `marker` bound to the *next frame
the queue appends*. When the audio callback pulls that frame, the queue hands
the marker to its `on_marker` handler. The invariant is deliberately strong:

> **A fired marker means the user heard this audio.**

So audio a `stop()` cuts off fires nothing — including audio that would only
have begun sounding *during* the 40 ms fade-out, since a sentence whose first
10 ms leaked out of a fade was not spoken in any meaningful sense. Dropped
frames are accounted for in the played counter, so markers registered after a
flush still line up with the clock.

Markers are collected under the queue's mutex and dispatched after releasing
it, at most once per marked frame (a handful per turn, never per frame). A
handler that raises is logged and ignored — a broken consumer must not break
playback.

### 2. `TtsSentenceStarted` (`eva.core.events`)

The orchestrator attaches a `_SpeechMark(epoch, index, text)` to each
sentence's first audio chunk and, when playback reports it, publishes
`TtsSentenceStarted(epoch, index, text)`. The handler runs on the audio
callback thread and does exactly one thing: hand the event to the bus via
`publish_threadsafe`.

Two details matter for consumers:

- **`text` is the raw segment** as published by `LlmSentence` — Markdown
  intact. The speech filter's output exists for the TTS engine, not for the
  reader (ADR-024), so a display keyed off spoken text has to match the token
  stream the client already holds.
- **`index` is the 1-based position among all segments, and gaps are real.** A
  segment the speech filter drops entirely (a fenced code block) is never
  announced and its index is never reused. A client revealing everything up to
  the next announced index therefore shows the code block *in place*, rather
  than leaving a hole in the reply.

### 3. Display policy: `ui.sync_text_to_speech` (default on)

Clients keep both facts and choose between them:

- the full streamed text (`LlmToken`/`LlmFinished`), and
- how much of it has been spoken (`TtsSentenceStarted`).

The web store advances a **character cursor into the streamed text** rather
than concatenating announced sentences, so slicing preserves formatting —
newlines, list structure, fences — byte for byte. The store holds the facts;
`Conversation.tsx` decides which to render. `eva run` applies the same policy
in `ConsoleRenderer`, and additionally holds the per-turn metrics line until
the turn ends so it cannot land mid-reply (`LlmFinished` arrives before the
last sentence is spoken).

Turning the setting off restores exact pre-M7.1 behavior on both surfaces.

## Rationale

- **It cannot drift.** The reveal is driven by frames leaving the speaker, so
  text and audio stay in step by construction, for any reply length, on any
  hardware, whether TTS runs fast or slow.
- **It costs nothing.** No extra generation, no extra synthesis, no added
  latency, no polling, no timers. One event per spoken sentence.
- **It degrades safely.** If audio never plays (TTS unavailable, no output
  device, engine without audio started), nothing is announced and the reply
  still appears in full when the turn finishes. Text is never lost — only
  delayed.
- **Barge-in stays honest.** Cut-off audio reveals nothing, so the live view
  shows exactly what was heard.
- **The concern lands where it belongs.** The engine publishes a fact ("this
  sentence is now audible"); each client owns its own pacing. A future voice-
  first surface can use the same event for a different presentation without
  engine changes.

## Consequences

- `AudioOutput` (the orchestrator's audio port) grows two members:
  `say(..., marker=...)` and `set_marker_handler(...)`. Both fakes in the test
  suite implement them; they drain instantly, so "queued == heard" there, and
  the audio-clock deferral is tested where it lives, in `test_playback.py`.
- `PlaybackQueue` now carries marker bookkeeping (`_appended`/`_played` frame
  counters). Frame handling itself is unchanged, so barge-in timing — a 40 ms
  fade, measured constant — is untouched.
- Marker dispatch happens on the PortAudio callback thread. The handler is a
  bus hand-off; anything heavier must not be added there.
- On a barge-in the *live* view stops at what was heard, while the archived
  transcript entry (and memory, and export) still carry the full generated
  reply, marked interrupted. That asymmetry is intentional: storage stays
  canonical (ADR-024), the live view stays honest about what was spoken.
- The reveal is one sentence granular. On CPU TTS a long sentence appears as
  one block after a pause rather than word by word. Finer granularity would
  need sub-sentence audio marks and is not worth the complexity until TTS is
  fast enough for it to be visible.

## Alternatives rejected

| Alternative | Why not |
|---|---|
| Reveal on `LlmSentence` / on `say()` | Runs ahead by the whole playback buffer (2–10 s measured) — the same defect, smaller. |
| Timer paced on estimated audio duration | Drifts against the real clock; unbounded error over a long reply. Explicitly excluded by the M7.1 brief. |
| Generate a short spoken reply plus a longer written one | Two generations, more latency, and the two can disagree. |
| Slow the LLM to synthesis speed | Wastes the buffered lead that keeps speech gapless, and delays the reply's completion for no gain. |
| Drop the live text entirely in voice mode | A real option for a future voice-first surface (M7 UX), but it removes a capability users have today rather than fixing the mismatch. |
