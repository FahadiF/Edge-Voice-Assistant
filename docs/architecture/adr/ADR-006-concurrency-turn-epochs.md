# ADR-006: asyncio orchestration + worker threads + turn-epoch cancellation

Status: Accepted · Date: 2026-07-03

## Context
The thesis used 4 threads with `maxsize=1` queues. That prevented stale pile-up but
made cancellation impossible: a barge-in could stop playback yet the stale LLM/TTS
work kept running and blocked the new turn.

## Decision
- **asyncio** event loop owns the turn state machine, orchestration, and the
  FastAPI/WebSocket server.
- Blocking inference runs in worker threads (executors) that stream chunks back via
  async queues and check for cancellation between chunks (llama.cpp abort callback,
  per-sentence TTS granularity).
- **Turn epochs**: a monotonically increasing integer; every pipeline artifact is
  tagged. Barge-in = epoch increment; all consumers drop stale-epoch items; producers
  observing a stale epoch abort. One mechanism handles interruption, repeated
  interruption, and shutdown uniformly.
- The PortAudio callback stays real-time-safe (ring buffers only, no locks/allocation).

## Rationale
- Voice pipelines are event-driven state machines — asyncio models this naturally and
  the server/UI layer is async anyway; one paradigm end to end.
- Epoch tagging is simpler and more robust than trying to flush queues correctly at
  every cancellation point (the thesis's structural bug).
- Threads (not processes) suffice: all heavy compute releases the GIL inside native
  code (CTranslate2, llama.cpp, ONNX, PortAudio).

## Consequences
- The FSM + epoch core is pure Python with no model dependencies → exhaustively unit
  tested with fake adapters, including double-interrupt races.
- Free-threaded Python or subprocess isolation can be revisited later without
  changing the port contracts.

## Amendment (M3, 2026-07-04): chunk-boundary cancellation + no separate `processing` state

- **Cancellation checkpoint granularity increased for TTS.** "Per-sentence TTS
  granularity" above described the only checkpoint that existed at the time —
  cancellation checked once per sentence, because `TTSEngine.synthesize()` was a
  single blocking call per sentence with nothing in between. ADR-018 adds
  `synthesize_stream()`; the speak worker now checks epoch staleness **between
  each streamed chunk**, not just between sentences. This is the same mechanism
  described here (drop stale-epoch items at the next boundary), just with more
  boundaries available for TTS specifically.
- **No separate `processing` state was added.** M3 considered whether the
  `idle/listening/thinking/speaking` FSM needed a fifth `processing` state for
  "ASR done, LLM generating." Reviewing the code: `thinking` already covers
  exactly that window (`_set_state("thinking")` is entered right after a turn
  starts and holds through ASR and LLM generation until the first TTS chunk
  flips it to `speaking`). A separate `processing` state would carry no new
  information and would duplicate `thinking` — rejected as unnecessary
  complexity, not an oversight.
