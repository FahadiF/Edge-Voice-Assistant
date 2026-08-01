# ADR-026: Engine lifecycle — startup, shutdown, cancellation, supervision

Status: Accepted · Date: 2026-07-06

## Context

M5.5's brief: make EVA feel like a real desktop application. The symptoms
were lifecycle symptoms — serial cold starts with no feedback, Ctrl+C
tracebacks, a real `ValueError: generator already executing` race in TTS
cancellation, Kokoro's event-loop cleanup fragility, unowned background
tasks, no process management, and no recovery when a component crashed.

## Decisions

### 1. Startup: parallel where safe, ordered where it matters, visible always

`Assistant.preload()` keeps ADR-015's GPU ordering — LLM first (owns the
GPU), then ASR — but loads the CPU-resident components (TTS, embedding)
concurrently on worker threads: total startup ≈ LLM+ASR time. Each
component brackets its load with `ComponentLoadStarted/Finished` events
(new in `eva.core.events`; failures are reported *before* they propagate),
which stream over the existing WebSocket: the header button narrates the
current component and the Dashboard renders a per-component checklist with
times. `ServerState.start_engine` binds the bus loop *before* preload so
the events actually flow during loading.

`tts.lazy_load` (default off) skips TTS at preload entirely — the Kokoro
adapter already self-loads on first synthesis, and the voices API registers
voices on demand. Off by default because for a voice-first product, moving
~3 s from startup (where the user already expects a wait) into the first
reply (where latency is the product metric) is the wrong default; the
toggle exists for typed-chat-heavy use.

Warm caches were reviewed, not built: llama.cpp already mmaps the GGUF,
the embedding tokenizer and Kokoro instance persist for the engine's
lifetime. Nothing to add.

### 2. Cancellation: one thread owns a synthesis stream

The `_drive_stream` bridge previously ran `next()` and `close()` on
whatever threads the shared executor picked. A barge-in close during an
in-flight pull raised `ValueError: generator already executing`, and
Kokoro's per-stream event loop was touched from multiple threads. Now each
stream gets a **single-worker executor**: every generator interaction is
serialized on one owner thread (request cancellation → generator exits →
join → close → cleanup, exactly the sequence the brief asked for), the
close is *queued before it is awaited* so it runs even if the awaiting task
is cancelled mid-teardown, and Kokoro's loop only ever runs on the thread
that created it. Kokoro's `finally` is additionally hardened: guarded
`aclose`, `shutdown_asyncgens`, and an unconditional `loop.close()`, each
exception-suppressed — cleanup can degrade but never crash the speak
worker, and `run_until_complete` is provably never called on a running
loop.

### 3. Shutdown: ordered and exception-proof

`ServerState.stop_engine`: background tasks → assistant (audio →
orchestrator → memory) → orchestrator loop task, every step individually
guarded so one failure cannot abort the rest. Combined with (2), Ctrl+C on
`eva serve` ends with "Engine stopped", not a traceback. (`eva run` already
had clean Ctrl+C handling since M3.)

### 4. Owned background tasks (`eva.core.tasks.TaskManager`)

Every fire-and-forget task now has a named owner: the server's downloads
(`server:download:<id>`), the orchestrator's barge-in latency measurements
and component recoveries (`orchestrator:*`). Strong references until
completion, uncaught exceptions logged with the task's name, and one
`shutdown()` that cancels-all/awaits-all — the teardown sequence the brief
sketched. The engine's `run()` task deliberately stays outside the manager:
its lifecycle *is* the engine lifecycle, awaited (not cancelled) in
`stop_engine`.

### 5. Supervised component recovery

A component crash costs at most one turn, never the assistant:
- **ASR** failure → that turn errors (contained as before), and the engine
  is reloaded (unload → load) in a background task.
- **TTS** failure → that *sentence* degrades to silence, the turn still
  completes (text reply, storage, events intact), engine reloaded in the
  background.
- Recovery is **cooldown-guarded** (one attempt per 30 s window) so a
  persistently broken component cannot trigger a reload storm.
- A WebSocket disconnect was already engine-independent; now pinned by a
  regression test.

### 6. Process lifecycle CLI (`eva start/stop/restart/status/logs`)

A PID-file + psutil layer over `eva serve` (`eva/service.py`): `start`
spawns a detached server (console output to `logs/server-console.log`),
records the PID, and polls `/health`; `stop` terminates gracefully
(terminate → wait → kill); `status` combines process liveness, API health,
and engine state; `logs` tails the newest log file. Stale or reused PIDs
are detected and cleaned. No service framework — a single-user local app
needs exactly this much.

## Consequences

- Startup progress is part of the event contract; new components must call
  `_load_component` (or publish the events themselves) to appear in the UI.
- Anything that wants a background task in the server or orchestrator
  should go through the owning `TaskManager`, not bare `create_task`.
- The M6 desktop shell inherits all of this for free: pywebview wraps the
  same server `eva start` manages, and the tray's start/stop actions can
  call the same `eva.service` functions.
