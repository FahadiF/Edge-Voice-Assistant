# Platform API Reference

The platform API (ADR-017) is the backend every client — CLI, desktop app, web
UI, and plugins — talks to. Start it with:

```bash
eva serve                  # binds to settings.server.host:port (default 127.0.0.1:8765)
eva serve --host 0.0.0.0 --port 9000
```

Interactive docs (Swagger UI) are generated automatically at `GET /docs`;
the raw OpenAPI document is at `GET /openapi.json`. This page is a map, not
the source of truth — the running server's `/docs` always reflects the
current code.

The web UI (`web/`, M5, ADR-023) is the reference consumer of this whole
API — every endpoint and WebSocket event below has a corresponding page or
live element in it (`eva serve --open` after `npm run build` in `web/`).
Its TypeScript type mirror (`web/src/api/types.ts`) is a second, executable
description of these shapes; if you change a schema here, that file (and
its pinned test in `web/src/components/SchemaForm.test.tsx`) needs the
matching update.

## Conventions

- All REST endpoints are versioned under `/api/v1`.
- Localhost-only by default; no authentication (see ADR-017 Part 9 — the
  `StateDep` indirection is the seam a future auth dependency attaches to).
- Every error response has the shape `{"detail": ..., "error_type": ...}`.
  Validation errors additionally include an `errors` list
  (`[{"loc": [...], "message": ..., "type": ...}]`).
- Nothing renders HTML; this is a pure JSON + WebSocket API.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/v1/health` | Liveness + version |
| GET | `/api/v1/system/hardware` | Detected CPU/GPU/RAM/VRAM + recommended tier |
| POST | `/api/v1/system/shutdown` | Gracefully stop the whole server process (M5.6) — engine stops, then uvicorn exits; how `eva stop` works |
| GET | `/api/v1/settings` | Current settings document |
| GET | `/api/v1/settings/schema` | JSON Schema (drives UI form generation) |
| PUT | `/api/v1/settings` | Replace the entire settings document |
| PATCH | `/api/v1/settings` | Deep-merge a partial document |
| POST | `/api/v1/settings/validate` | Validate a document without saving |
| POST | `/api/v1/settings/reset` | Reset to schema defaults |
| GET | `/api/v1/models` | List catalog models (+ install/compat state); `?kind=llm\|asr\|tts\|vad` |
| GET | `/api/v1/models/{id}` | Full model card |
| POST | `/api/v1/models/{id}/download` | Start a background download |
| DELETE | `/api/v1/models/{id}` | Remove an installed model |
| POST | `/api/v1/models/{id}/activate` | Set as active for its kind; profile → `custom` |
| GET | `/api/v1/diagnostics` | Full `RuntimeSnapshot` (models, devices, state, resources, latency) |
| GET | `/api/v1/plugins` | Discovered plugins (entry point group `eva.plugins`) |
| GET | `/api/v1/plugins/{id}` | Plugin status |
| POST | `/api/v1/plugins/{id}/enable` \| `/disable` \| `/reload` | Lifecycle actions |
| GET | `/api/v1/engine/status` | Running? current pipeline state |
| GET | `/api/v1/engine/readiness` | Same check as `eva doctor`, as JSON |
| POST | `/api/v1/engine/start` | Load models, open audio, start the orchestrator |
| POST | `/api/v1/engine/stop` | Stop and release everything |
| GET | `/api/v1/conversation/history` | Turns in the active session (needs a running engine) |
| GET | `/api/v1/conversation/current` | Current turn/pipeline state |
| POST | `/api/v1/conversation/say` | Start a turn from typed text (composer, M5.3) — same pipeline minus ASR |
| POST | `/api/v1/conversation/microphone` | Mute/unmute the microphone (M5.7); `{muted: bool}`. Broadcasts `MicrophoneMuted` |
| POST | `/api/v1/conversation/interrupt` \| `/cancel` | Stop the current turn now (aliases) |
| POST | `/api/v1/conversation/clear` | Clear history (starts a fresh conversation; old data kept) |
| POST | `/api/v1/conversation/resume` | Continue a stored conversation where it ended (M5.6): id, context, summary, and title preserved |
| GET | `/api/v1/conversation/export` | Export history as JSON |
| POST | `/api/v1/conversation/import` | Replace history from a JSON payload |
| POST | `/api/v1/memory/search` | Keyword search across all conversations (M4) |
| GET | `/api/v1/memory/stats` | `MemoryStats`: conversation/turn/embedding counts, db size |
| GET | `/api/v1/memory/context-preview` | Preview the exact LLM prompt + `ContextTrace` for `?text=` |
| DELETE | `/api/v1/memory/turns/{id}` | Forget one turn permanently |
| POST | `/api/v1/memory/turns/{id}/pin` \| `/favorite` | Boost retrieval scoring; `?pinned=false` to undo |
| POST | `/api/v1/memory/conversations/{id}/archive` | Hide from listings (reversible); `?archived=false` to restore |
| DELETE | `/api/v1/memory/conversations/{id}` | Delete a conversation and everything in it |
| POST | `/api/v1/memory/conversations/merge` | Move all turns from one conversation into another |
| POST | `/api/v1/memory/conversations/{id}/summarize` | LLM-generate and store a summary (originals kept) |
| GET | `/api/v1/memory/export` | Export conversations as JSON; `?conversation_id=` for one |
| POST | `/api/v1/memory/import` | Import a previously exported snapshot |
| DELETE | `/api/v1/memory` | Delete *all* memory (privacy: "delete my data") |
| GET | `/api/v1/personas` | List built-in + custom personas (no engine required) |
| GET | `/api/v1/personas/{id}` | One persona |
| POST | `/api/v1/personas` | Create/replace a custom persona |
| DELETE | `/api/v1/personas/{id}` | Delete a custom persona (built-ins cannot be deleted) |
| GET | `/api/v1/users` | List user profiles |
| POST | `/api/v1/users` | Create a user profile (id auto-generated if omitted) |
| GET | `/api/v1/users/{id}` | One user profile |
| PATCH | `/api/v1/users/{id}` | Partially update a user profile |
| POST | `/api/v1/users/{id}/activate` | Set as the active profile |
| DELETE | `/api/v1/users/{id}` | Delete a user profile |
| GET | `/api/v1/voices` | Voices available for the active TTS engine |
| POST | `/api/v1/voices/{id}/preview` | Synthesize a short phrase; returns raw 16 kHz PCM |
| WS | `/api/v1/ws` | Live event stream (see below) |

Everything under `/memory`, `/users`, and `/voices` needs a running engine
(`POST /engine/start` first) — they read/write the assistant's `MemoryStore`.
`/personas` does not: personas are configuration (ADR-022), served from
`settings.json` the same way `/settings` is.

**CLI equivalents.** Every one of these endpoints also has a direct CLI
command that works without `eva serve` running at all — `eva personas`,
`eva users`, `eva voices`, `eva memory`, and `eva profile` (the active user
profile shortcut; not to be confused with the unrelated, pre-existing `eva
profiles`, the hardware/model presets). The CLI commands open the memory
database directly (the same way `eva models` opens `ModelManager` directly)
rather than going through HTTP — see
[MANUAL_TESTING.md](MANUAL_TESTING.md) for a full walkthrough of both.

## WebSocket event stream

Connect to `ws://127.0.0.1:8765/api/v1/ws`. Browser connections must come
from a localhost origin: CORS middleware does not cover WebSocket
handshakes, so the endpoint checks the `Origin` header itself and rejects
foreign origins with close code 1008 (M5.6) — otherwise any website the
user visits could read live transcripts. Requests without an `Origin`
header (CLI tools, the desktop shell) are always accepted.

On connect you immediately receive:

```json
{"type": "snapshot", "data": { /* RuntimeSnapshot */ }}
```

After that, every engine event is forwarded as it happens:

```json
{"type": "PartialTranscript", "data": {"epoch": 3, "text": "what's the weather"}}
{"type": "LlmToken", "data": {"epoch": 3, "token": "It"}}
{"type": "LlmSentence", "data": {"epoch": 3, "text": "It looks sunny today."}}
{"type": "TtsAudioReady", "data": {"epoch": 3, "ttfa_ms": 1180}}
{"type": "StateChanged", "data": {"state": "speaking"}}
{"type": "MicrophoneMuted", "data": {"muted": true}}
{"type": "TurnCancelled", "data": {"epoch": 3, "reason": "barge-in"}}
{"type": "BargeInLatencyMeasured", "data": {"epoch": 3, "detected_to_silent_ms": 62}}
{"type": "ModelDownloadProgress", "data": {"model_id": "...", "bytes_done": ..., "bytes_total": ...}}
```

`type` is the event class name from `eva.core.events` — the same events the
orchestrator has always published; this is not a separate protocol. Clients
never poll: settings changes, model downloads, and conversation activity are
all observable purely by staying connected.

## Design notes

- **Engine lifecycle is explicit.** The server process starting does not open
  microphones or load multi-gigabyte models — `POST /engine/start` does, after
  the same readiness check `eva doctor` runs. This keeps `eva serve` safe to
  run in any environment (including CI) without side effects until asked.
- **No duplicated business logic.** Every router calls existing services
  (`ModelManager`, `eva.config.service`, `eva.onboarding`, the `Orchestrator`)
  — the API and the CLI are two thin clients of the same engine.
- See [ADR-017](adr/ADR-017-platform-api.md) for the full rationale, and
  [ADR-019](adr/ADR-019-memory-subsystem-and-sqlite-storage.md)–
  [ADR-022](adr/ADR-022-personas-user-profiles-voices.md) for the M4 memory/
  personalization endpoints specifically — they're additive to this API,
  not a new one.
