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
| POST | `/api/v1/conversation/interrupt` \| `/cancel` | Stop the current turn now (aliases) |
| POST | `/api/v1/conversation/clear` | Clear history |
| GET | `/api/v1/conversation/export` | Export history as JSON |
| POST | `/api/v1/conversation/import` | Replace history from a JSON payload |
| WS | `/api/v1/ws` | Live event stream (see below) |

## WebSocket event stream

Connect to `ws://127.0.0.1:8765/api/v1/ws`. On connect you immediately receive:

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
- See [ADR-017](adr/ADR-017-platform-api.md) for the full rationale.
