# ADR-017: Platform API — FastAPI backend for every client

Status: Accepted · Date: 2026-07-04

## Context
Through M2.6, the CLI was the only client of the engine. The roadmap requires a
desktop app, a web UI, and a plugin ecosystem, all needing the same
capabilities (settings, models, diagnostics, conversation control) without
duplicating engine logic per client. ARCHITECTURE.md §2/§8 already specified a
FastAPI + WebSocket engine server as the intended shape; this milestone builds
it, backend only — no desktop or web UI code.

## Decision

1. **Versioned REST + one WebSocket, under `/api/v1`.** Every capability the
   CLI already has — settings, models, diagnostics, plugins, conversation,
   engine lifecycle — is a router in `eva/server/routers/`. `eva/server/app.py`
   is the only place that assembles them; the CLI (`eva serve`) is one caller
   of `create_app()`, not a special case.

2. **`ServerState` is the single engine-lifecycle owner** (`eva/server/state.py`).
   Opening audio devices and loading models is an explicit `POST
   /api/v1/engine/start` — never an implicit side effect of the server process
   starting. Every router is a thin translation from HTTP to `ServerState`
   methods; no router re-implements engine construction (mirrors
   `eva.engine.build_assistant`, `eva.onboarding.check_readiness`,
   `eva.config.service`, `ModelManager` — all reused, not duplicated, per
   Part 10).

3. **The event bus is the WebSocket's only content.** `GET /api/v1/ws`
   subscribes to the same `EventBus` the orchestrator already publishes every
   turn/transcript/token/TTS/state event to (ADR-006). Model-download progress
   and engine start/stop got two new event types
   (`ModelDownloadProgress/Completed/Failed`, `EngineStarted/Stopped`) so the
   whole platform speaks one event vocabulary — clients never poll.

4. **Settings are schema-driven, not duplicated.** `GET /settings/schema`
   returns `Settings.model_json_schema()` directly (ADR-009); `PATCH
   /settings` deep-merges a partial document through the same
   `eva.config.service` functions `eva config` (new CLI group) calls. One
   validation path, one persistence path.

5. **Plugin lifecycle backend, no loader yet.** `eva/plugins/` adds a
   manifest schema and a `PluginManager` using standard entry points (group
   `eva.plugins`) — genuinely functional discovery/enable/disable/reload,
   even though the discovered list is normally empty until a plugin package
   exists. This fulfills ADR-011's contract incrementally without inventing a
   custom loader now and replacing it later.

6. **Turn control gets one new capability: `interrupt()`.** The orchestrator
   already cancels turns for barge-in/supersede/shutdown (ADR-006); the API
   needed the same action reachable without a microphone. `TurnCancelled`
   gained a `"manual"` reason. `/conversation/interrupt` and `/conversation/cancel`
   are intentionally the same operation under two names — the turn FSM has
   exactly one way to stop a turn, and pretending otherwise would be two
   endpoints hiding one behavior.

7. **Security: localhost-only, no auth, by design — for now.** CORS is
   restricted to `localhost`/`127.0.0.1` origins; no authentication exists
   because there is currently exactly one trust boundary (the user's own
   machine). Every router is separated from auth concerns via the
   `StateDep` dependency indirection specifically so a future auth dependency
   can be inserted at the router level without touching business logic.

8. **Uniform errors.** Every `EvaError` subclass maps to one HTTP status
   (`eva/server/errors.py`); every error response is
   `{"detail": ..., "error_type": ...}`. Pydantic validation errors get the
   same shape with a `errors` list. Clients branch on `error_type`, not on
   parsing prose.

## Alternatives rejected
- **GraphQL** — REST + WebSocket matches the existing event-driven engine
  design (ADR-006) far more directly; no query-shape flexibility is needed for
  a single-user local app.
- **Auto-starting the engine on server boot** — would open audio devices and
  load multi-GB models as a side effect of running `eva serve`, surprising in
  a dev/CI context and wrong for a desktop app that wants to show a "ready to
  start" state first.
- **A single monolithic router file** — one file per concern (Part 11:
  consistent naming/validation/errors per router) scales better as the desktop
  UI, web UI, and plugin surface all grow against this API independently.

## Consequences
- `pyproject.toml` gains `fastapi`, `uvicorn`, `websockets` as base
  dependencies (all ship universal wheels — no clean-install regression, per
  ADR-013's standing rule) and `httpx` as a dev dependency (TestClient
  transport only).
- The desktop shell and web UI (M5+) are pure API consumers from day one —
  no engine code will ever need to move when they are built.
- `docs/API.md` is the human-readable map; `GET /docs` (Swagger UI) and
  `GET /openapi.json` are generated automatically and are the authoritative,
  always-current reference.
