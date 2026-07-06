# ADR-023: Web UI architecture and hosting

Status: Accepted · Date: 2026-07-05

## Context

M5 builds the first full consumer of the platform API (ADR-017): the React
web UI the roadmap has planned since ADR-007, plus — per an explicit scope
decision during M5 planning — a *minimal* desktop shell landing one
milestone earlier than the roadmap's M6 slot (window only; tray, global
hotkey, process supervision, and installers remain M6).

Constraints already decided elsewhere and honored here, not re-decided:
ADR-007 (React + Vite web UI; pywebview desktop shell — Tauri explicitly
rejected; audio stays in the engine process), ADR-009 (settings UI is
generated from the settings JSON Schema, component pickers come from
registries), ADR-017 (UIs are pure API clients; `error_type` field on every
error; localhost-only, no auth), ADR-006 (renderers must drop stale-epoch
artifacts).

## Decision

### 1. Frontend stack (in `web/`)

- **React 18 + TypeScript (strict) + Vite.** Already named by ADR-007 and
  the roadmap; TypeScript strict because the API surface is fully typed on
  the Python side (pydantic) and the frontend mirrors those types 1:1 in
  `web/src/api/types.ts` — a hand-maintained mirror, deliberately: an
  OpenAPI codegen step would add a generator dependency and a build-order
  coupling for ~40 interfaces that change rarely and fail loudly in tests
  when they drift.
- **State: TanStack Query for REST, one zustand store for the WebSocket.**
  REST data (models, personas, settings, ...) is classic cache-and-
  invalidate — TanStack Query is the boring, battle-tested answer. Live
  data (snapshot, pipeline state, transcripts, tokens, download progress)
  arrives over one WebSocket and is *pushed*, not fetched — that's a single
  reducer-style store, not a query cache. Two tools because they are two
  different data lifecycles; forcing either into the other produces
  polling-shaped hacks the architecture explicitly forbids ("clients never
  poll" — ADR-017).
- **Styling: CSS custom properties (design tokens) + CSS modules. No
  Tailwind, no component framework.** The product is fully offline — every
  byte ships in the bundle; a component framework buys speed at the cost of
  a large dependency surface and fighting its theme system to implement
  `settings.ui.theme` (dark/light/system, already a settings field).
  Design tokens in `:root`/`[data-theme="dark"]` make the theme switch one
  attribute flip, and `prefers-color-scheme` handles "system". Dialogs use
  the native `<dialog>` element; other primitives are native elements with
  ARIA attributes — accessibility via the platform, not a library.
- **Routing: react-router**, one route per page (dashboard, conversation,
  memory, personas, users, models, voices, settings, diagnostics,
  plugins).

### 2. Hosting: the API server serves the built UI, when present

ADR-017 said "no static files" — correct for M2.6, when no UI existed. The
roadmap anticipated exactly this gap ("no new backend surface expected,
only whatever small gaps using it in anger reveals"). Serving static files
is not business logic; it does not violate "UIs are API clients" — the UI
still talks to itself over `/api/v1` regardless of who serves its HTML.

`create_app()` mounts the built UI at `/` **only if a dist directory
exists**, resolved in order by `eva.server.static.ui_dist_dir()`:

1. `EVA_UI_DIST` environment variable (explicit override),
2. `src/eva/server/static/ui/` (where packaging copies the build),
3. `<repo>/web/dist/` (developer checkout convenience).

When none exists, the app is byte-for-byte the old API-only app — CI and
headless deployments are unaffected. The mount is an SPA mount: real files
are served as-is; any other non-`/api` GET returns `index.html` so
client-side routes deep-link correctly. `/api/v1/*`, `/docs`, and
`/openapi.json` always win over the SPA fallback.

During development the UI instead runs on Vite's dev server with a proxy
(`/api` → `127.0.0.1:8765`) — the existing CORS regex (any localhost port)
already permits this; no backend change.

### 3. Voice preview: decode raw PCM in the browser

`POST /voices/{id}/preview` returns raw 16 kHz mono int16 PCM (ADR-022
deferred the container question to "a UI-driven decision for M5"). The
decision: **keep the endpoint raw**. The Web Audio API constructs an
`AudioBuffer` from the Int16 samples in ~10 lines; a WAV container would
add backend code to serve exactly one client that doesn't need it. If a
future non-browser client wants WAV, that's its adapter concern.

### 4. Desktop shell (minimal, `eva.desktop`)

One Python module — `src/eva/desktop.py`, console script `eva-desktop` —
not a `desktop/` package tree: it starts uvicorn(create_app) on a
localhost port in a background thread, polls `/health` until ready, opens
one pywebview window at `/`, and shuts the server down when the window
closes. `pywebview` is an optional extra (`pip install
edge-voice-assistant[desktop]`); the base install never imports it.
This is ADR-007's shell, scope-limited: no tray, no hotkey, no supervision,
no installer — those stay in M6 as planned. `eva serve --open` (opens the
default browser) is the zero-extra-dependency alternative.

## Consequences

- The wheel gains optional static assets: packaging copies `web/dist` into
  `src/eva/server/static/ui/` and `pyproject.toml` includes it as package
  data. A source checkout without a frontend build still works (API-only).
- CI grows a Node job (npm ci / lint / test / build). The Python job is
  unchanged.
- `web/src/api/types.ts` is a contract mirror: when a pydantic schema
  changes, the mirrored interface must change with it. Frontend tests pin
  the settings-schema shape via a captured fixture so drift fails a test
  rather than a user.
- The M6 desktop milestone shrinks to: tray, global PTT hotkey, engine
  process supervision, first-run wizard window, installers.

## Amendment (M5.1, 2026-07-05): react-markdown is not a "component framework"

The "no component framework" decision (§1) rejected broad UI kits (Tailwind,
MUI) whose value is a large surface of pre-styled widgets. ADR-024 adds
`react-markdown` + `remark-gfm` for assistant-message rendering. That does
not reverse this decision: they are focused libraries for one bounded
problem (parsing + rendering a documented text format), bundle into the same
offline build, ship no styling of their own (all CSS is ours, themed by our
tokens), and have no lighter in-house alternative — a hand-rolled Markdown
parser would be strictly worse. The bar this ADR set was "don't take a
dependency whose job we could do with tokens + native elements"; a
CommonMark+GFM parser clears it. Bundle cost: ~160 KB min (~50 KB gzip),
accepted for an offline-first app where the download is one-time.
