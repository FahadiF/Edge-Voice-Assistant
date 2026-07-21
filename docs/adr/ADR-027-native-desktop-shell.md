# ADR-027: Native desktop shell — supervision, window state, and the client boundary

Status: Accepted · Date: 2026-07-12

## Context

M6 turns EVA from a development server into a polished desktop app for daily
use. M5 shipped a *minimal* `eva-desktop`: it opened one pywebview window
pointing at a FastAPI server it hosted **in-process on a background thread**.
That was enough to prove the window works, but it cannot support what M6 needs:

- If the server (or a component inside it, e.g. the LLM) dies, an in-process
  server dies with the shell — there is nothing to restart, and a crashed
  llama.cpp can leave VRAM debt that only a fresh process reclaims.
- The tray, global hotkey, and notifier (M6.2–M6.5) must drive the engine from
  outside the window; hosting the server in the GUI thread entangles GUI and
  engine lifecycles.
- ADR-007 already decided the desktop app is *a separate process talking to the
  engine over HTTP/WS*, never an importer of engine internals. The in-thread
  shell technically imported `create_app`, drifting from that decision.

This ADR sets the desktop shell's architecture for the whole of M6. It is
introduced with M6.1 (supervision + window state); later phases (tray, hotkey,
notifications, first-run wizard, installer) build on the seams defined here.

## Decisions

### 1. The shell is another client; it supervises the server as a separate process

`eva/desktop/` (promoted from the single `desktop.py` module) runs the same
`eva serve` the CLI runs — via `eva.service.spawn_server` — and talks to it only
over `/api/v1` + `/ws`. No engine internals are imported (ADR-007 honored in
fact, not just intent). `ServerSupervisor` owns the process lifecycle and reuses
**every** `eva.service` primitive (`probe_health`, `spawn_server`,
`wait_until_healthy`, `read_server_pid`, `terminate_server`); no lifecycle logic
is reinvented.

### 2. Attach-or-spawn — one server, one source of truth

On launch the supervisor **attaches** to an already-healthy EVA server (e.g.
one started with `eva start`) and leaves it running on quit; otherwise it
**spawns** one, **owns** it, and stops it gracefully on quit
(`terminate_server`, which itself prefers the M5.6 API shutdown before
terminating). This keeps a single backend as the source of truth (M6 goal)
instead of racing a second server onto the same port.

### 3. Owned-server crash recovery is bounded (amends ADR-026's recovery model)

A shell-owned server is health-polled on a daemon thread. If it dies, the
supervisor restarts it with **capped exponential backoff**, bounded by a
consecutive-failure limit: a server that crashes on every boot becomes a
reported `FAILED` state, never an infinite restart loop. A server that restarts
and *stays* up clears the failure counter. An *attached* (externally-started)
server that vanishes is reported, not fought — the shell never restarts a
process it did not start. This extends ADR-026 (which supervised only ASR/TTS
*inside* the engine) to the server-process layer, and is the honest home for
LLM crash recovery (process restart reclaims VRAM; in-process llama reload does
not — completed in a later M6 phase).

### 4. OS-native pieces sit behind ports with fakes; window chrome is persisted state

The pieces that cannot run headless (tray, global hotkey, notifications,
autostart — M6.2+) will sit behind a `DesktopPlatform` port with real adapters
**and a fake**, so their logic is unit-tested in CI. M6.1 establishes the
testable-seam pattern: `ServerSupervisor` (injected `service` + `sleep`),
`DesktopState` (window geometry/route persistence, same forgiving-load pattern
as `SetupState`), and `DesktopClient` (injected opener) are each exercised with
fakes/fake-clocks; the only untestable-here code is the thin `main()` glue and
the pywebview calls, covered by a `MANUAL_TESTING.md` checklist. Window size,
position, and last-open route persist to `desktop_state.json`; the window theme
is **not** duplicated — it reuses `ui.theme`.

### 5. Native-shell configuration is a `DesktopSettings` schema section

`close_to_tray`, `minimize_to_tray`, `start_minimized`, `start_with_os`,
`auto_start_engine`, `notifications_enabled`, `hotkey_enabled`,
`hotkey_binding`, `hotkey_mode` — each a schema field with a description (so the
schema-driven Settings UI renders them automatically) and bounds where numeric.
They only take effect under `eva-desktop`; the CLI/server ignore them. Several
activate in later M6 phases (tray/hotkey/notifications/autostart); they are
declared together as one cohesive section (the same forward-declaration pattern
ADR-025 used for the permissions section), and every field is wired by the end
of M6. `auto_start_engine` is honored from M6.1 — and stays consistent with
"the engine never auto-starts as a side effect": it is an explicit, opt-in
action the *shell* performs via `POST /engine/start`, not a server behavior.

## Rationale

- **Single mechanism, reused:** process lifecycle already exists and is tested
  in `eva.service` (M5.5/M5.6). Supervision composes those primitives rather
  than duplicating spawn/terminate/health logic — the "one mechanism per
  concern" rule.
- **Testability without a display:** confining OS calls to injected
  adapters/openers means the supervision state machine, window-state
  round-trip, and client action-mapping are all covered headless; only
  irreducibly-GUI code needs manual validation.
- **Graceful by construction:** attach-or-spawn + owned-only teardown means the
  shell never orphans a server nor kills one the user manages, and the bounded
  backoff means a broken server surfaces as a message, not a CPU-spinning loop.

## Alternatives rejected

- **Keep hosting the server in-thread (M5 approach).** Rejected: cannot recover
  a crashed server/LLM, entangles GUI and engine lifecycles, and drifts from
  ADR-007. The whole point of M6's reliability goals is defeated by it.
- **Spawn a second server on a random free port, isolated from `eva start`.**
  Rejected: produces two backends and two memory databases racing for the same
  models — contradicts "one backend is the source of truth." Attach-or-spawn on
  the configured port is the single-source design.
- **Import the engine directly for speed (skip HTTP).** Rejected outright by
  ADR-007 — it would couple the desktop app to engine internals and break the
  "desktop/web/CLI are equal clients of one API" invariant.
- **Native-widget wizard / native settings dialogs.** Deferred to ADR-028: the
  web UI is the single presentation layer; the wizard is a React route.

## Consequences

- The desktop shell becomes a small, well-seamed package that later M6 phases
  extend by adding controllers behind the `DesktopPlatform` port — no rework of
  the supervision or client boundary.
- `eva start`/`eva stop`/`eva status` and the desktop app now share one PID
  file and port by design; a user can start the server either way and the
  other tool sees it. This is intended (single source of truth), and is why the
  shell distinguishes *owned* from *attached* servers.
- pywebview's lack of a portable "is maximized" / reliable geometry-on-close API
  means maximized-state restore is best-effort (carried over rather than
  guessed); documented as a manual-test caveat.
- The optional `[desktop]` extra will gain LGPL-3.0 dependencies (pystray,
  pynput) in M6.2/M6.3. LGPL is acceptable here — they are optional,
  dynamically-imported, and replaceable, so they do not affect the Apache-2.0
  licensing of EVA itself — but the choice is flagged deliberately per ADR-004's
  license discipline and revisited if a permissively-licensed equivalent proves
  viable.

## Amendment (M6.2) — System tray realized

The `DesktopPlatform` seam anticipated in Decision 4 is now concrete for the
tray:

- **Port:** `DesktopPlatform` (`platform.py`) with `start_tray` /
  `set_tray_state` / `stop_tray` over `TraySpec` / `TrayMenuItem` /
  `TrayIconState` data types. One real adapter (`PystrayDesktopPlatform`, all
  pystray/PIL use lazily imported and confined to it) and a `FakeDesktopPlatform`
  in the tests.
- **Controller:** `TrayController` (`tray.py`) is pure logic — it maps
  `SupervisorStatus` → icon + status text (a data table, no branching in the
  render path), builds the menu (Open / Hide / live "Engine: <status>" line /
  Settings / Quit), and routes clicks to injected shell callbacks
  (window show/hide, navigate-to-settings, quit). It holds no engine logic and
  never touches anything native directly.
- **State source:** the tray reflects *supervisor* state (is the backend up),
  which the supervisor already computes. `ServerSupervisor` gained an
  `on_status_change` callback fired only on transitions; the tray subscribes to
  it, so state is **pushed, never polled** (the supervisor's existing health
  loop is the only thing checking).
- **Icons:** drawn at runtime with PIL (a colored status dot) — no binary
  assets are shipped, keeping the repo source-only.
- **Graceful degradation:** `create_platform()` returns `None` when the desktop
  extra is absent; the window then runs tray-less rather than failing.

The tray drives the window and quit through the shell's callbacks and the
engine only through the existing supervisor/API — no desktop-specific business
logic was introduced. `[desktop]` now includes `pystray` (LGPL-3.0, as flagged
in Consequences) and `pillow`.

### Amendment (M6.2, cont.) — Window lifecycle controller

Close-to-tray, minimize-to-tray, and start-minimized are realized by a
`WindowController` (`window.py`), separated from pywebview event plumbing so
the decisions are unit-tested against a fake window (the first M6.2 pass
shipped these settings without the interception glue, so they silently did
nothing — the missing tests are why). It relies on pywebview's synchronous
`closing` event (a handler returning `False` vetoes the close → hide-to-tray)
and the side-effect `minimized` event (hide-to-tray). Tray **Quit** sets a
quitting flag so it always exits regardless of close-to-tray; all three
hide behaviors are no-ops without a tray (nowhere to hide → normal OS
behavior).

### Amendment (M6.2, cont.) — Restore path fixed (measured), tray activation

The first lifecycle pass could hide the window but not reliably bring it back.
Two causes, both **measured** against pywebview 6.2.1 (winforms/EdgeChromium)
and pystray 0.19.5, not guessed:

- **Restore order.** A window hidden while minimized sits at
  `Visible=False, WindowState=Minimized`. `Form.Show()` re-applies the *last
  shown* window state, so the old `restore()`→`show()` order set `Normal` and
  then `Show()` clobbered it back to `Minimized` — the window became
  visible-but-minimized and never appeared. The correct, verified sequence is
  `show()` → `restore()` → `show()` (make visible, un-minimize, then re-activate
  for focus — the trailing `show()` is safe once the state is `Normal`). The
  cross-thread call itself is fine: pywebview's window methods self-marshal onto
  the GUI thread via `Control.Invoke`, so no manual thread hop is needed.
- **Tray activation.** pystray fires a menu item on plain left-click only if it
  is the `default` item; with none set, clicking the icon did nothing. The
  "Restore Window" item (renamed from "Open") is now the `default`, so both
  left-click and the menu entry restore the window.

The restore sequence is asserted in `WindowController` unit tests (call order)
and the `default`-item wiring in both the fake-platform and real-pystray tests;
the visible outcome stays on the `MANUAL_TESTING.md` Windows checklist.
