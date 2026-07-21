"""Desktop shell entry point (M6.1, ADR-027).

`main()` supervises the server process (a separate `eva serve`, not in-thread
— ADR-007), restores the last window geometry and route, opens the pywebview
window, and on close persists window state and stops a shell-owned server
gracefully. `pywebview` is an optional extra imported only inside `main()`, so
the base install never needs it and importing this module stays cheap.

The window-facing glue (`main`) is thin; the reusable logic lives in
`ServerSupervisor` (`supervisor.py`), `DesktopState` (`state.py`), and
`DesktopClient` (`client.py`), each unit-tested headless. `capture_window_state`
is kept here but takes a duck-typed window so the geometry→state mapping is
testable without a real window.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from eva.config import get_app_paths, load_settings
from eva.desktop.client import DesktopClient
from eva.desktop.platform import create_platform
from eva.desktop.state import MIN_HEIGHT, MIN_WIDTH, DesktopState
from eva.desktop.supervisor import ServerSupervisor
from eva.desktop.tray import TrayController
from eva.desktop.window import WindowController, should_start_hidden
from eva.service import display_host

logger = logging.getLogger(__name__)


class _WindowLike(Protocol):
    """The pywebview window surface `capture_window_state` reads. Defined
    structurally so tests can pass a plain fake window."""

    width: int
    height: int
    x: int | None
    y: int | None

    def get_current_url(self) -> str | None: ...


def capture_window_state(window: _WindowLike, previous: DesktopState) -> DesktopState:
    """Read geometry + current route off the window into a `DesktopState`.

    Falls back to the previous values for anything the window can't report
    (pywebview does not expose a reliable cross-platform "is maximized", so
    `maximized` is carried over rather than guessed). Never raises — a window
    that misbehaves at close time must not break shutdown.
    """
    try:
        width = int(getattr(window, "width", previous.width)) or previous.width
        height = int(getattr(window, "height", previous.height)) or previous.height
        x = _opt_coord(getattr(window, "x", previous.x))
        y = _opt_coord(getattr(window, "y", previous.y))
    except (TypeError, ValueError):
        width, height, x, y = previous.width, previous.height, previous.x, previous.y
    route = previous.last_route
    try:
        current = window.get_current_url()
        if current:
            route = _route_of(current)
    except Exception:  # broad by design: route memory is cosmetic, never fatal
        logger.debug("Could not read current window URL", exc_info=True)
    return DesktopState(
        width=max(MIN_WIDTH, width),
        height=max(MIN_HEIGHT, height),
        x=x,
        y=y,
        maximized=previous.maximized,
        last_route=route,
    )


def _opt_coord(value: object) -> int | None:
    if isinstance(value, int | float | str):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    return None


def _route_of(url: str) -> str:
    """The SPA hash route (e.g. '#/memory') from a full URL, '' if none."""
    _, _, fragment = url.partition("#")
    return f"#{fragment}" if fragment else ""


def _initial_url(host: str, port: int, state: DesktopState) -> str:
    base = f"http://{display_host(host)}:{port}/"
    return base + state.last_route if state.last_route else base


# Chromium backgrounds a hidden/minimized renderer: it clamps JS timers to ~1 Hz
# and, after a while, freezes the page entirely — which would drop the live
# WebSocket and stall the UI while EVA sits in the tray, then cost a reconnect on
# restore. The engine (a separate process, ADR-007) keeps running regardless, but
# the *window* must too, so minimize-to-tray is "hidden only". These flags — the
# same ones Electron apps use — keep the renderer full-speed while hidden;
# measured to hold timers at full rate with the window minimized. WebView2 reads
# this env var and appends it to pywebview's own arguments.
_WEBVIEW2_NO_BACKGROUNDING = (
    "--disable-background-timer-throttling "
    "--disable-renderer-backgrounding "
    "--disable-backgrounding-occluded-windows"
)
_WEBVIEW2_ARGS_ENV = "WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS"


def _keep_webview_awake_while_hidden() -> None:
    """Stop Chromium throttling/freezing the WebView2 renderer when the window
    is hidden to the tray, so streaming and the WebSocket stay live and restore
    is instant. WebView2/Windows-only; preserves any user-set value. Must run
    before the window (and thus the WebView2 environment) is created."""
    if sys.platform == "win32":
        existing = os.environ.get(_WEBVIEW2_ARGS_ENV, "")
        if "renderer-backgrounding" in existing:  # already configured (user or us)
            return
        os.environ[_WEBVIEW2_ARGS_ENV] = (
            f"{existing} {_WEBVIEW2_NO_BACKGROUNDING}".strip()
            if existing
            else _WEBVIEW2_NO_BACKGROUNDING
        )


# pywebview's file-dialog filter grammar is 'Description (*.ext;*.ext)', where
# the description is validated against `[\w ]+` — words and spaces only. A '/'
# (the earlier "Text/log files") fails with "is not a valid file filter". Keep
# descriptions to words/spaces; a test validates these against pywebview's own
# parser so this can't regress.
_SAVE_FILE_TYPES = ("Log files (*.txt;*.log)", "All files (*.*)")


class DesktopBridge:
    """The small, native-only capability the desktop shell exposes to the web
    UI over pywebview's JS bridge (`window.pywebview.api.*`).

    Kept deliberately tiny: the web UI is still an HTTP/WS client of the engine
    (ADR-007) — this bridge is only for things a browser genuinely cannot do,
    like a native Save-As dialog for an export. The browser build never sees
    `window.pywebview`, so it feature-detects this and falls back to a normal
    download; nothing here is required for the app to work.

    Public method names are exposed verbatim as `window.pywebview.api.<name>`;
    the leading-underscore `_window` is not exposed (pywebview skips it).
    """

    def __init__(self) -> None:
        self._window: Any = None

    def save_text_file(self, suggested_name: str, content: str) -> dict[str, str]:
        """Show a native Save-As dialog and write `content` to the chosen path.

        Returns a small status dict the UI turns into a toast:
        ``{"status": "saved", "path": ...}`` / ``{"status": "cancelled"}`` /
        ``{"status": "error", "message": ...}``. Never raises across the
        bridge — a failure becomes a reported error, not a silent no-op (the
        whole point: the user must always know where the export went)."""
        import webview

        window = self._window
        if window is None:  # bridge called before the window bound itself
            return {"status": "error", "message": "desktop window not ready"}
        try:
            result = window.create_file_dialog(
                webview.FileDialog.SAVE,
                save_filename=suggested_name,
                file_types=_SAVE_FILE_TYPES,
            )
        except Exception as exc:  # broad by design: report, never crash the bridge
            logger.warning("Save dialog failed", exc_info=True)
            return {"status": "error", "message": str(exc)}
        if not result:
            return {"status": "cancelled"}
        path = result[0] if isinstance(result, list | tuple) else result
        try:
            Path(path).write_text(content, encoding="utf-8")
        except OSError as exc:
            logger.warning("Writing exported file failed", exc_info=True)
            return {"status": "error", "message": str(exc)}
        logger.info("Exported file written to %s", path)
        return {"status": "saved", "path": str(path)}


def main() -> int:
    try:
        import webview  # optional extra; guarded so the base install never needs it
    except ImportError:
        # Fail with a remedy, never a bare ModuleNotFoundError (AI_CONTEXT).
        print(
            "The desktop window needs the optional 'desktop' extra.\n"
            '  Install it with:  pip install -e ".[desktop]"\n'
            "Then run `eva-desktop` again. (For a browser instead, use `eva serve --open`.)"
        )
        return 1

    paths = get_app_paths()
    paths.ensure_exists()
    settings = load_settings(paths.settings_file)
    host, port = settings.server.host, settings.server.port

    supervisor = ServerSupervisor(paths, host, port)
    if not supervisor.ensure_running():
        print("error: the EVA server did not start — check `eva logs`.")
        return 1

    state = DesktopState.load(paths)
    if settings.desktop.auto_start_engine:
        DesktopClient(host, port).start_engine()  # best-effort; UI can start it too

    # The tray must exist before the window so we know whether hiding-to-tray
    # is possible (close/minimize-to-tray, start-minimized). A tray problem
    # never stops the app — degrade to windowed-only, logged (also the case
    # when the tray libs are simply absent).
    platform = None
    try:
        platform = create_platform()
    except Exception:
        logger.warning("Tray platform unavailable; continuing without a tray", exc_info=True)
    tray_available = platform is not None
    start_hidden = should_start_hidden(settings.desktop, tray_available=tray_available)

    # Keep the renderer alive while hidden to the tray (see helper) — must be set
    # before the window/WebView2 environment is created.
    _keep_webview_awake_while_hidden()

    # Native-only capability bridge for the web UI (Save-As dialog). Bound to
    # the window right after creation; the browser build never sees it.
    bridge = DesktopBridge()

    # Duck-typed: pywebview's Window API varies by version/backend, and the
    # optional dep is absent in CI — we only use a small, stable surface
    # (show/hide/restore/destroy/events/evaluate_js/geometry), so treat as Any.
    window: Any = webview.create_window(
        "Edge Voice Assistant",
        _initial_url(host, port, state),
        width=state.width,
        height=state.height,
        x=state.x,
        y=state.y,
        min_size=(MIN_WIDTH, MIN_HEIGHT),
        hidden=start_hidden,  # start-minimized to the tray
        js_api=bridge,
    )
    bridge._window = window  # the bridge needs the window for the Save dialog

    controller = WindowController(
        window,
        settings.desktop,
        on_save=lambda: capture_window_state(window, state).save(paths),
        tray_available=tray_available,
    )

    # Wire the pywebview lifecycle events to the controller. `closing` is
    # synchronous and its return value is honored (return False vetoes the
    # close → close-to-tray); `minimized` is a side-effect hook. Event wiring
    # is best-effort — a backend that lacks a hook must not stop the launch.
    _bind_event(window, "closing", controller.on_closing)
    _bind_event(window, "minimized", controller.on_minimized)

    tray: TrayController | None = None
    if platform is not None:
        try:
            tray = TrayController(
                platform,
                on_open=controller.show,
                on_hide=controller.hide,
                on_settings=controller.show_settings,
                on_quit=controller.request_quit,
            )
            supervisor.on_status_change = tray.on_supervisor_status
            tray.start()
        except Exception:
            logger.warning("System tray unavailable; continuing without it", exc_info=True)
            tray = None

    supervisor_thread = threading.Thread(target=supervisor.run, name="eva-supervisor", daemon=True)
    supervisor_thread.start()
    try:
        webview.start()  # blocks the main thread until the window is destroyed
    finally:
        # Deterministic teardown: no hanging threads, no orphan server. Save
        # again here in case the close bypassed the `closing` handler.
        if tray is not None:
            tray.stop()
        capture_window_state(window, state).save(paths)
        supervisor.stop()
        supervisor_thread.join(timeout=5)
    return 0


def _bind_event(window: Any, name: str, handler: Callable[[], Any]) -> None:
    """Subscribe a zero-arg controller handler to a pywebview window event.

    pywebview's Event calls a zero-parameter callable with no args and honors
    its return value (for the synchronous `closing` event). Best-effort: a
    backend missing the event must not prevent the window opening.
    """
    try:
        event = getattr(window.events, name)
        event += handler
    except (AttributeError, TypeError):
        logger.debug("Window event %r unavailable on this backend", name, exc_info=True)
