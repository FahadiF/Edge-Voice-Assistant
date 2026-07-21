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
import threading
from collections.abc import Callable
from typing import Any, Protocol

from eva.config import get_app_paths, load_settings
from eva.desktop.client import DesktopClient
from eva.desktop.platform import create_platform
from eva.desktop.state import MIN_HEIGHT, MIN_WIDTH, DesktopState
from eva.desktop.supervisor import ServerSupervisor
from eva.desktop.tray import TrayController
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

    # Duck-typed: pywebview's Window API varies by version/backend, and the
    # optional dep is absent in CI — we only use a small, stable surface
    # (show/hide/destroy/events/evaluate_js/geometry), so treat it as Any.
    window: Any = webview.create_window(
        "Edge Voice Assistant",
        _initial_url(host, port, state),
        width=state.width,
        height=state.height,
        x=state.x,
        y=state.y,
        min_size=(MIN_WIDTH, MIN_HEIGHT),
    )

    def _persist_on_close() -> None:
        capture_window_state(window, state).save(paths)

    # pywebview exposes window lifecycle events as subscribable callbacks, but
    # the exact API differs across versions/backends — a missing hook must not
    # stop the window opening (the finally-block save covers the common case).
    try:
        window.events.closing += _persist_on_close
    except (AttributeError, TypeError):
        logger.debug("Window close-event hook unavailable on this backend", exc_info=True)

    # System tray (M6.2): reflects supervisor state and controls the window.
    # A tray problem must never stop the app opening — degrade to windowed-only
    # (also true when the tray libs are simply absent). Logged, not silent.
    tray: TrayController | None = None
    try:
        tray = _make_tray(window)
        if tray is not None:
            supervisor.on_status_change = tray.on_supervisor_status
            tray.start()
    except Exception:
        logger.warning("System tray unavailable; continuing without it", exc_info=True)
        tray = None

    supervisor_thread = threading.Thread(target=supervisor.run, name="eva-supervisor", daemon=True)
    supervisor_thread.start()
    try:
        webview.start()  # blocks the main thread until the window closes
    finally:
        # Deterministic teardown: no hanging threads, no orphan server.
        if tray is not None:
            tray.stop()
        _persist_on_close()
        supervisor.stop()
        supervisor_thread.join(timeout=5)
    return 0


def _make_tray(window: Any) -> TrayController | None:
    """Build the tray controller with window-driving callbacks, or None if the
    tray libraries aren't installed. Menu actions are best-effort (a failing
    backend call is logged, never propagated into the tray thread)."""
    platform = create_platform()
    if platform is None:
        return None

    def settings() -> None:
        window.show()
        window.evaluate_js("window.location.hash = '#/settings'")

    return TrayController(
        platform,
        on_open=_safe(window.show),
        on_hide=_safe(window.hide),
        on_settings=_safe(settings),
        on_quit=_safe(window.destroy),
    )


def _safe(action: Callable[[], None]) -> Callable[[], None]:
    """Wrap a tray-menu action so a backend hiccup is logged, not raised into
    pystray's thread (which would silently kill the menu)."""

    def wrapped() -> None:
        try:
            action()
        except Exception:  # broad by design: a tray click must never crash the tray
            logger.debug("Tray action failed", exc_info=True)

    return wrapped
