"""Window lifecycle controller (M6.2, ADR-027).

Encapsulates the *decisions* behind close-to-tray, minimize-to-tray,
start-minimized, and tray-driven quit, separated from the pywebview event
plumbing so they are unit-tested against a fake window. The shell subscribes
the pywebview `closing`/`minimized` events to this controller and routes the
tray's Open/Hide/Settings/Quit through it too.

pywebview specifics this relies on (verified against pywebview 6.2.1):
- `closing` is a *synchronous* event whose handlers' return value is honored —
  returning ``False`` cancels the close. That is how close-to-tray vetoes the
  window's X button while `request_quit()` (tray Quit) still exits.
- `minimized` fires as a side-effect-only event (return ignored); hiding the
  window there realizes minimize-to-tray.
- A window hidden while minimized is brought back with `restore()` + `show()`.

The window is duck-typed (`_WindowLike`) — the shell passes the real pywebview
window; tests pass a fake that records calls.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Protocol

from eva.config.settings import DesktopSettings

logger = logging.getLogger(__name__)


class _WindowLike(Protocol):
    def show(self) -> None: ...
    def hide(self) -> None: ...
    def restore(self) -> None: ...
    def destroy(self) -> None: ...
    def evaluate_js(self, script: str) -> None: ...


class WindowController:
    def __init__(
        self,
        window: _WindowLike,
        settings: DesktopSettings,
        *,
        on_save: Callable[[], None],
        tray_available: bool,
    ) -> None:
        self._window = window
        self._settings = settings
        self._on_save = on_save
        # Hiding to the tray only makes sense if a tray exists to hide into;
        # without one, close/minimize keep their normal OS behavior.
        self._tray_available = tray_available
        self._quitting = False

    # ── pywebview event handlers ──

    def on_closing(self) -> bool:
        """`closing` handler. Return True to allow the close, False to veto it.

        Tray-driven quit always closes. Otherwise, with close-to-tray on (and a
        tray to hide into), the X button hides the window and vetoes the close;
        a real close persists window state first.
        """
        if self._quitting:
            self._on_save()
            return True
        if self._settings.close_to_tray and self._tray_available:
            logger.debug("Close-to-tray: hiding the window instead of quitting")
            self._window.hide()
            return False
        self._on_save()
        return True

    def on_minimized(self) -> None:
        """`minimized` handler — hide to the tray when configured."""
        if self._settings.minimize_to_tray and self._tray_available:
            logger.debug("Minimize-to-tray: hiding the minimized window")
            self._window.hide()

    # ── tray-driven actions ──

    def show(self) -> None:
        """Bring the window back (from hidden and/or minimized) and focus it.

        Order is load-bearing and was measured against pywebview 6.2.1 winforms:
        a window hidden while minimized sits at ``Visible=False,
        WindowState=Minimized``. ``Form.Show()`` re-applies the *last shown*
        window state, so ``restore()`` (WindowState=Normal) followed by
        ``show()`` gets clobbered back to Minimized — the window ends up
        visible-but-minimized and never appears (the exact reported bug). The
        correct sequence is ``show()`` (make visible) → ``restore()``
        (un-minimize) → ``show()`` (Activate for foreground focus; safe now that
        the state is Normal, so it does not re-minimize).
        """
        self._window.show()
        self._window.restore()
        self._window.show()

    def hide(self) -> None:
        self._window.hide()

    def show_settings(self) -> None:
        self.show()
        self._window.evaluate_js("window.location.hash = '#/settings'")

    def request_quit(self) -> None:
        """Tray Quit: bypass close-to-tray and really exit (persist first)."""
        self._quitting = True
        self._on_save()
        self._window.destroy()


def should_start_hidden(settings: DesktopSettings, *, tray_available: bool) -> bool:
    """Whether to create the window hidden (start-minimized to the tray).

    Only when start-minimized is set AND there's a tray to hide into — without
    one there'd be no way to bring the window back. Kept as a pure function so
    the shell can decide before the window (and controller) exist, and so it is
    unit-tested directly.
    """
    return settings.start_minimized and tray_available
