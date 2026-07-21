"""System-tray controller (M6.2, ADR-027).

Pure logic: it maps `ServerSupervisor` state onto a tray icon + status text,
builds the tray menu, and routes menu clicks to injected callbacks (show/hide
window, open settings, quit). It holds no engine logic and talks to nothing
native directly — it drives a `DesktopPlatform` and calls back into the shell,
so the whole thing is unit-tested against a fake platform. The tray reflects
*supervisor* state (is the backend up), which the supervisor already computes;
it subscribes to `on_status_change` rather than polling.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from eva.desktop.platform import DesktopPlatform, TrayIconState, TrayMenuItem, TraySpec
from eva.desktop.supervisor import SupervisorStatus

logger = logging.getLogger(__name__)

_TITLE = "Edge Voice Assistant"

# Supervisor state → the icon shown and the human status text. Kept as data so
# the mapping is trivially testable and has no branching in the render path.
_ICON_BY_STATUS: dict[SupervisorStatus, TrayIconState] = {
    SupervisorStatus.STOPPED: TrayIconState.STOPPED,
    SupervisorStatus.RUNNING: TrayIconState.RUNNING,
    SupervisorStatus.ATTACHED: TrayIconState.RUNNING,
    SupervisorStatus.RESTARTING: TrayIconState.STARTING,
    SupervisorStatus.FAILED: TrayIconState.ERROR,
}
_TEXT_BY_STATUS: dict[SupervisorStatus, str] = {
    SupervisorStatus.STOPPED: "Stopped",
    SupervisorStatus.RUNNING: "Running",
    SupervisorStatus.ATTACHED: "Running",
    SupervisorStatus.RESTARTING: "Starting…",
    SupervisorStatus.FAILED: "Error",
}


class TrayController:
    def __init__(
        self,
        platform: DesktopPlatform,
        *,
        on_open: Callable[[], None],
        on_hide: Callable[[], None],
        on_settings: Callable[[], None],
        on_quit: Callable[[], None],
        initial_status: SupervisorStatus = SupervisorStatus.STOPPED,
    ) -> None:
        self._platform = platform
        self._on_open = on_open
        self._on_hide = on_hide
        self._on_settings = on_settings
        self._on_quit = on_quit
        self._status = initial_status

    def start(self) -> None:
        """Show the tray and apply the current status."""
        self._platform.start_tray(TraySpec(title=_TITLE, menu=self._menu()))
        self._render()

    def on_supervisor_status(self, status: SupervisorStatus) -> None:
        """Subscribed to `ServerSupervisor.on_status_change` — no polling."""
        self._status = status
        self._render()

    def stop(self) -> None:
        self._platform.stop_tray()

    # ── internals ──

    def _render(self) -> None:
        self._platform.set_tray_state(icon_for(self._status), text_for(self._status))

    def _status_line(self) -> str:
        return f"Engine: {text_for(self._status)}"

    def _menu(self) -> tuple[TrayMenuItem, ...]:
        return (
            TrayMenuItem("Open", self._on_open),
            TrayMenuItem("Hide", self._on_hide),
            # Live, non-clickable status line (re-rendered when the menu opens).
            TrayMenuItem(self._status_line, separator_before=True),
            TrayMenuItem("Settings", self._on_settings, separator_before=True),
            TrayMenuItem("Quit", self._on_quit, separator_before=True),
        )


def icon_for(status: SupervisorStatus) -> TrayIconState:
    return _ICON_BY_STATUS.get(status, TrayIconState.STOPPED)


def text_for(status: SupervisorStatus) -> str:
    return _TEXT_BY_STATUS.get(status, "Stopped")
