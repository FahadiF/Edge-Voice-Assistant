"""DesktopPlatform port + native adapter for OS-shell integration (M6.2, ADR-027).

The pieces of the desktop shell that touch the operating system — the system
tray now, global hotkeys/notifications later — sit behind this port so the
shell's *logic* (which menu item does what, which icon a state maps to) is
unit-tested headless against a fake, and only the thin OS binding is
manual-tested. It is a UI/OS boundary, never business logic: the tray drives
the engine exclusively through the shell's existing HTTP/supervisor seams
(ADR-007/ADR-027).

`create_platform()` returns the real pystray-backed adapter, or ``None`` when
the optional desktop libraries aren't installed — the window still works
without a tray (graceful degradation over hard failure).
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


class TrayIconState(StrEnum):
    """The visual states the tray icon can show (mapped from supervisor state
    by the controller, so this stays UI-only)."""

    STARTING = "starting"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass(frozen=True)
class TrayMenuItem:
    """One tray menu entry. `label` may be a callable so a live value (e.g. the
    engine-status line) re-renders each time the menu opens. `on_activate` None
    means a non-clickable label; `separator_before` inserts a divider above."""

    label: str | Callable[[], str]
    on_activate: Callable[[], None] | None = None
    separator_before: bool = False


@dataclass(frozen=True)
class TraySpec:
    title: str
    menu: tuple[TrayMenuItem, ...]


class DesktopPlatform(ABC):
    """OS-native shell surface. Extended in later M6 phases (hotkey, notify);
    M6.2 defines the tray surface."""

    @abstractmethod
    def start_tray(self, spec: TraySpec) -> None:
        """Create and show the tray icon and run its event loop (off the main
        thread). Idempotent-safe to call once per shell run."""

    @abstractmethod
    def set_tray_state(self, icon: TrayIconState, status_text: str) -> None:
        """Update the tray icon and hover text to reflect the current state."""

    @abstractmethod
    def stop_tray(self) -> None:
        """Remove the tray icon and join its thread. Idempotent."""


# Colors for the generated status dot (no binary assets shipped — the adapter
# draws the icon so the repo stays source-only).
_ICON_RGB: dict[TrayIconState, tuple[int, int, int]] = {
    TrayIconState.STARTING: (0xE0, 0xA0, 0x30),  # amber
    TrayIconState.RUNNING: (0x3F, 0xB9, 0x50),  # green
    TrayIconState.STOPPED: (0x8A, 0x8A, 0x8A),  # grey
    TrayIconState.ERROR: (0xDC, 0x26, 0x26),  # red
}


class PystrayDesktopPlatform(DesktopPlatform):
    """Real adapter over `pystray` (tray) + `pillow` (icon rendering). All
    library use is confined here and imported lazily, so importing this module
    never requires the optional desktop extra."""

    def __init__(self) -> None:
        self._icon: Any = None
        self._thread: threading.Thread | None = None

    def start_tray(self, spec: TraySpec) -> None:
        import pystray

        items: list[Any] = []
        for entry in spec.menu:
            if entry.separator_before:
                items.append(pystray.Menu.SEPARATOR)
            items.append(self._menu_item(pystray, entry))
        self._icon = pystray.Icon(
            "edge-voice-assistant",
            self._image(TrayIconState.STOPPED),
            spec.title,
            pystray.Menu(*items),
        )
        # pystray's own loop blocks; run it on a daemon thread so it lives
        # alongside the pywebview main-thread loop and dies with the process.
        self._thread = threading.Thread(target=self._icon.run, name="eva-tray", daemon=True)
        self._thread.start()

    def set_tray_state(self, icon: TrayIconState, status_text: str) -> None:
        if self._icon is None:
            return
        self._icon.icon = self._image(icon)
        self._icon.title = f"Edge Voice Assistant — {status_text}"

    def stop_tray(self) -> None:
        if self._icon is not None:
            self._icon.stop()  # thread-safe; unblocks the loop
            self._icon = None
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None

    @staticmethod
    def _menu_item(pystray: Any, entry: TrayMenuItem) -> Any:
        label = entry.label
        text: Any = (lambda _item, f=label: f()) if callable(label) else label
        action = entry.on_activate
        handler = (lambda _icon, _item, a=action: a()) if action is not None else None
        return pystray.MenuItem(text, handler, enabled=action is not None)

    @staticmethod
    def _image(state: TrayIconState) -> Any:
        from PIL import Image, ImageDraw

        size = 64
        image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.ellipse((8, 8, size - 8, size - 8), fill=_ICON_RGB[state])
        return image


def create_platform() -> DesktopPlatform | None:
    """The native platform adapter, or None if the desktop extra is absent —
    the shell then runs windowed without a tray (graceful degradation)."""
    import importlib.util

    if importlib.util.find_spec("pystray") is None or importlib.util.find_spec("PIL") is None:
        logger.info("Tray unavailable (install the 'desktop' extra for the system tray)")
        return None
    return PystrayDesktopPlatform()
