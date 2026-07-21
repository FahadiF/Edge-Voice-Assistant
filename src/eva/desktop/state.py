"""Persisted desktop window state (M6.1, ADR-027).

Remembers window geometry, maximized state, and the last-open page across
launches — the small bits of native-shell state that don't belong in the
engine's `Settings` (they're per-machine window chrome, not configuration a
user tweaks). Stored as JSON next to `setup_state.json` in the config dir,
following the same forgiving load pattern: a missing or corrupt file yields
sensible defaults, never an error.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from eva.config.paths import AppPaths

logger = logging.getLogger(__name__)

_FILENAME = "desktop_state.json"

# Matches the previous hard-coded window size, so first launch is unchanged.
DEFAULT_WIDTH = 1200
DEFAULT_HEIGHT = 800
MIN_WIDTH = 800
MIN_HEIGHT = 600


@dataclass
class DesktopState:
    """Restorable window state. `x`/`y` are None until the window has been
    moved (None ⇒ let the OS center it); `last_route` is the SPA hash route
    (e.g. ``#/memory``) so the app reopens where the user left off."""

    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    x: int | None = None
    y: int | None = None
    maximized: bool = False
    last_route: str = ""

    @classmethod
    def load(cls, paths: AppPaths) -> DesktopState:
        path = _state_path(paths)
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.debug("Ignoring unreadable desktop state at %s", path)
            return cls()
        return cls(
            width=_positive_int(data.get("width"), DEFAULT_WIDTH, MIN_WIDTH),
            height=_positive_int(data.get("height"), DEFAULT_HEIGHT, MIN_HEIGHT),
            x=_opt_int(data.get("x")),
            y=_opt_int(data.get("y")),
            maximized=bool(data.get("maximized", False)),
            last_route=str(data.get("last_route", "")),
        )

    def save(self, paths: AppPaths) -> None:
        path = _state_path(paths)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
        except OSError:
            # Losing window geometry is cosmetic — never let it break shutdown.
            logger.debug("Could not persist desktop state to %s", path, exc_info=True)


def _state_path(paths: AppPaths) -> Path:
    return paths.config_dir / _FILENAME


def _positive_int(value: object, default: int, minimum: int) -> int:
    """A stored dimension, clamped to a sane minimum — a corrupt tiny/zero
    size must never produce an unusable window."""
    if isinstance(value, int | float | str):
        try:
            return max(minimum, int(value))
        except (TypeError, ValueError):
            return default
    return default


def _opt_int(value: object) -> int | None:
    if isinstance(value, int | float | str):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    return None
