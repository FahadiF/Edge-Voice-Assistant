"""Subprocess helpers — chiefly: never flash a console window on Windows.

When the server runs detached (`eva start` → no console), spawning a
console program such as ``nvidia-smi`` without ``CREATE_NO_WINDOW`` makes
Windows allocate a fresh console for it, which flashes a black window on
screen. Every external probe goes through :func:`no_window_kwargs` so that
never happens. On POSIX the flag does not exist and this is an empty dict.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Any

# `CREATE_NO_WINDOW` only exists in the Windows build of the subprocess
# module, so it is read dynamically rather than referenced at import time
# (keeps `mypy --platform linux` happy).
_CREATE_NO_WINDOW: int = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def no_window_kwargs() -> dict[str, Any]:
    """`subprocess.run`/`Popen` kwargs that suppress a console window on
    Windows; empty on every other platform."""
    # Explicit if/else so mypy's platform-check exemption covers both
    # branches under `--platform linux` as well as the native run (a bare
    # trailing `return {}` reads as unreachable on Windows). RET505 wants the
    # else gone, but removing it reintroduces the mypy warning.
    if sys.platform == "win32":
        return {"creationflags": _CREATE_NO_WINDOW}
    else:  # noqa: RET505 — required for the cross-platform mypy exemption
        return {}
