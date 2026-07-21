"""Native desktop shell (M6, ADR-007/ADR-023/ADR-027).

The shell wraps the existing web UI in a native window and supervises the
server process — it is another HTTP/WS client of the engine, never an
importer of engine internals. `pywebview` and the tray/hotkey libraries are
optional extras (`pip install edge-voice-assistant[desktop]`); nothing in the
base CLI/server path imports this package.

`main` is re-exported so the console entry point stays `eva.desktop:main`.
"""

from eva.desktop.shell import main

__all__ = ["main"]
