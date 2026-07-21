"""Thin HTTP client the desktop shell uses to drive the engine (M6.1, ADR-027).

The desktop app is *another client* (ADR-007): it never imports engine
internals, it calls the same `/api/v1` endpoints the web UI and CLI use. This
module is the single, tested boundary for those calls; later phases (tray,
hotkey) grow it with more actions, always mapping to an existing endpoint.
The network opener is injected so the mapping is unit-tested without a server.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from collections.abc import Callable
from typing import Any

from eva.service import display_host

logger = logging.getLogger(__name__)

# (url, data, method) → an object with .read()/.status usable as a context manager.
Opener = Callable[..., Any]


class DesktopClient:
    def __init__(self, host: str, port: int, *, opener: Opener | None = None) -> None:
        self._base = f"http://{display_host(host)}:{port}/api/v1"
        self._opener = opener or urllib.request.urlopen

    def start_engine(self) -> bool:
        """POST /engine/start (auto-start on launch). Best-effort: a failure is
        logged, never fatal — the user can still start the engine from the UI."""
        return self._post("/engine/start")

    def _post(self, path: str, payload: dict[str, Any] | None = None) -> bool:
        data = json.dumps(payload).encode() if payload is not None else b""
        request = urllib.request.Request(
            self._base + path,
            data=data,
            method="POST",
            headers={"Content-Type": "application/json"} if payload is not None else {},
        )
        try:
            with self._opener(request, timeout=120) as response:
                return bool(200 <= response.status < 300)
        except Exception:  # broad by design: a convenience call must never be fatal
            logger.warning("Desktop client POST %s failed", path, exc_info=True)
            return False
