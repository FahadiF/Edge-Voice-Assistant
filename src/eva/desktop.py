"""Minimal desktop shell (ADR-007, ADR-023).

Not a second application: this starts the same `create_app()` FastAPI
server uvicorn already runs for `eva serve`, on a background thread, and
opens one native OS window (`pywebview`) pointing at it. No tray, no global
hotkey, no process supervision, no installer — those are M6 (ADR-023 scope
note). `pywebview` is an optional extra (`pip install
edge-voice-assistant[desktop]`); this module is never imported by the base
CLI/server code path.
"""

from __future__ import annotations

import socket
import threading
import time
import urllib.error
import urllib.request

import uvicorn

from eva.config import get_app_paths, load_settings
from eva.server import create_app

_HEALTH_TIMEOUT_S = 30.0
_HEALTH_POLL_INTERVAL_S = 0.1


def find_free_port(host: str = "127.0.0.1") -> int:
    """An OS-assigned free TCP port — avoids colliding with an already-
    running `eva serve` on the configured port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        port: int = sock.getsockname()[1]
        return port


def wait_for_health(url: str, *, timeout_s: float = _HEALTH_TIMEOUT_S) -> bool:
    """Poll `url` until it responds or `timeout_s` elapses."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(_HEALTH_POLL_INTERVAL_S)
    return False


def _run_server(host: str, port: int) -> None:
    paths = get_app_paths()
    paths.ensure_exists()
    config = uvicorn.Config(create_app(paths), host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    server.run()


def main() -> int:
    import webview  # optional extra; import guarded so the base install never needs it

    paths = get_app_paths()
    paths.ensure_exists()
    settings = load_settings(paths.settings_file)
    host = settings.server.host if settings.server.host != "0.0.0.0" else "127.0.0.1"
    port = find_free_port(host)

    server_thread = threading.Thread(target=_run_server, args=(host, port), daemon=True)
    server_thread.start()

    url = f"http://{host}:{port}/"
    if not wait_for_health(f"http://{host}:{port}/api/v1/health"):
        print(f"error: backend did not become ready at {url}")
        return 1

    webview.create_window("Edge Voice Assistant", url, width=1200, height=800, min_size=(800, 600))
    webview.start()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
