"""Server process lifecycle: `eva start/stop/restart/status/logs` (M5.5).

A thin daemon-management layer over `eva serve` (still uvicorn inside,
ADR-026): `start` spawns a detached server process and records its PID under
the config dir; `stop` terminates it gracefully (terminate → wait → kill);
`status` combines process liveness with the API's own health/engine state;
`logs` tails the newest log file. No service framework, no registry — a PID
file and psutil (already a dependency), which is exactly enough for a local
single-user app.
"""

from __future__ import annotations

import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import psutil

from eva.config.paths import AppPaths

_HEALTH_TIMEOUT_S = 90.0  # model preload can be slow on first run
_STOP_TIMEOUT_S = 15.0
_PID_FILENAME = "eva-server.pid"
_CONSOLE_LOG = "server-console.log"


def pid_file(paths: AppPaths) -> Path:
    return paths.config_dir / _PID_FILENAME


def read_server_pid(paths: AppPaths) -> int | None:
    """The recorded server PID, or None if absent/stale. A stale file (no
    such process, or the PID reused by an unrelated program) is removed."""
    path = pid_file(paths)
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
    if not psutil.pid_exists(pid):
        path.unlink(missing_ok=True)
        return None
    try:
        # PID reuse guard: it must at least be a python process.
        if "python" not in psutil.Process(pid).name().lower():
            path.unlink(missing_ok=True)
            return None
    except psutil.Error:
        return None
    return pid


def display_host(host: str) -> str:
    """A connectable host for URLs/messages: the wildcard bind ``0.0.0.0``
    means "listen on every interface", which is not an address you can open
    — show ``127.0.0.1`` instead. Any concrete host passes through."""
    return "127.0.0.1" if host == "0.0.0.0" else host


def health_url(host: str, port: int) -> str:
    return f"http://{display_host(host)}:{port}/api/v1/health"


def probe_health(url: str, timeout_s: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            return bool(response.status == 200)
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
        return False


def spawn_server(paths: AppPaths, host: str, port: int) -> int:
    """Start a detached `eva serve` process; returns its PID. Console output
    goes to logs/server-console.log (the app's own file logging is separate,
    via eva.logging_setup)."""
    console_log = paths.logs_dir / _CONSOLE_LOG
    command = [sys.executable, "-m", "eva.cli", "serve", "--host", host, "--port", str(port)]
    with console_log.open("ab") as log_handle:
        if sys.platform == "win32":
            process = subprocess.Popen(
                command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
            )
        else:
            process = subprocess.Popen(
                command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
    pid_file(paths).write_text(str(process.pid), encoding="utf-8")
    return process.pid


def wait_until_healthy(url: str, timeout_s: float = _HEALTH_TIMEOUT_S) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if probe_health(url):
            return True
        time.sleep(0.25)
    return False


def request_graceful_shutdown(host: str, port: int, timeout_s: float = 5.0) -> bool:
    """Ask a running server to stop itself via POST /system/shutdown (M5.6).

    This is the clean path: the engine stops (audio devices released, the
    memory database flushed), then uvicorn exits. Returns True when the
    server accepted the request. On Windows, `Process.terminate()` is
    TerminateProcess — a hard kill with zero cleanup — so this API call is
    the only genuinely graceful stop for a detached background server.
    """
    url = f"http://{display_host(host)}:{port}/api/v1/system/shutdown"
    request = urllib.request.Request(url, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return bool(response.status == 200)
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
        return False


def wait_for_exit(pid: int, timeout_s: float) -> bool:
    """True once `pid` no longer exists (polls; tolerates PID errors)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not psutil.pid_exists(pid):
            return True
        time.sleep(0.2)
    return not psutil.pid_exists(pid)


def terminate_server(paths: AppPaths, pid: int, *, host: str = "", port: int = 0) -> bool:
    """Stop the server: graceful API shutdown first (when host/port are
    known), then terminate → wait → kill. Returns True when the process is
    gone (however it went)."""
    try:
        process = psutil.Process(pid)
    except psutil.NoSuchProcess:
        pid_file(paths).unlink(missing_ok=True)
        return True
    if port and request_graceful_shutdown(host, port) and wait_for_exit(pid, _STOP_TIMEOUT_S):
        pid_file(paths).unlink(missing_ok=True)
        return True
    process.terminate()
    try:
        process.wait(timeout=_STOP_TIMEOUT_S)
    except psutil.TimeoutExpired:
        process.kill()
        try:
            process.wait(timeout=5)
        except psutil.TimeoutExpired:
            return False
    pid_file(paths).unlink(missing_ok=True)
    return True


def newest_log_lines(paths: AppPaths, count: int) -> list[str]:
    """The last `count` lines of the most recently modified log file."""
    log_files = sorted(paths.logs_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not log_files:
        return []
    text = log_files[0].read_text(encoding="utf-8", errors="replace")
    return text.splitlines()[-count:]
