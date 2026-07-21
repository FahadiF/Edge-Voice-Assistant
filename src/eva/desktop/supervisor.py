"""Server-process supervision for the desktop shell (M6.1, ADR-027).

The desktop shell is *another client* of the engine (ADR-007): it does not
host the server in-process, it runs the same `eva serve` the CLI does and
talks to it over HTTP/WS. `ServerSupervisor` owns that process's lifecycle,
reusing `eva.service` for every primitive (spawn/health/terminate) — no
lifecycle logic is reinvented here.

Attach-or-spawn keeps a single source of truth: if a healthy EVA server is
already running (e.g. the user ran `eva start`), the shell *attaches* to it
and leaves it alone on quit; otherwise the shell *spawns* one, *owns* it, and
stops it gracefully on quit. A server the shell owns is health-polled and, if
it dies, restarted with capped exponential backoff — bounded by a
consecutive-failure limit so a server that crashes on every start becomes a
reported failure, never an infinite restart loop.

All OS/network work goes through an injected `_ServiceLike` (default:
`eva.service`) and an injected `sleep`, so the whole state machine is
exercised headless with a fake service and a fake clock.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from enum import StrEnum
from typing import Protocol, cast

from eva import service as _service_module
from eva.config.paths import AppPaths

logger = logging.getLogger(__name__)


class SupervisorStatus(StrEnum):
    STOPPED = "stopped"
    RUNNING = "running"  # we spawned it and it is healthy
    ATTACHED = "attached"  # a pre-existing server we did not start
    RESTARTING = "restarting"
    FAILED = "failed"  # gave up after too many consecutive restart failures


class _ServiceLike(Protocol):
    """The subset of `eva.service` the supervisor uses (injected for tests)."""

    def health_url(self, host: str, port: int) -> str: ...
    def probe_health(self, url: str, timeout_s: float = ...) -> bool: ...
    def spawn_server(self, paths: AppPaths, host: str, port: int) -> int: ...
    def wait_until_healthy(self, url: str, timeout_s: float = ...) -> bool: ...
    def read_server_pid(self, paths: AppPaths) -> int | None: ...
    def terminate_server(
        self, paths: AppPaths, pid: int, *, host: str = ..., port: int = ...
    ) -> bool: ...


class ServerSupervisor:
    def __init__(
        self,
        paths: AppPaths,
        host: str,
        port: int,
        *,
        service: _ServiceLike | None = None,
        sleep: Callable[[float], None] | None = None,
        poll_interval_s: float = 2.0,
        startup_timeout_s: float = 90.0,
        backoff_base_s: float = 1.0,
        backoff_max_s: float = 30.0,
        max_consecutive_failures: int = 5,
    ) -> None:
        self._paths = paths
        self._host = host
        self._port = port
        self._service: _ServiceLike = service or cast("_ServiceLike", _service_module)
        self._sleep = sleep or threading.Event().wait  # interruptible sleep by default
        self._poll_interval_s = poll_interval_s
        self._startup_timeout_s = startup_timeout_s
        self._backoff_base_s = backoff_base_s
        self._backoff_max_s = backoff_max_s
        self._max_failures = max_consecutive_failures
        self._owned = False
        self._failures = 0
        self._stop = threading.Event()
        self.status = SupervisorStatus.STOPPED

    @property
    def health_url(self) -> str:
        return self._service.health_url(self._host, self._port)

    @property
    def owns_server(self) -> bool:
        """True when the shell started the server (and must stop it on quit)."""
        return self._owned

    def ensure_running(self) -> bool:
        """Attach to an already-healthy server, or spawn and own a new one.
        Returns True when a server is up and reachable."""
        if self._service.probe_health(self.health_url):
            self._owned = False
            self.status = SupervisorStatus.ATTACHED
            logger.info("Attached to an already-running server at %s", self.health_url)
            return True
        logger.info("Starting the EVA server (%s)", self.health_url)
        self._service.spawn_server(self._paths, self._host, self._port)
        self._owned = True
        if self._service.wait_until_healthy(self.health_url, self._startup_timeout_s):
            self.status = SupervisorStatus.RUNNING
            self._failures = 0
            return True
        self.status = SupervisorStatus.FAILED
        logger.error("Server did not become healthy within %.0fs", self._startup_timeout_s)
        return False

    def check(self) -> SupervisorStatus:
        """One supervision step: health-check and, for an owned server that
        died, one backoff-guarded restart attempt. Returns the new status.

        - A healthy server clears the failure counter (a server that restarts
          and *stays* up recovers fully).
        - An attached (external) server that vanished is reported STOPPED and
          left alone — the shell does not fight a server it did not start.
        - An owned server that keeps failing hits the consecutive-failure cap
          and becomes FAILED, so a crash-on-boot server never loops forever.
        """
        if self._stop.is_set():
            return self.status
        if self._service.probe_health(self.health_url):
            self._failures = 0
            self.status = SupervisorStatus.RUNNING if self._owned else SupervisorStatus.ATTACHED
            return self.status
        if not self._owned:
            self.status = SupervisorStatus.STOPPED
            return self.status
        if self._failures >= self._max_failures:
            self.status = SupervisorStatus.FAILED
            logger.error("Server failed %d times in a row — not restarting again", self._failures)
            return self.status
        self._failures += 1
        self.status = SupervisorStatus.RESTARTING
        delay = min(self._backoff_max_s, self._backoff_base_s * (2 ** (self._failures - 1)))
        logger.warning(
            "Server is down — restart attempt %d/%d after %.1fs",
            self._failures,
            self._max_failures,
            delay,
        )
        self._sleep(delay)
        if self._stop.is_set():
            return self.status
        self._service.spawn_server(self._paths, self._host, self._port)
        if self._service.wait_until_healthy(self.health_url, self._startup_timeout_s):
            self.status = SupervisorStatus.RUNNING
        else:
            self.status = (
                SupervisorStatus.FAILED
                if self._failures >= self._max_failures
                else SupervisorStatus.RESTARTING
            )
        return self.status

    def run(self) -> None:
        """Poll until stopped or permanently failed. Runs on a daemon thread;
        `_stop.wait()` is the pollable sleep so `stop()` wakes it immediately."""
        while not self._stop.wait(self._poll_interval_s):
            if self.check() is SupervisorStatus.FAILED:
                break

    def stop(self) -> None:
        """Signal the poll loop to exit and, if we own the server, stop it
        gracefully (`eva.service` prefers the API shutdown, then terminate).
        Idempotent; safe to call from the window-close handler."""
        self._stop.set()
        if not self._owned:
            return
        pid = self._service.read_server_pid(self._paths)
        if pid is not None:
            self._service.terminate_server(self._paths, pid, host=self._host, port=self._port)
        self._owned = False
        self.status = SupervisorStatus.STOPPED
