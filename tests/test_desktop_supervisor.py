"""ServerSupervisor tests (M6.1, ADR-027).

The supervisor's whole state machine — attach-vs-spawn, ownership, bounded
backoff restart, graceful stop — is exercised headless with a fake `eva.service`
and a fake clock. No real process is ever spawned.
"""

from __future__ import annotations

from pathlib import Path

from eva.config.paths import AppPaths, get_app_paths
from eva.desktop.supervisor import ServerSupervisor, SupervisorStatus


class FakeService:
    """Scriptable stand-in for the `eva.service` primitives the supervisor uses.

    `probe_results` is a queue of health answers (falls back to `healthy` once
    exhausted); `spawn_ok` decides whether a spawn brings the server up.
    """

    def __init__(self, *, spawn_ok: bool = True) -> None:
        self.healthy = False
        self.spawn_ok = spawn_ok
        self.probe_results: list[bool] = []
        self.spawns = 0
        self.terminations = 0
        self.pid = 4321

    def health_url(self, host: str, port: int) -> str:
        return f"http://{host}:{port}/api/v1/health"

    def probe_health(self, url: str, timeout_s: float = 2.0) -> bool:
        if self.probe_results:
            return self.probe_results.pop(0)
        return self.healthy

    def spawn_server(self, paths: AppPaths, host: str, port: int) -> int:
        self.spawns += 1
        self.healthy = self.spawn_ok
        return self.pid

    def wait_until_healthy(self, url: str, timeout_s: float = 90.0) -> bool:
        return self.spawn_ok

    def read_server_pid(self, paths: AppPaths) -> int | None:
        return self.pid

    def terminate_server(self, paths: AppPaths, pid: int, *, host: str = "", port: int = 0) -> bool:
        self.terminations += 1
        self.healthy = False
        return True


def _paths(tmp_path: Path) -> AppPaths:
    paths = get_app_paths(home=tmp_path)
    paths.ensure_exists()
    return paths


def _supervisor(tmp_path: Path, service: FakeService, **kw: object) -> ServerSupervisor:
    sleeps: list[float] = []
    sup = ServerSupervisor(
        _paths(tmp_path),
        "127.0.0.1",
        8765,
        service=service,  # type: ignore[arg-type]
        sleep=sleeps.append,
        max_consecutive_failures=3,
        **kw,  # type: ignore[arg-type]
    )
    sup._sleeps = sleeps  # type: ignore[attr-defined]  # for backoff assertions
    return sup


class TestAttachVsSpawn:
    def test_attaches_to_already_healthy_server(self, tmp_path: Path) -> None:
        service = FakeService()
        service.probe_results = [True]  # a server is already up
        sup = _supervisor(tmp_path, service)
        assert sup.ensure_running() is True
        assert sup.status is SupervisorStatus.ATTACHED
        assert sup.owns_server is False
        assert service.spawns == 0

    def test_spawns_and_owns_when_none_running(self, tmp_path: Path) -> None:
        service = FakeService(spawn_ok=True)
        service.probe_results = [False]  # nothing running → spawn
        sup = _supervisor(tmp_path, service)
        assert sup.ensure_running() is True
        assert sup.status is SupervisorStatus.RUNNING
        assert sup.owns_server is True
        assert service.spawns == 1

    def test_ensure_running_fails_when_spawn_never_healthy(self, tmp_path: Path) -> None:
        service = FakeService(spawn_ok=False)
        service.probe_results = [False]
        sup = _supervisor(tmp_path, service)
        assert sup.ensure_running() is False
        assert sup.status is SupervisorStatus.FAILED


class TestSupervisionLoop:
    def test_owned_server_restarts_with_backoff(self, tmp_path: Path) -> None:
        service = FakeService(spawn_ok=True)
        service.probe_results = [False]  # initial spawn
        sup = _supervisor(tmp_path, service)
        sup.ensure_running()
        assert service.spawns == 1
        # It died: next check probes unhealthy, then restarts.
        service.probe_results = [False]
        service.healthy = False
        assert sup.check() is SupervisorStatus.RUNNING
        assert service.spawns == 2  # one restart
        assert sup._sleeps == [1.0]  # type: ignore[attr-defined]  # first backoff step

    def test_healthy_check_resets_failure_count(self, tmp_path: Path) -> None:
        service = FakeService(spawn_ok=True)
        service.probe_results = [False]
        sup = _supervisor(tmp_path, service)
        sup.ensure_running()
        service.probe_results = [False]  # one failure → restart succeeds
        assert sup.check() is SupervisorStatus.RUNNING
        assert sup._failures == 1  # counted, not yet cleared
        service.probe_results = [True]  # server is now stably healthy
        assert sup.check() is SupervisorStatus.RUNNING
        assert sup._failures == 0  # a server that stays up recovers fully

    def test_crash_loop_gives_up_after_cap(self, tmp_path: Path) -> None:
        service = FakeService(spawn_ok=False)  # every (re)start fails
        service.probe_results = [False]
        sup = _supervisor(tmp_path, service)
        sup.ensure_running()  # FAILED (spawn never healthy), owned=True
        spawns_after_start = service.spawns
        statuses = [sup.check() for _ in range(6)]
        assert SupervisorStatus.FAILED in statuses
        # Restart attempts are bounded by the cap (3), not infinite.
        assert service.spawns - spawns_after_start <= 3
        # Backoff grows then is capped — never a tight loop.
        assert sup._sleeps == sorted(sup._sleeps)  # type: ignore[attr-defined]

    def test_attached_server_is_not_restarted(self, tmp_path: Path) -> None:
        service = FakeService()
        service.probe_results = [True]  # attach
        sup = _supervisor(tmp_path, service)
        sup.ensure_running()
        service.probe_results = [False]  # external server vanished
        assert sup.check() is SupervisorStatus.STOPPED
        assert service.spawns == 0  # we never restart a server we didn't start


class TestStatusChangeCallback:
    """M6.2: the tray subscribes to on_status_change so it reflects server
    state without polling — the callback must fire only on transitions."""

    def test_callback_fires_on_transitions_only(self, tmp_path: Path) -> None:
        service = FakeService(spawn_ok=True)
        service.probe_results = [False]  # spawn → RUNNING
        sup = _supervisor(tmp_path, service)
        seen: list[SupervisorStatus] = []
        sup.on_status_change = seen.append
        sup.ensure_running()  # STOPPED → RUNNING (one transition)
        assert seen == [SupervisorStatus.RUNNING]
        # A healthy re-check does not re-fire (no transition).
        service.probe_results = [True]
        sup.check()
        assert seen == [SupervisorStatus.RUNNING]

    def test_callback_reports_failure_transition(self, tmp_path: Path) -> None:
        service = FakeService(spawn_ok=False)
        service.probe_results = [False]
        sup = _supervisor(tmp_path, service)
        seen: list[SupervisorStatus] = []
        sup.on_status_change = seen.append
        sup.ensure_running()  # → FAILED
        assert seen[-1] is SupervisorStatus.FAILED


class TestStop:
    def test_stop_terminates_owned_server(self, tmp_path: Path) -> None:
        service = FakeService()
        service.probe_results = [False]
        sup = _supervisor(tmp_path, service)
        sup.ensure_running()  # owned
        sup.stop()
        assert service.terminations == 1
        assert sup.owns_server is False

    def test_stop_leaves_attached_server_running(self, tmp_path: Path) -> None:
        service = FakeService()
        service.probe_results = [True]
        sup = _supervisor(tmp_path, service)
        sup.ensure_running()  # attached
        sup.stop()
        assert service.terminations == 0

    def test_stop_halts_the_poll_loop(self, tmp_path: Path) -> None:
        service = FakeService()
        service.probe_results = [True]
        sup = _supervisor(tmp_path, service, poll_interval_s=0.01)
        sup.ensure_running()
        import threading

        thread = threading.Thread(target=sup.run)
        thread.start()
        sup.stop()
        thread.join(timeout=5)
        assert not thread.is_alive()  # stop() woke the loop promptly
