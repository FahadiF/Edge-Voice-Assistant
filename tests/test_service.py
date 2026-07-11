"""Server lifecycle management tests (M5.5): PID handling, stale-PID
cleanup, graceful termination, log tailing — process/network boundaries
mocked; a real spawn is exercised in manual validation."""

from __future__ import annotations

import pytest

from eva import service
from eva.config.paths import AppPaths


class TestPidFile:
    def test_no_pid_file_means_not_running(self, app_paths: AppPaths) -> None:
        assert service.read_server_pid(app_paths) is None

    def test_valid_pid_of_python_process_is_returned(
        self, app_paths: AppPaths, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        service.pid_file(app_paths).write_text("4242", encoding="utf-8")
        monkeypatch.setattr(service.psutil, "pid_exists", lambda pid: pid == 4242)

        class _Proc:
            def __init__(self, pid: int) -> None: ...

            def name(self) -> str:
                return "python.exe"

        monkeypatch.setattr(service.psutil, "Process", _Proc)
        assert service.read_server_pid(app_paths) == 4242

    def test_stale_pid_is_cleaned_up(
        self, app_paths: AppPaths, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        service.pid_file(app_paths).write_text("999999", encoding="utf-8")
        monkeypatch.setattr(service.psutil, "pid_exists", lambda pid: False)
        assert service.read_server_pid(app_paths) is None
        assert not service.pid_file(app_paths).exists()  # stale file removed

    def test_pid_reused_by_unrelated_process_is_rejected(
        self, app_paths: AppPaths, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        service.pid_file(app_paths).write_text("1234", encoding="utf-8")
        monkeypatch.setattr(service.psutil, "pid_exists", lambda pid: True)

        class _Proc:
            def __init__(self, pid: int) -> None: ...

            def name(self) -> str:
                return "notepad.exe"

        monkeypatch.setattr(service.psutil, "Process", _Proc)
        assert service.read_server_pid(app_paths) is None

    def test_garbage_pid_file_is_ignored(self, app_paths: AppPaths) -> None:
        service.pid_file(app_paths).write_text("not-a-pid", encoding="utf-8")
        assert service.read_server_pid(app_paths) is None


class TestTerminate:
    def test_already_gone_process_counts_as_stopped(
        self, app_paths: AppPaths, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        service.pid_file(app_paths).write_text("777", encoding="utf-8")

        class _Gone:
            def __init__(self, pid: int) -> None:
                raise service.psutil.NoSuchProcess(pid)

        monkeypatch.setattr(service.psutil, "Process", _Gone)
        assert service.terminate_server(app_paths, 777) is True
        assert not service.pid_file(app_paths).exists()

    def test_graceful_terminate_removes_pid_file(
        self, app_paths: AppPaths, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        service.pid_file(app_paths).write_text("777", encoding="utf-8")
        calls: list[str] = []

        class _Proc:
            def __init__(self, pid: int) -> None: ...

            def terminate(self) -> None:
                calls.append("terminate")

            def wait(self, timeout: float | None = None) -> None:
                calls.append("wait")

        monkeypatch.setattr(service.psutil, "Process", _Proc)
        assert service.terminate_server(app_paths, 777) is True
        assert calls == ["terminate", "wait"]
        assert not service.pid_file(app_paths).exists()


class TestHealthAndLogs:
    def test_health_url_rewrites_wildcard_bind(self) -> None:
        assert service.health_url("0.0.0.0", 8765).startswith("http://127.0.0.1:8765")

    def test_probe_health_false_when_nothing_listens(self) -> None:
        assert service.probe_health("http://127.0.0.1:1/api/v1/health", timeout_s=0.2) is False

    def test_newest_log_lines_tails_most_recent_file(self, app_paths: AppPaths) -> None:
        old = app_paths.logs_dir / "old.log"
        new = app_paths.logs_dir / "new.log"
        old.write_text("ancient\n", encoding="utf-8")
        new.write_text("\n".join(f"line{i}" for i in range(10)), encoding="utf-8")
        import os
        import time

        past = time.time() - 3600
        os.utime(old, (past, past))
        assert service.newest_log_lines(app_paths, 3) == ["line7", "line8", "line9"]

    def test_no_logs_returns_empty(self, app_paths: AppPaths) -> None:
        assert service.newest_log_lines(app_paths, 5) == []
