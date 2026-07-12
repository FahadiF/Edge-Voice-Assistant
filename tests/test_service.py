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


class TestGracefulShutdown:
    """M5.6: `eva stop` asks the server to stop itself over the API before
    falling back to terminate — on Windows, terminate is TerminateProcess
    (no cleanup at all), so the API call is the only graceful path."""

    def test_request_graceful_shutdown_posts_to_endpoint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import urllib.request

        seen: dict[str, str] = {}

        class _Response:
            status = 200

            def __enter__(self) -> _Response:
                return self

            def __exit__(self, *exc: object) -> None:
                return None

        def fake_urlopen(request: urllib.request.Request, timeout: float = 0) -> _Response:
            seen["url"] = request.full_url
            seen["method"] = request.get_method()
            return _Response()

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        assert service.request_graceful_shutdown("127.0.0.1", 8765)
        assert seen["url"] == "http://127.0.0.1:8765/api/v1/system/shutdown"
        assert seen["method"] == "POST"

    def test_request_graceful_shutdown_false_when_unreachable(self) -> None:
        # Port 9 (discard) is never an EVA server.
        assert not service.request_graceful_shutdown("127.0.0.1", 9, timeout_s=0.2)

    def test_terminate_prefers_graceful_path(
        self, app_paths: AppPaths, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import subprocess
        import sys

        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
        )
        try:
            service.pid_file(app_paths).write_text(str(process.pid), encoding="utf-8")
            calls: list[str] = []

            def fake_graceful(host: str, port: int, timeout_s: float = 5.0) -> bool:
                calls.append("graceful")
                process.terminate()  # simulate the server exiting on request
                return True

            monkeypatch.setattr(service, "request_graceful_shutdown", fake_graceful)
            assert service.terminate_server(app_paths, process.pid, host="127.0.0.1", port=8765)
            assert calls == ["graceful"]
            assert not service.pid_file(app_paths).exists()
        finally:
            if process.poll() is None:
                process.kill()

    def test_terminate_falls_back_when_graceful_fails(
        self, app_paths: AppPaths, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import subprocess
        import sys

        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
        )
        try:
            service.pid_file(app_paths).write_text(str(process.pid), encoding="utf-8")
            monkeypatch.setattr(
                service, "request_graceful_shutdown", lambda host, port, timeout_s=5.0: False
            )
            assert service.terminate_server(app_paths, process.pid, host="127.0.0.1", port=8765)
            assert process.poll() is not None  # hard-terminated
        finally:
            if process.poll() is None:
                process.kill()
