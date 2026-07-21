"""Desktop shell tests (M6.1, ADR-027).

`pywebview` is an optional extra not installed in the test environment, so
`main()`'s window/event-loop calls are exercised against a fake `webview`
module injected into `sys.modules`. The reusable logic (supervisor, state) is
tested in `test_desktop_supervisor.py` / `test_desktop_state.py`; here we cover
the entry point, the geometry→state mapping, and `main()`'s wiring.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Iterator

import pytest

from eva.desktop import main
from eva.desktop.platform import DesktopPlatform, TrayIconState, TraySpec
from eva.desktop.shell import (
    _initial_url,
    _keep_webview_awake_while_hidden,
    _route_of,
    capture_window_state,
)
from eva.desktop.state import MIN_WIDTH, DesktopState


def test_entry_point_main_is_the_shell_main() -> None:
    # The console script `eva-desktop = "eva.desktop:main"` must keep resolving
    # after the module → package promotion.
    assert main.__module__ == "eva.desktop.shell"


class TestRouteAndUrl:
    def test_route_of_extracts_hash(self) -> None:
        assert _route_of("http://127.0.0.1:8765/#/memory") == "#/memory"
        assert _route_of("http://127.0.0.1:8765/") == ""

    def test_initial_url_appends_saved_route(self) -> None:
        state = DesktopState(last_route="#/settings")
        assert _initial_url("127.0.0.1", 8765, state) == "http://127.0.0.1:8765/#/settings"

    def test_initial_url_wildcard_host_becomes_localhost(self) -> None:
        assert _initial_url("0.0.0.0", 8765, DesktopState()).startswith("http://127.0.0.1:8765/")


class TestCaptureWindowState:
    def test_reads_geometry_and_route(self) -> None:
        window = types.SimpleNamespace(
            width=1024,
            height=768,
            x=30,
            y=40,
            get_current_url=lambda: "http://127.0.0.1:8765/#/models",
        )
        state = capture_window_state(window, DesktopState())
        assert (state.width, state.height, state.x, state.y) == (1024, 768, 30, 40)
        assert state.last_route == "#/models"

    def test_falls_back_to_previous_on_bad_window(self) -> None:
        previous = DesktopState(width=900, height=700, last_route="#/memory")

        class _BadWindow:
            width = "nope"  # not an int
            height = 700
            x = None
            y = None

            def get_current_url(self) -> str:
                raise RuntimeError("window went away")

        state = capture_window_state(_BadWindow(), previous)
        assert state.width >= MIN_WIDTH
        assert state.last_route == "#/memory"  # kept the previous route


class TestKeepWebviewAwake:
    """The renderer must not be backgrounded while hidden to the tray, or the
    live WebSocket/UI would stall while minimized (measured Chromium throttling).
    Windows-only; user overrides are preserved."""

    ENV = "WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS"

    def test_sets_anti_backgrounding_flags_on_windows(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("eva.desktop.shell.sys.platform", "win32")
        monkeypatch.delenv(self.ENV, raising=False)
        _keep_webview_awake_while_hidden()
        import os

        assert "--disable-renderer-backgrounding" in os.environ[self.ENV]
        assert "--disable-background-timer-throttling" in os.environ[self.ENV]

    def test_noop_off_windows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("eva.desktop.shell.sys.platform", "linux")
        monkeypatch.delenv(self.ENV, raising=False)
        _keep_webview_awake_while_hidden()
        import os

        assert self.ENV not in os.environ

    def test_preserves_existing_user_args(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("eva.desktop.shell.sys.platform", "win32")
        monkeypatch.setenv(self.ENV, "--my-flag")
        _keep_webview_awake_while_hidden()
        import os

        value = os.environ[self.ENV]
        assert "--my-flag" in value
        assert "--disable-renderer-backgrounding" in value

    def test_idempotent_when_already_configured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("eva.desktop.shell.sys.platform", "win32")
        monkeypatch.setenv(self.ENV, "--disable-renderer-backgrounding")
        _keep_webview_awake_while_hidden()
        import os

        # Left as-is (our marker already present) — not doubled up.
        assert os.environ[self.ENV].count("--disable-renderer-backgrounding") == 1


@pytest.fixture
def fake_webview() -> Iterator[types.SimpleNamespace]:
    calls: dict[str, object] = {}

    class _Signal:
        def __iadd__(self, handler: object) -> _Signal:
            return self

    class _Events:
        def __init__(self) -> None:
            self.closing = _Signal()
            self.minimized = _Signal()

    class _Window:
        def __init__(self) -> None:
            self.events = _Events()
            self.width, self.height, self.x, self.y = 1200, 800, None, None

        def get_current_url(self) -> str:
            return "http://127.0.0.1:8765/"

        # The tray/window controller wires callbacks to these; no-ops here.
        def show(self) -> None: ...
        def hide(self) -> None: ...
        def restore(self) -> None: ...
        def destroy(self) -> None: ...
        def evaluate_js(self, _script: str) -> None: ...

    def create_window(title: str, url: str, **kwargs: object) -> _Window:
        calls["title"] = title
        calls["url"] = url
        calls["kwargs"] = kwargs
        return _Window()

    def start() -> None:
        calls["started"] = True

    fake = types.SimpleNamespace(create_window=create_window, start=start, calls=calls)
    sys.modules["webview"] = fake  # type: ignore[assignment]
    yield fake
    del sys.modules["webview"]


class _FakeSupervisor:
    def __init__(self, *_a: object, healthy: bool = True, **_k: object) -> None:
        self._healthy = healthy
        self.stopped = False

    def ensure_running(self) -> bool:
        return self._healthy

    def run(self) -> None:
        return None

    def stop(self) -> None:
        self.stopped = True


def test_main_opens_window_when_server_starts(
    fake_webview: types.SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("eva.desktop.shell.ServerSupervisor", _FakeSupervisor)
    # Isolate from whether the tray libs happen to be installed in this env —
    # the tray has its own tests; here we only assert the window opens.
    monkeypatch.setattr("eva.desktop.shell.create_platform", lambda: None)
    assert main() == 0
    assert fake_webview.calls["title"] == "Edge Voice Assistant"
    assert str(fake_webview.calls["url"]).startswith("http://127.0.0.1:")
    assert fake_webview.calls.get("started") is True


class _RecordingPlatform(DesktopPlatform):
    """Minimal fake platform for the shell's tray-wiring tests."""

    def __init__(self, *, fail_start: bool = False) -> None:
        self.fail_start = fail_start
        self.started = False
        self.stopped = False

    def start_tray(self, spec: TraySpec) -> None:
        if self.fail_start:
            raise RuntimeError("no tray backend on this box")
        self.started = True

    def set_tray_state(self, icon: TrayIconState, status_text: str) -> None: ...

    def stop_tray(self) -> None:
        self.stopped = True


def test_main_wires_tray_when_platform_available(
    fake_webview: types.SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    supervisor = _FakeSupervisor()
    supervisor.on_status_change = None  # type: ignore[attr-defined]
    monkeypatch.setattr("eva.desktop.shell.ServerSupervisor", lambda *a, **k: supervisor)
    platform = _RecordingPlatform()
    monkeypatch.setattr("eva.desktop.shell.create_platform", lambda: platform)
    assert main() == 0
    assert platform.started is True  # tray was started
    assert platform.stopped is True  # and stopped on teardown
    assert supervisor.on_status_change is not None  # tray subscribed to status


def test_main_survives_tray_failure(
    fake_webview: types.SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A tray that fails to start must NOT stop the window opening (graceful
    # degradation) — this softens exactly the class of the M6.2 pystray bug.
    monkeypatch.setattr("eva.desktop.shell.ServerSupervisor", _FakeSupervisor)
    monkeypatch.setattr(
        "eva.desktop.shell.create_platform", lambda: _RecordingPlatform(fail_start=True)
    )
    assert main() == 0
    assert fake_webview.calls.get("started") is True  # window still opened


def test_main_returns_error_when_server_never_starts(
    fake_webview: types.SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "eva.desktop.shell.ServerSupervisor",
        lambda *a, **k: _FakeSupervisor(healthy=False),
    )
    assert main() == 1
    assert "started" not in fake_webview.calls  # window never opened


def test_main_reports_missing_desktop_extra_without_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # pywebview absent → a friendly remedy, never a bare ModuleNotFoundError.
    monkeypatch.setitem(sys.modules, "webview", None)  # forces ImportError on `import webview`
    assert main() == 1
    out = capsys.readouterr().out
    assert 'pip install -e ".[desktop]"' in out
    assert "Traceback" not in out
