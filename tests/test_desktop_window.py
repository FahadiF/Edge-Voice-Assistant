"""WindowController tests (M6.2, ADR-027).

The close-to-tray / minimize-to-tray / start-minimized / tray-quit decisions
are exercised against a fake window that records calls — the exact logic that
was missing when those settings "did nothing" on Windows.
"""

from __future__ import annotations

from eva.config.settings import DesktopSettings
from eva.desktop.window import WindowController, should_start_hidden


class FakeWindow:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.scripts: list[str] = []

    def show(self) -> None:
        self.calls.append("show")

    def hide(self) -> None:
        self.calls.append("hide")

    def restore(self) -> None:
        self.calls.append("restore")

    def destroy(self) -> None:
        self.calls.append("destroy")

    def evaluate_js(self, script: str) -> None:
        self.scripts.append(script)


def _controller(
    window: FakeWindow, settings: DesktopSettings, *, tray_available: bool = True
) -> tuple[WindowController, list[int]]:
    saves: list[int] = []
    controller = WindowController(
        window,
        settings,
        on_save=lambda: saves.append(1),
        tray_available=tray_available,
    )
    return controller, saves


class TestCloseToTray:
    def test_close_to_tray_hides_and_vetoes_close(self) -> None:
        window = FakeWindow()
        controller, saves = _controller(window, DesktopSettings(close_to_tray=True))
        allow = controller.on_closing()
        assert allow is False  # close is vetoed
        assert window.calls == ["hide"]  # hidden to tray instead
        assert saves == []  # not quitting → no state save

    def test_close_actually_closes_when_setting_off(self) -> None:
        window = FakeWindow()
        controller, saves = _controller(window, DesktopSettings(close_to_tray=False))
        allow = controller.on_closing()
        assert allow is True  # close proceeds
        assert window.calls == []  # not hidden
        assert saves == [1]  # state persisted before closing

    def test_close_to_tray_ignored_without_a_tray(self) -> None:
        # No tray to hide into → the X button must still close the app.
        window = FakeWindow()
        controller, saves = _controller(
            window, DesktopSettings(close_to_tray=True), tray_available=False
        )
        assert controller.on_closing() is True
        assert window.calls == []
        assert saves == [1]


class TestMinimizeToTray:
    def test_minimize_to_tray_hides(self) -> None:
        window = FakeWindow()
        controller, _ = _controller(window, DesktopSettings(minimize_to_tray=True))
        controller.on_minimized()
        assert window.calls == ["hide"]

    def test_minimize_does_nothing_when_setting_off(self) -> None:
        window = FakeWindow()
        controller, _ = _controller(window, DesktopSettings(minimize_to_tray=False))
        controller.on_minimized()
        assert window.calls == []

    def test_minimize_to_tray_ignored_without_a_tray(self) -> None:
        window = FakeWindow()
        controller, _ = _controller(
            window, DesktopSettings(minimize_to_tray=True), tray_available=False
        )
        controller.on_minimized()
        assert window.calls == []


class TestTrayQuitVsCloseToTray:
    def test_request_quit_saves_and_destroys(self) -> None:
        window = FakeWindow()
        controller, saves = _controller(window, DesktopSettings(close_to_tray=True))
        controller.request_quit()
        assert saves == [1]
        assert window.calls == ["destroy"]

    def test_quit_closes_even_with_close_to_tray(self) -> None:
        # Tray Quit must exit even though close-to-tray is on: the subsequent
        # `closing` event (if the backend fires one on destroy) must ALLOW it.
        window = FakeWindow()
        controller, _ = _controller(window, DesktopSettings(close_to_tray=True))
        controller.request_quit()
        window.calls.clear()
        assert controller.on_closing() is True  # not vetoed after a quit
        assert "hide" not in window.calls


class TestShowActions:
    def test_show_uses_show_restore_show_order(self) -> None:
        # Order is load-bearing: a window hidden while minimized needs show()
        # BEFORE restore() (Form.Show re-applies the last shown state and would
        # re-minimize otherwise), then a trailing show() to focus. Measured
        # against pywebview 6.2.1 winforms; asserted here to lock the sequence.
        window = FakeWindow()
        controller, _ = _controller(window, DesktopSettings())
        controller.show()
        assert window.calls == ["show", "restore", "show"]

    def test_show_settings_navigates(self) -> None:
        window = FakeWindow()
        controller, _ = _controller(window, DesktopSettings())
        controller.show_settings()
        assert window.calls == ["show", "restore", "show"]
        assert window.scripts == ["window.location.hash = '#/settings'"]

    def test_hide(self) -> None:
        window = FakeWindow()
        controller, _ = _controller(window, DesktopSettings())
        controller.hide()
        assert window.calls == ["hide"]


class TestShouldStartHidden:
    def test_hidden_only_with_setting_and_tray(self) -> None:
        assert should_start_hidden(DesktopSettings(start_minimized=True), tray_available=True)
        assert not should_start_hidden(DesktopSettings(start_minimized=True), tray_available=False)
        assert not should_start_hidden(DesktopSettings(start_minimized=False), tray_available=True)
