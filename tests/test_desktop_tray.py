"""TrayController tests (M6.2, ADR-027).

The tray's logic — supervisor-state → icon/text, the menu shape, and click
dispatch — is exercised against a `FakeDesktopPlatform` that records what the
real pystray adapter would render. No native tray is created.
"""

from __future__ import annotations

from eva.desktop.platform import DesktopPlatform, TrayIconState, TraySpec
from eva.desktop.supervisor import SupervisorStatus
from eva.desktop.tray import TrayController, icon_for, text_for


class FakeDesktopPlatform(DesktopPlatform):
    def __init__(self) -> None:
        self.spec: TraySpec | None = None
        self.states: list[tuple[TrayIconState, str]] = []
        self.stopped = False

    def start_tray(self, spec: TraySpec) -> None:
        self.spec = spec

    def set_tray_state(self, icon: TrayIconState, status_text: str) -> None:
        self.states.append((icon, status_text))

    def stop_tray(self) -> None:
        self.stopped = True


def _controller(platform: DesktopPlatform, calls: dict[str, int] | None = None) -> TrayController:
    calls = calls if calls is not None else {}

    def bump(name: str) -> None:
        calls[name] = calls.get(name, 0) + 1

    return TrayController(
        platform,
        on_open=lambda: bump("open"),
        on_hide=lambda: bump("hide"),
        on_settings=lambda: bump("settings"),
        on_quit=lambda: bump("quit"),
    )


class TestStatusMapping:
    def test_icon_and_text_for_each_status(self) -> None:
        assert icon_for(SupervisorStatus.RUNNING) is TrayIconState.RUNNING
        assert icon_for(SupervisorStatus.ATTACHED) is TrayIconState.RUNNING
        assert icon_for(SupervisorStatus.RESTARTING) is TrayIconState.STARTING
        assert icon_for(SupervisorStatus.STOPPED) is TrayIconState.STOPPED
        assert icon_for(SupervisorStatus.FAILED) is TrayIconState.ERROR
        assert text_for(SupervisorStatus.RUNNING) == "Running"
        assert text_for(SupervisorStatus.FAILED) == "Error"

    def test_start_applies_initial_status(self) -> None:
        platform = FakeDesktopPlatform()
        controller = TrayController(
            platform,
            on_open=lambda: None,
            on_hide=lambda: None,
            on_settings=lambda: None,
            on_quit=lambda: None,
            initial_status=SupervisorStatus.RUNNING,
        )
        controller.start()
        assert platform.states[-1] == (TrayIconState.RUNNING, "Running")

    def test_status_change_updates_the_icon(self) -> None:
        platform = FakeDesktopPlatform()
        controller = _controller(platform)
        controller.start()
        controller.on_supervisor_status(SupervisorStatus.RESTARTING)
        assert platform.states[-1] == (TrayIconState.STARTING, "Starting…")
        controller.on_supervisor_status(SupervisorStatus.FAILED)
        assert platform.states[-1] == (TrayIconState.ERROR, "Error")


class TestMenu:
    def test_menu_has_the_expected_actions(self) -> None:
        platform = FakeDesktopPlatform()
        _controller(platform).start()
        assert platform.spec is not None
        labels = [m.label for m in platform.spec.menu]
        # Restore Window / Hide are static; the status line is a callable; Settings/Quit static.
        assert "Restore Window" in labels
        assert "Hide" in labels
        assert "Settings" in labels
        assert "Quit" in labels

    def test_restore_window_is_the_default_activation_item(self) -> None:
        # Left-clicking the tray icon must restore the window; pystray only fires
        # a `default` item on activation, so exactly the Restore item carries it.
        platform = FakeDesktopPlatform()
        _controller(platform).start()
        assert platform.spec is not None
        defaults = [m for m in platform.spec.menu if m.default]
        assert len(defaults) == 1
        assert defaults[0].label == "Restore Window"
        assert defaults[0].on_activate is not None

    def test_status_line_is_a_live_callable(self) -> None:
        platform = FakeDesktopPlatform()
        controller = _controller(platform)
        controller.start()
        assert platform.spec is not None
        status_item = next(m for m in platform.spec.menu if callable(m.label))
        assert status_item.on_activate is None  # non-clickable label
        assert status_item.label() == "Engine: Stopped"  # type: ignore[operator]
        controller.on_supervisor_status(SupervisorStatus.RUNNING)
        assert status_item.label() == "Engine: Running"  # type: ignore[operator]  # re-renders live

    def test_menu_clicks_dispatch_to_callbacks(self) -> None:
        platform = FakeDesktopPlatform()
        calls: dict[str, int] = {}
        _controller(platform, calls).start()
        assert platform.spec is not None
        by_label = {m.label: m for m in platform.spec.menu if isinstance(m.label, str)}
        for label in ("Restore Window", "Hide", "Settings", "Quit"):
            action = by_label[label].on_activate
            assert action is not None
            action()
        assert calls == {"open": 1, "hide": 1, "settings": 1, "quit": 1}


def test_stop_stops_the_platform_tray() -> None:
    platform = FakeDesktopPlatform()
    controller = _controller(platform)
    controller.start()
    controller.stop()
    assert platform.stopped is True
