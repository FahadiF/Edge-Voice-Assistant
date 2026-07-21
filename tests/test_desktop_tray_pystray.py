"""pystray adapter integration tests (M6.2, ADR-027).

The headless unit tests use a `FakeDesktopPlatform`, so they never touch the
real `pystray` API — which is exactly why the M6.2 arg-count bug
(`pystray` rejects a menu-action callable with more than two parameters, and a
default parameter counts) slipped through. These tests build the REAL pystray
Icon + menu via `PystrayDesktopPlatform.build_icon` (the display-independent
construction path that raised `ValueError`) and would have caught it.

They require the optional desktop extra and are skipped when it is absent, so
base CI stays green while any environment with `pip install -e ".[desktop]"`
(and a future desktop-extra CI leg) runs them. No tray window is opened —
`build_icon` does not call `Icon.run()`.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pystray")
pytest.importorskip("PIL")

from eva.desktop.platform import (
    PystrayDesktopPlatform,
    TrayIconState,
    TraySpec,
)
from eva.desktop.supervisor import SupervisorStatus
from eva.desktop.tray import TrayController


def _spec_and_calls() -> tuple[TraySpec, dict[str, int]]:
    calls: dict[str, int] = {}

    def bump(name: str) -> None:
        calls[name] = calls.get(name, 0) + 1

    controller = TrayController(
        PystrayDesktopPlatform(),  # unused here; we only want its menu
        on_open=lambda: bump("open"),
        on_hide=lambda: bump("hide"),
        on_settings=lambda: bump("settings"),
        on_quit=lambda: bump("quit"),
        initial_status=SupervisorStatus.RUNNING,
    )
    return TraySpec(title="Edge Voice Assistant", menu=controller._menu()), calls


def test_build_icon_constructs_without_valueerror() -> None:
    # This is the exact path that raised `ValueError(action)` for a 3-arg
    # menu callback. Construction succeeding IS the regression guard.
    spec, _ = _spec_and_calls()
    icon = PystrayDesktopPlatform().build_icon(spec)
    assert icon is not None


def test_menu_texts_render() -> None:
    spec, _ = _spec_and_calls()
    icon = PystrayDesktopPlatform().build_icon(spec)
    texts = [item.text for item in icon.menu]  # triggers callable text(item)
    assert "Open" in texts
    assert "Hide" in texts
    assert "Settings" in texts
    assert "Quit" in texts
    assert any(t.startswith("Engine:") for t in texts)


def test_menu_actions_invoke_the_shell_callbacks() -> None:
    spec, calls = _spec_and_calls()
    icon = PystrayDesktopPlatform().build_icon(spec)
    for item in icon.menu:
        item(icon)  # pystray MenuItem.__call__(icon) -> action(icon, item)
    # Every clickable item dispatched exactly once; the disabled status label
    # (a no-op wrapped action) fired nothing extra.
    assert calls == {"open": 1, "hide": 1, "settings": 1, "quit": 1}


def test_status_icon_images_render_for_every_state() -> None:
    platform = PystrayDesktopPlatform()
    for state in TrayIconState:
        image = platform._image(state)
        assert image.size == (64, 64)
