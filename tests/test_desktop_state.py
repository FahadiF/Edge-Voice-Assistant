"""DesktopState persistence tests (M6.1, ADR-027)."""

from __future__ import annotations

from pathlib import Path

from eva.config.paths import AppPaths, get_app_paths
from eva.desktop.state import (
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    MIN_HEIGHT,
    MIN_WIDTH,
    DesktopState,
)


def _paths(tmp_path: Path) -> AppPaths:
    paths = get_app_paths(home=tmp_path)
    paths.ensure_exists()
    return paths


def test_defaults_when_no_file(tmp_path: Path) -> None:
    state = DesktopState.load(_paths(tmp_path))
    assert (state.width, state.height) == (DEFAULT_WIDTH, DEFAULT_HEIGHT)
    assert state.x is None and state.y is None
    assert state.maximized is False
    assert state.last_route == ""


def test_round_trip(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    DesktopState(width=1024, height=768, x=100, y=50, maximized=True, last_route="#/memory").save(
        paths
    )
    restored = DesktopState.load(paths)
    assert restored.width == 1024
    assert restored.height == 768
    assert restored.x == 100
    assert restored.y == 50
    assert restored.maximized is True
    assert restored.last_route == "#/memory"


def test_corrupt_file_falls_back_to_defaults(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    (paths.config_dir / "desktop_state.json").write_text("{not json", encoding="utf-8")
    state = DesktopState.load(paths)
    assert state.width == DEFAULT_WIDTH  # no crash, sane defaults


def test_tiny_or_bad_dimensions_are_clamped(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    (paths.config_dir / "desktop_state.json").write_text(
        '{"width": 10, "height": "oops", "x": "nope"}', encoding="utf-8"
    )
    state = DesktopState.load(paths)
    assert state.width == MIN_WIDTH  # clamped up from an unusable 10px
    assert state.height == DEFAULT_HEIGHT  # non-int → default
    assert state.x is None  # non-int coordinate → None (let OS place it)
    assert state.height >= MIN_HEIGHT
