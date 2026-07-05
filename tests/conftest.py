"""Shared fixtures. Tests never touch real user directories: EVA_HOME is
redirected into a temp dir for every test."""

from __future__ import annotations

from pathlib import Path

import pytest

from eva.config.paths import AppPaths, get_app_paths


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "eva-home"
    monkeypatch.setenv("EVA_HOME", str(home))
    # Point EVA_UI_DIST at a path with no index.html so tests never mount a
    # real `web/dist` build that happens to exist in the checkout (ADR-023:
    # an explicit override is authoritative — no fall-through). Tests that
    # exercise UI hosting set their own value.
    monkeypatch.setenv("EVA_UI_DIST", str(tmp_path / "no-ui"))
    return home


@pytest.fixture
def app_paths(isolated_home: Path) -> AppPaths:
    paths = get_app_paths()
    paths.ensure_exists()
    return paths
