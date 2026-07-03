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
    return home


@pytest.fixture
def app_paths(isolated_home: Path) -> AppPaths:
    paths = get_app_paths()
    paths.ensure_exists()
    return paths
