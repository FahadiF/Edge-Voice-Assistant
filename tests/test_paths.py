from __future__ import annotations

from pathlib import Path

import pytest

from eva.config.paths import get_app_paths


def test_env_home_overrides_platform_dirs(isolated_home: Path) -> None:
    paths = get_app_paths()
    assert paths.config_dir == isolated_home / "config"
    assert paths.models_dir == isolated_home / "models"
    assert paths.settings_file == isolated_home / "config" / "settings.json"


def test_explicit_home_beats_env(tmp_path: Path) -> None:
    explicit = tmp_path / "elsewhere"
    paths = get_app_paths(home=explicit)
    assert paths.data_dir == explicit / "data"


def test_ensure_exists_creates_all_dirs(isolated_home: Path) -> None:
    paths = get_app_paths()
    paths.ensure_exists()
    for directory in (
        paths.config_dir,
        paths.data_dir,
        paths.models_dir,
        paths.conversations_dir,
        paths.logs_dir,
    ):
        assert directory.is_dir()


def test_paths_are_immutable(isolated_home: Path) -> None:
    paths = get_app_paths()
    with pytest.raises(AttributeError):
        paths.config_dir = Path("/nope")  # type: ignore[misc]
