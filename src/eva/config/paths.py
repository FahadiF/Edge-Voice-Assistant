"""Platform-appropriate application directories.

All user data lives under OS-conventional locations (``%APPDATA%`` /
``~/.config`` / ``~/.local/share``), never inside the installation directory,
so the app can be updated or reinstalled without losing models, settings, or
conversations. ``EVA_HOME`` overrides everything for portable installs and tests.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path

from platformdirs import user_config_dir, user_data_dir, user_log_dir

from eva import APP_NAME

_ENV_HOME = "EVA_HOME"


@dataclass(frozen=True)
class AppPaths:
    """Resolved application directories. Instances are immutable."""

    config_dir: Path
    data_dir: Path
    models_dir: Path
    conversations_dir: Path
    logs_dir: Path

    @property
    def settings_file(self) -> Path:
        return self.config_dir / "settings.json"

    def ensure_exists(self) -> None:
        """Create all directories that do not yet exist."""
        for f in fields(self):
            value: Path = getattr(self, f.name)
            value.mkdir(parents=True, exist_ok=True)


def get_app_paths(home: Path | None = None) -> AppPaths:
    """Resolve application paths.

    Precedence: explicit ``home`` argument → ``EVA_HOME`` environment variable →
    per-OS user directories via platformdirs.
    """
    env_home = os.environ.get(_ENV_HOME)
    if home is None and env_home:
        home = Path(env_home)

    if home is not None:
        return AppPaths(
            config_dir=home / "config",
            data_dir=home / "data",
            models_dir=home / "models",
            conversations_dir=home / "conversations",
            logs_dir=home / "logs",
        )

    data_dir = Path(user_data_dir(APP_NAME, appauthor=False))
    return AppPaths(
        config_dir=Path(user_config_dir(APP_NAME, appauthor=False)),
        data_dir=data_dir,
        models_dir=data_dir / "models",
        conversations_dir=data_dir / "conversations",
        logs_dir=Path(user_log_dir(APP_NAME, appauthor=False)),
    )
