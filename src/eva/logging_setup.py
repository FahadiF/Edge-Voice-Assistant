"""Logging configuration.

Two sinks:
- console: concise human-readable lines (what a developer watches),
- rotating file in the logs dir: full detail, optionally JSON for machine
  consumption (log viewer in the Developer UI reads this file).

Built on stdlib logging so third-party library logs flow through the same pipe.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
from datetime import UTC, datetime
from pathlib import Path

_FILE_MAX_BYTES = 5 * 1024 * 1024
_FILE_BACKUPS = 3
_CONSOLE_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
_CONSOLE_DATEFMT = "%H:%M:%S"


class JsonFormatter(logging.Formatter):
    """One JSON object per line; safe for any extra attributes."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(
    level: str = "INFO",
    logs_dir: Path | None = None,
    json_file: bool = False,
) -> None:
    """Configure the root logger. Idempotent: clears previously added handlers."""
    root = logging.getLogger()
    root.setLevel(level.upper())
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(_CONSOLE_FORMAT, datefmt=_CONSOLE_DATEFMT))
    root.addHandler(console)

    if logs_dir is not None:
        logs_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            logs_dir / "eva.log",
            maxBytes=_FILE_MAX_BYTES,
            backupCount=_FILE_BACKUPS,
            encoding="utf-8",
        )
        file_handler.setFormatter(
            JsonFormatter() if json_file else logging.Formatter(_CONSOLE_FORMAT)
        )
        root.addHandler(file_handler)
