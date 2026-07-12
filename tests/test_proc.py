"""Console-window suppression helper (M5.7)."""

from __future__ import annotations

import subprocess
import sys

from eva.core.proc import no_window_kwargs


def test_no_window_kwargs_matches_platform() -> None:
    kwargs = no_window_kwargs()
    if sys.platform == "win32":
        assert kwargs == {"creationflags": subprocess.CREATE_NO_WINDOW}
    else:
        assert kwargs == {}


def test_run_probe_uses_no_window(monkeypatch: object) -> None:
    """Every hardware probe must pass the no-window kwargs so a detached
    server never flashes a console (nvidia-smi runs on every snapshot)."""
    import eva.hardware.detect as detect

    captured: dict[str, object] = {}

    def fake_which(_cmd: str) -> str:
        return "nvidia-smi"

    class _Result:
        returncode = 0
        stdout = "ok"

    def fake_run(cmd: list[str], **kwargs: object) -> _Result:
        captured.update(kwargs)
        return _Result()

    monkeypatch.setattr(detect.shutil, "which", fake_which)  # type: ignore[attr-defined]
    monkeypatch.setattr(detect.subprocess, "run", fake_run)  # type: ignore[attr-defined]
    detect.run_probe(["nvidia-smi", "--x"])
    if sys.platform == "win32":
        assert captured.get("creationflags") == subprocess.CREATE_NO_WINDOW
    else:
        assert "creationflags" not in captured
