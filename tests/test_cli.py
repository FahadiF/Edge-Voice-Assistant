from __future__ import annotations

import pytest

from eva.cli import main


def test_version_command(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["version"]) == 0
    out = capsys.readouterr().out
    assert "Edge Voice Assistant" in out


def test_diagnose_command(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["diagnose"]) == 0
    out = capsys.readouterr().out
    for heading in ("System", "Hardware", "Recommended profile", "Configuration", "Paths"):
        assert heading in out


def test_no_command_errors() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2
