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


def test_doctor_reports_status(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["doctor"])
    out = capsys.readouterr().out
    assert "Runtime dependencies" in out
    assert "Models" in out
    # Exit is 0 only if fully set up; in a bare test env models are absent.
    assert code in (0, 1)


def test_setup_dry_run_prints_command(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["setup", "--cpu", "--dry-run", "--force"]) == 0
    out = capsys.readouterr().out
    assert "pip" in out
    assert "llama-cpp-python" in out
    assert "abetlen.github.io" in out


def test_run_without_setup_guides_user(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Force "nothing installed" so run() reports guidance, not a traceback.
    monkeypatch.setattr(
        "eva.cli._readiness_problems",
        lambda settings, paths: [
            "missing runtime 'llama_cpp' (language model (LLM)) — run: eva setup"
        ],
    )
    assert main(["run"]) == 1
    out = capsys.readouterr().out
    assert "setup is incomplete" in out.lower()
    assert "eva doctor" in out
