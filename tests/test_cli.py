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


def test_run_not_ready_returns_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    from eva.onboarding import OnboardingResult

    # Onboarding could not complete (e.g. non-interactive, nothing installed).
    monkeypatch.setattr(
        "eva.onboarding.run_onboarding",
        lambda *a, **k: OnboardingResult(ready=False, declined=False),
    )
    assert main(["run"]) == 1


def test_run_user_declined_returns_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    from eva.onboarding import OnboardingResult

    monkeypatch.setattr(
        "eva.onboarding.run_onboarding",
        lambda *a, **k: OnboardingResult(ready=False, declined=True),
    )
    assert main(["run"]) == 0


def test_first_run_setup_only(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from eva.onboarding import OnboardingResult

    monkeypatch.setattr(
        "eva.onboarding.run_onboarding", lambda *a, **k: OnboardingResult(ready=True)
    )
    assert main(["first-run", "--setup-only"]) == 0
    assert "eva run" in capsys.readouterr().out
