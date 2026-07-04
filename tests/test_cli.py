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


def test_keyboard_interrupt_exits_cleanly(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """M3: Ctrl+C during any command must exit with a clean message and a
    conventional exit code, never an uncaught traceback."""

    def _boom(*_a: object, **_k: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr("eva.cli.detect_hardware", _boom)
    code = main(["diagnose"])
    assert code == 130
    assert "Cancelled" in capsys.readouterr().err


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


def test_config_show_prints_json(capsys: pytest.CaptureFixture[str]) -> None:
    import json

    assert main(["config", "show"]) == 0
    body = json.loads(capsys.readouterr().out)
    assert "llm" in body


def test_config_schema_prints_json_schema(capsys: pytest.CaptureFixture[str]) -> None:
    import json

    assert main(["config", "schema"]) == 0
    schema = json.loads(capsys.readouterr().out)
    assert "VADSettings" in schema["$defs"]


def test_config_reset(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["config", "reset"]) == 0
    assert "reset to defaults" in capsys.readouterr().out.lower()


def test_serve_command_starts_uvicorn(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {}

    def fake_run(app: object, host: str, port: int, log_level: str) -> None:
        calls["host"] = host
        calls["port"] = port

    monkeypatch.setattr("uvicorn.run", fake_run)
    assert main(["serve", "--host", "0.0.0.0", "--port", "9999"]) == 0
    assert calls == {"host": "0.0.0.0", "port": 9999}
