"""CLI persona command tests (M4 integration pass)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from eva.cli import main
from eva.conversation.personas import persona_registry

_TEST_PERSONA_IDS = ("pirate", "temp")


@pytest.fixture(autouse=True)
def _cleanup_test_personas() -> Iterator[None]:
    """`persona_registry` is a process-wide singleton (ADR-010) shared with
    every other test file — any custom persona created here must be
    unregistered afterward or it leaks into other tests' `existing -
    custom_ids` collision checks (e.g. `tests/test_server_personas.py`,
    `tests/test_personas.py`, both of which also use the id "pirate")."""
    yield
    for persona_id in _TEST_PERSONA_IDS:
        if persona_id in persona_registry:
            persona_registry.unregister(persona_id)


def test_list_shows_builtins_and_marks_active(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["personas", "list"]) == 0
    out = capsys.readouterr().out
    assert "default" in out
    assert "technical" in out
    assert "* = active persona" in out


def test_show_prints_persona_fields(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["personas", "show", "creative"]) == 0
    out = capsys.readouterr().out
    assert "creative" in out.lower()


def test_show_unknown_id_errors(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["personas", "show", "does-not-exist"]) == 1
    assert "unknown id" in capsys.readouterr().err.lower()


def test_create_then_use_then_list_marks_it_active(capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        main(
            [
                "personas",
                "create",
                "--id",
                "pirate",
                "--name",
                "Pirate",
                "--prompt",
                "Speak like a pirate.",
                "--tone",
                "boisterous",
            ]
        )
        == 0
    )
    assert main(["personas", "use", "pirate"]) == 0
    capsys.readouterr()
    assert main(["personas", "list"]) == 0
    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if "pirate" in line]
    assert lines and lines[0].startswith("*")


def test_create_cannot_shadow_builtin(capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        main(
            [
                "personas",
                "create",
                "--id",
                "default",
                "--name",
                "Fake Default",
                "--prompt",
                "x",
            ]
        )
        == 1
    )
    assert "built-in" in capsys.readouterr().err.lower()


def test_use_unknown_persona_errors(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["personas", "use", "nope"]) == 1
    assert "unknown id" in capsys.readouterr().err.lower()


def test_delete_builtin_rejected(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["personas", "delete", "default"]) == 1
    assert "built-in" in capsys.readouterr().err.lower()


def test_create_then_delete_custom_persona() -> None:
    assert (
        main(
            [
                "personas",
                "create",
                "--id",
                "temp",
                "--name",
                "Temp",
                "--prompt",
                "x",
            ]
        )
        == 0
    )
    assert main(["personas", "delete", "temp"]) == 0
