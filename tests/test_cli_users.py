"""CLI user-profile command tests (M4 integration pass)."""

from __future__ import annotations

import json
import re

import pytest

from eva.cli import main


def _created_id(out: str) -> str:
    match = re.search(r"User profile '([^']+)' created", out)
    assert match is not None
    return match.group(1)


def test_create_then_list_then_show(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["users", "create", "--nickname", "Alice", "--units", "imperial"]) == 0
    user_id = _created_id(capsys.readouterr().out)

    assert main(["users", "list"]) == 0
    assert "Alice" in capsys.readouterr().out

    assert main(["users", "show", user_id]) == 0
    out = capsys.readouterr().out
    assert "Alice" in out
    assert "imperial" in out


def test_activate_marks_active_in_list_and_config(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["users", "create", "--nickname", "Bea"]) == 0
    user_id = _created_id(capsys.readouterr().out)

    assert main(["users", "activate", user_id]) == 0
    capsys.readouterr()

    assert main(["users", "list"]) == 0
    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if "Bea" in line]
    assert lines and lines[0].startswith("*")

    assert main(["config", "show"]) == 0
    body = json.loads(capsys.readouterr().out)
    assert body["conversation"]["active_profile_id"] == user_id


def test_edit_updates_fields(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["users", "create", "--nickname", "Carl"]) == 0
    user_id = _created_id(capsys.readouterr().out)

    assert main(["users", "edit", user_id, "--nickname", "Carlos"]) == 0
    capsys.readouterr()
    assert main(["users", "show", user_id]) == 0
    assert "Carlos" in capsys.readouterr().out


def test_delete_removes_profile(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["users", "create", "--nickname", "Dana"]) == 0
    user_id = _created_id(capsys.readouterr().out)

    assert main(["users", "delete", user_id]) == 0
    capsys.readouterr()
    assert main(["users", "list"]) == 0
    assert "Dana" not in capsys.readouterr().out


def test_show_unknown_profile_errors() -> None:
    assert main(["users", "show", "does-not-exist"]) == 1


def test_profile_singular_show_and_use(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["profile", "show"]) == 0
    assert "no active user profile" in capsys.readouterr().out.lower()

    assert main(["users", "create", "--nickname", "Eve"]) == 0
    user_id = _created_id(capsys.readouterr().out)

    assert main(["profile", "use", user_id]) == 0
    capsys.readouterr()
    assert main(["profile", "show"]) == 0
    assert "Eve" in capsys.readouterr().out
