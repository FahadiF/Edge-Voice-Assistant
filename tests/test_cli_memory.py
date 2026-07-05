"""CLI memory command tests (M4 integration pass) — exercises the SQLite
memory store directly through the CLI, no engine/models needed except
`summarize`, which is covered separately with a stubbed LLM.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from eva.cli import main
from eva.config import get_app_paths
from eva.memory.registry import create_stores


def _seed_conversation() -> tuple[str, int]:
    """Create one conversation with a turn directly via the store (faster
    and more deterministic than driving it through `eva run`)."""
    paths = get_app_paths()
    paths.ensure_exists()
    from eva.config import load_settings

    settings = load_settings(paths.settings_file)
    memory, _profiles = create_stores(settings, paths)
    try:
        conv = memory.start_conversation()
        turn = memory.add_turn(conv.id, "user", "my favorite color is teal")
        assert turn.id is not None
        return conv.id, turn.id
    finally:
        memory.close()


def test_stats_reports_zero_on_empty_db(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["memory", "stats"]) == 0
    out = capsys.readouterr().out
    assert "conversation_count" in out


def test_list_and_show(capsys: pytest.CaptureFixture[str]) -> None:
    conv_id, _turn_id = _seed_conversation()
    assert main(["memory", "list"]) == 0
    assert conv_id in capsys.readouterr().out

    assert main(["memory", "show", conv_id]) == 0
    assert "teal" in capsys.readouterr().out


def test_search_finds_keyword(capsys: pytest.CaptureFixture[str]) -> None:
    _seed_conversation()
    assert main(["memory", "search", "teal"]) == 0
    assert "teal" in capsys.readouterr().out


def test_pin_favorite_forget(capsys: pytest.CaptureFixture[str]) -> None:
    conv_id, turn_id = _seed_conversation()
    assert main(["memory", "pin", str(turn_id)]) == 0
    assert main(["memory", "favorite", str(turn_id)]) == 0
    capsys.readouterr()

    assert main(["memory", "show", conv_id]) == 0
    out = capsys.readouterr().out
    assert "pinned" in out
    assert "favorite" in out

    assert main(["memory", "forget", str(turn_id)]) == 0
    capsys.readouterr()
    assert main(["memory", "show", conv_id]) == 0
    assert "teal" not in capsys.readouterr().out


def test_archive_and_restore(capsys: pytest.CaptureFixture[str]) -> None:
    conv_id, _turn_id = _seed_conversation()
    assert main(["memory", "archive", conv_id]) == 0
    capsys.readouterr()
    assert main(["memory", "list"]) == 0
    assert conv_id not in capsys.readouterr().out

    assert main(["memory", "list", "--include-archived"]) == 0
    assert conv_id in capsys.readouterr().out

    assert main(["memory", "archive", conv_id, "--unset"]) == 0
    capsys.readouterr()
    assert main(["memory", "list"]) == 0
    assert conv_id in capsys.readouterr().out


def test_delete_conversation(capsys: pytest.CaptureFixture[str]) -> None:
    conv_id, _turn_id = _seed_conversation()
    assert main(["memory", "delete-conversation", conv_id]) == 0
    capsys.readouterr()
    assert main(["memory", "list"]) == 0
    assert conv_id not in capsys.readouterr().out


def test_export_to_file_then_delete_all_then_import(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _seed_conversation()
    out_file = tmp_path / "snapshot.json"
    assert main(["memory", "export", "--out", str(out_file)]) == 0
    capsys.readouterr()
    assert out_file.exists()
    payload = json.loads(out_file.read_text(encoding="utf-8"))
    assert payload

    assert main(["memory", "delete-all", "--yes"]) == 0
    capsys.readouterr()
    assert main(["memory", "stats"]) == 0
    stats = dict(
        line.split(None, 1) for line in capsys.readouterr().out.splitlines() if line.strip()
    )
    assert stats["turn_count"] == "0"

    assert main(["memory", "import", str(out_file)]) == 0
    out = capsys.readouterr().out
    assert re.search(r"Imported \d+ turn", out)


def test_delete_all_requires_confirmation(capsys: pytest.CaptureFixture[str]) -> None:
    _seed_conversation()
    assert main(["memory", "delete-all"]) == 1
    assert "--yes" in capsys.readouterr().err
