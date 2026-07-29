"""Tests for _migrate_raw schema migrations in settings.py.

Covers the v4→v5 A8 bounded-chunking migration and verifies that
earlier migrations (v3→v4 max_tokens) still compose correctly.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from eva.config.settings import (
    SETTINGS_SCHEMA_VERSION,
    Settings,
    _migrate_raw,
    load_settings,
    save_settings,
)

# ── helpers ──────────────────────────────────────────────────────────────


def _v4_doc(**conversation_overrides: object) -> dict:
    """Return a minimal v4 settings document with the old defaults."""
    conv = {
        "sentence_min_chars": 12,
        "sentence_max_chars": 350,
        "first_sentence_min_chars": 4,
        "max_tokens": 2048,
    }
    conv.update(conversation_overrides)
    return {"schema_version": 4, "conversation": conv}


def _v3_doc(**conversation_overrides: object) -> dict:
    """Return a minimal v3 document (pre max_tokens migration)."""
    conv = {
        "sentence_min_chars": 12,
        "sentence_max_chars": 350,
        "first_sentence_min_chars": 4,
        "max_tokens": 512,
    }
    conv.update(conversation_overrides)
    return {"schema_version": 3, "conversation": conv}


# ── v4 → v5 (A8: sentence_max_chars 350 → 50) ───────────────────────────


class TestA8Migration:
    def test_old_default_migrates(self) -> None:
        """sentence_max_chars == 350 (old default) migrates to 50."""
        raw = _v4_doc()
        result = _migrate_raw(raw)
        assert result["conversation"]["sentence_max_chars"] == 50
        assert result["schema_version"] == SETTINGS_SCHEMA_VERSION

    def test_custom_value_preserved(self) -> None:
        """A user who deliberately set sentence_max_chars != 350 keeps it."""
        raw = _v4_doc(sentence_max_chars=200)
        result = _migrate_raw(raw)
        assert result["conversation"]["sentence_max_chars"] == 200

    def test_custom_350_after_migration(self, tmp_path: Path) -> None:
        """Post-migration, a user can deliberately set 350 and it sticks."""
        settings = Settings()
        assert settings.conversation.sentence_max_chars == 50  # new default
        settings.conversation.sentence_max_chars = 350
        path = tmp_path / "settings.json"
        save_settings(settings, path)

        reloaded = load_settings(path)
        assert reloaded.conversation.sentence_max_chars == 350

    def test_unrelated_settings_untouched(self) -> None:
        """Migration does not alter fields it should not touch."""
        raw = _v4_doc()
        raw["conversation"]["temperature"] = 0.7
        raw["conversation"]["top_p"] = 0.95
        raw["audio"] = {"mic_gain": 2.5}
        original = copy.deepcopy(raw)
        result = _migrate_raw(raw)
        assert result["conversation"]["temperature"] == original["conversation"]["temperature"]
        assert result["conversation"]["top_p"] == original["conversation"]["top_p"]
        assert result["audio"] == original["audio"]
        assert result["conversation"]["sentence_min_chars"] == 12

    def test_idempotent(self) -> None:
        """Running migration twice yields the same result."""
        raw = _v4_doc()
        first = _migrate_raw(copy.deepcopy(raw))
        second = _migrate_raw(copy.deepcopy(first))
        assert first == second
        assert second["conversation"]["sentence_max_chars"] == 50
        assert second["schema_version"] == SETTINGS_SCHEMA_VERSION

    def test_already_v5_not_touched(self) -> None:
        """A document already at v5 is returned unchanged."""
        raw = {"schema_version": 5, "conversation": {"sentence_max_chars": 350}}
        result = _migrate_raw(raw)
        # Should NOT migrate because schema_version >= SETTINGS_SCHEMA_VERSION
        assert result["conversation"]["sentence_max_chars"] == 350

    def test_missing_conversation_section(self) -> None:
        """A v4 document without a conversation section doesn't crash."""
        raw = {"schema_version": 4}
        result = _migrate_raw(raw)
        assert result["schema_version"] == SETTINGS_SCHEMA_VERSION

    def test_full_round_trip(self, tmp_path: Path) -> None:
        """A v4 settings file on disk gets migrated on load."""
        path = tmp_path / "settings.json"
        path.write_text(json.dumps(_v4_doc()), encoding="utf-8")
        settings = load_settings(path)
        assert settings.conversation.sentence_max_chars == 50
        assert settings.schema_version == SETTINGS_SCHEMA_VERSION


# ── v3 → v4 max_tokens migration still works ─────────────────────────────


class TestMaxTokensMigrationComposition:
    def test_max_tokens_still_migrates(self) -> None:
        """v3 documents with max_tokens=512 still migrate to 2048."""
        raw = _v3_doc()
        result = _migrate_raw(raw)
        assert result["conversation"]["max_tokens"] == 2048
        # And A8 also applied
        assert result["conversation"]["sentence_max_chars"] == 50
        assert result["schema_version"] == SETTINGS_SCHEMA_VERSION

    def test_custom_max_tokens_preserved(self) -> None:
        """Custom max_tokens is not overwritten."""
        raw = _v3_doc(max_tokens=1024)
        result = _migrate_raw(raw)
        assert result["conversation"]["max_tokens"] == 1024

    def test_v3_full_chain(self, tmp_path: Path) -> None:
        """A v3 file migrates both max_tokens and sentence_max_chars."""
        path = tmp_path / "settings.json"
        path.write_text(json.dumps(_v3_doc()), encoding="utf-8")
        settings = load_settings(path)
        assert settings.conversation.max_tokens == 2048
        assert settings.conversation.sentence_max_chars == 50
        assert settings.schema_version == SETTINGS_SCHEMA_VERSION
