"""Permission-gated system information tests (M5.3, ADR-025)."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import pytest

from eva.config.settings import PermissionsSettings, Settings
from eva.conversation.context_builder import ContextBuilder
from eva.conversation.system_info import system_facts_block
from eva.memory import db
from eva.memory.sqlite_store import SQLiteMemoryStore


@pytest.fixture
def store(tmp_path: Path) -> Iterator[SQLiteMemoryStore]:
    conn = db.connect(tmp_path / "memory.db")
    s = SQLiteMemoryStore(conn)
    yield s
    s.close()


class TestFactsBlock:
    def test_default_permissions_include_read_only_facts(self) -> None:
        block = system_facts_block(PermissionsSettings())
        assert "date and time" in block
        assert "Timezone" in block
        assert "Operating system" in block

    def test_current_date_is_fresh(self) -> None:
        block = system_facts_block(PermissionsSettings())
        assert datetime.now().astimezone().strftime("%Y") in block

    def test_all_permissions_off_yields_empty_block(self) -> None:
        perms = PermissionsSettings(
            date_time=False,
            timezone=False,
            locale=False,
            cpu=False,
            gpu=False,
            ram=False,
            os=False,
        )
        assert system_facts_block(perms) == ""

    def test_individual_toggle_removes_only_its_fact(self) -> None:
        with_dt = system_facts_block(PermissionsSettings())
        without_dt = system_facts_block(PermissionsSettings(date_time=False))
        assert "date and time" in with_dt
        assert "date and time" not in without_dt
        assert "Operating system" in without_dt  # others unaffected

    def test_action_permissions_default_off(self) -> None:
        perms = PermissionsSettings()
        for name in (
            "internet",
            "local_files",
            "camera",
            "clipboard",
            "browser",
            "shell",
            "python",
        ):
            assert getattr(perms, name) is False, f"{name} must default to off"

    def test_read_only_permissions_default_on(self) -> None:
        perms = PermissionsSettings()
        for name in ("date_time", "timezone", "locale", "cpu", "gpu", "ram", "os"):
            assert getattr(perms, name) is True, f"{name} must default to on"


class TestContextIntegration:
    def test_facts_injected_into_system_message(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation()
        built = ContextBuilder(Settings(), store).build(conv.id, "what time is it?")
        system = built.messages[0].content
        assert "Local system information" in system
        assert "date and time" in system

    def test_facts_omitted_when_all_denied(self, store: SQLiteMemoryStore) -> None:
        settings = Settings()
        settings.permissions = PermissionsSettings(
            date_time=False,
            timezone=False,
            locale=False,
            cpu=False,
            gpu=False,
            ram=False,
            os=False,
        )
        conv = store.start_conversation()
        built = ContextBuilder(settings, store).build(conv.id, "what time is it?")
        assert "Local system information" not in built.messages[0].content

    def test_permission_gap_guidance_present(self, store: SQLiteMemoryStore) -> None:
        """With a permission off, the prompt tells the model to attribute the
        gap to permissions — not to permanent inability (M5.3 §Permissions)."""
        conv = store.start_conversation()
        system = ContextBuilder(Settings(), store).build(conv.id, "hi").messages[0].content
        assert "has not granted" in system

    def test_facts_precede_technical_backend_section(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation()
        system = ContextBuilder(Settings(), store).build(conv.id, "hi").messages[0].content
        assert system.index("Local system information") < system.index("Technical backend details")
