"""Permission-gated system information tests (ADR-025, regrouped in M5.4)."""

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


def _perms(**overrides: object) -> PermissionsSettings:
    perms = PermissionsSettings()
    for dotted, value in overrides.items():
        group, _, field = dotted.partition("__")
        setattr(getattr(perms, group), field, value)
    return perms


class TestFactsBlock:
    def test_default_permissions_include_read_only_facts(self) -> None:
        block = system_facts_block(PermissionsSettings())
        assert "date and time" in block
        assert "Timezone" in block
        assert "Operating system" in block

    def test_current_date_is_fresh(self) -> None:
        block = system_facts_block(PermissionsSettings())
        assert datetime.now().astimezone().strftime("%Y") in block

    def test_all_general_permissions_off_yields_empty_block(self) -> None:
        perms = _perms(general__date_time=False, general__system_information=False)
        assert system_facts_block(perms) == ""

    def test_date_time_toggle_covers_date_and_timezone(self) -> None:
        without_dt = system_facts_block(_perms(general__date_time=False))
        assert "date and time" not in without_dt
        assert "Timezone" not in without_dt
        assert "Operating system" in without_dt  # system_information unaffected

    def test_system_information_toggle_covers_hardware_and_os(self) -> None:
        without_si = system_facts_block(_perms(general__system_information=False))
        assert "Operating system" not in without_si
        assert "CPU" not in without_si
        assert "date and time" in without_si  # date_time unaffected

    def test_action_permissions_default_off(self) -> None:
        perms = PermissionsSettings()
        assert perms.general.internet is False
        assert perms.files.read_files is False
        assert perms.files.write_files is False
        assert perms.devices.camera is False
        assert perms.tools.browser is False
        assert perms.tools.python is False
        assert perms.tools.shell is False

    def test_core_function_permissions_default_on(self) -> None:
        perms = PermissionsSettings()
        assert perms.general.date_time is True
        assert perms.general.system_information is True
        assert perms.devices.microphone is True
        assert perms.tools.plugins is True
        assert perms.privacy.remember_conversations is True


class TestContextIntegration:
    def test_facts_injected_into_system_message(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation()
        built = ContextBuilder(Settings(), store).build(conv.id, "what time is it?")
        system = built.messages[0].content
        assert "Local system information" in system
        assert "date and time" in system

    def test_facts_omitted_when_all_denied(self, store: SQLiteMemoryStore) -> None:
        settings = Settings()
        settings.permissions = _perms(general__date_time=False, general__system_information=False)
        conv = store.start_conversation()
        built = ContextBuilder(settings, store).build(conv.id, "what time is it?")
        assert "Local system information" not in built.messages[0].content

    def test_permission_gap_guidance_present(self, store: SQLiteMemoryStore) -> None:
        """With a permission off, the prompt tells the model to attribute the
        gap to permissions — not to permanent inability."""
        conv = store.start_conversation()
        system = ContextBuilder(Settings(), store).build(conv.id, "hi").messages[0].content
        assert "has not granted" in system

    def test_facts_precede_technical_backend_section(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation()
        system = ContextBuilder(Settings(), store).build(conv.id, "hi").messages[0].content
        assert system.index("Local system information") < system.index("Technical backend details")


class TestPrivacyEnforcement:
    def test_remember_conversations_off_stores_nothing(self, store: SQLiteMemoryStore) -> None:
        """privacy.remember_conversations is the real gate on turn storage
        (replaces the dead conversation.memory_enabled flag)."""
        import asyncio

        from eva.audio.segmenter import UtteranceEnd
        from eva.conversation.orchestrator import Orchestrator
        from eva.core.events import EventBus
        from tests.server_fakes import FakeMemoryStore
        from tests.test_orchestrator import (
            AUDIO,
            FakeASR,
            FakeAudioOut,
            FakeLLM,
            FakeTTS,
            drive,
        )

        async def scenario() -> None:
            settings = Settings()
            settings.permissions.privacy.remember_conversations = False
            memory = FakeMemoryStore()
            bus = EventBus()
            orch = Orchestrator(
                settings, bus, FakeAudioOut(), FakeASR(), FakeLLM(), FakeTTS(), memory
            )

            async def script() -> None:
                orch.feed_audio_event(UtteranceEnd(AUDIO, 1000, 800, False))
                for _ in range(200):
                    if orch._turn_task is not None and orch._turn_task.done():
                        break
                    await asyncio.sleep(0.01)

            await drive(orch, bus, script)
            assert memory.recent_turns(orch.conversation_id, 10) == []

        asyncio.run(scenario())


class TestSettingsMigration:
    def test_v1_flat_permissions_migrate_to_groups(self, tmp_path: Path) -> None:
        import json

        from eva.config.settings import load_settings

        v1 = {
            "schema_version": 1,
            "permissions": {
                "date_time": True,
                "timezone": True,
                "locale": True,
                "cpu": False,
                "gpu": False,
                "ram": False,
                "os": False,
                "internet": False,
                "local_files": True,
                "camera": False,
                "clipboard": False,
                "browser": True,
                "shell": False,
                "python": False,
                "plugins": True,
            },
            "conversation": {"memory_enabled": False},
        }
        path = tmp_path / "settings.json"
        path.write_text(json.dumps(v1), encoding="utf-8")
        settings = load_settings(path)
        assert settings.schema_version == 2
        assert settings.permissions.general.date_time is True
        # cpu/gpu/ram/os all False, locale True → any() keeps it on
        assert settings.permissions.general.system_information is True
        assert settings.permissions.files.read_files is True  # was local_files
        assert settings.permissions.tools.browser is True
        # memory_enabled=False carried into the new privacy toggle
        assert settings.permissions.privacy.remember_conversations is False

    def test_v1_without_permissions_section_migrates(self, tmp_path: Path) -> None:
        import json

        from eva.config.settings import load_settings

        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"schema_version": 1, "profile": "fast"}), encoding="utf-8")
        settings = load_settings(path)
        assert settings.schema_version == 2
        assert settings.profile == "fast"
        assert settings.permissions.devices.microphone is True

    def test_v2_documents_load_unchanged(self, tmp_path: Path) -> None:

        from eva.config.settings import load_settings, save_settings

        path = tmp_path / "settings.json"
        original = Settings()
        original.permissions.general.date_time = False
        save_settings(original, path)
        loaded = load_settings(path)
        assert loaded.permissions.general.date_time is False
