"""Multilingual foundation tests (English, Finnish, Swedish, Bengali)."""

from __future__ import annotations

import logging

import pytest

from eva.config.settings import Settings
from eva.conversation.language import (
    effective_asr_language,
    effective_system_prompt,
    effective_voice,
    language_registry,
    register_builtin_languages,
    resolve_language,
)
from eva.core.errors import RegistryError


@pytest.fixture(autouse=True)
def _register() -> None:
    register_builtin_languages()


class TestRegistry:
    @pytest.mark.parametrize("code", ["en", "fi", "sv", "bn"])
    def test_required_languages_registered(self, code: str) -> None:
        profile = language_registry.get(code)
        assert profile.asr_language == code
        assert profile.display_name

    def test_unknown_language_rejected_with_known_list(self) -> None:
        with pytest.raises(RegistryError, match="unknown id 'xx'"):
            language_registry.get("xx")


class TestResolution:
    def test_english_default_no_prompt_note(self) -> None:
        settings = Settings()
        lang = resolve_language(settings)
        assert lang.code == "en"
        prompt = effective_system_prompt(settings, lang)
        assert prompt == settings.conversation.system_prompt

    @pytest.mark.parametrize(
        ("code", "expected_fragment"),
        [("fi", "suomeksi"), ("sv", "svenska"), ("bn", "বাংলায়")],
    )
    def test_language_note_appended_to_prompt(self, code: str, expected_fragment: str) -> None:
        settings = Settings()
        settings.conversation.language = code
        lang = resolve_language(settings)
        assert expected_fragment in effective_system_prompt(settings, lang)

    @pytest.mark.parametrize("code", ["fi", "sv", "bn"])
    def test_asr_hint_follows_conversation_language(self, code: str) -> None:
        settings = Settings()
        settings.conversation.language = code
        assert effective_asr_language(settings, resolve_language(settings)) == code

    def test_explicit_asr_override_wins(self) -> None:
        settings = Settings()
        settings.conversation.language = "fi"
        settings.asr.language = "en"
        assert effective_asr_language(settings, resolve_language(settings)) == "en"

    def test_english_has_native_voice(self) -> None:
        settings = Settings()
        assert effective_voice(settings, resolve_language(settings)) == "af_heart"

    @pytest.mark.parametrize("code", ["fi", "sv", "bn"])
    def test_missing_voice_falls_back_with_warning(
        self, code: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        settings = Settings()
        settings.conversation.language = code
        with caplog.at_level(logging.WARNING):
            voice = effective_voice(settings, resolve_language(settings))
        assert voice == settings.tts.voice  # graceful fallback, never a crash
        assert any("non-native voice" in r.message for r in caplog.records)


class TestOrchestratorIntegration:
    def test_orchestrator_speaks_configured_language(self) -> None:
        """Language flows into prompt composition and ASR hints end to end."""
        from tests.test_orchestrator import make_orchestrator

        orch, _, _, _ = make_orchestrator()
        orch_settings = orch._settings
        orch_settings.conversation.language = "fi"
        # Rebuild with Finnish configured.
        from eva.conversation.orchestrator import Orchestrator
        from eva.core.events import EventBus
        from tests.test_orchestrator import FakeASR, FakeAudioOut, FakeLLM, FakeTTS

        orch2 = Orchestrator(
            orch_settings, EventBus(), FakeAudioOut(), FakeASR(), FakeLLM(), FakeTTS()
        )
        assert orch2._asr_language == "fi"
        messages = orch2._history.messages("hei")
        assert "suomeksi" in messages[0].content
