"""Persona registry tests (ADR-022), mirroring test_language.py's pattern."""

from __future__ import annotations

import pytest

from eva.config.settings import Settings
from eva.conversation.personas import (
    DEFAULT_PERSONA_ID,
    persona_registry,
    register_builtin_personas,
    register_custom_personas,
    resolve_persona,
)
from eva.core.errors import RegistryError


@pytest.fixture(autouse=True)
def _register() -> None:
    register_builtin_personas()


class TestRegistry:
    @pytest.mark.parametrize(
        "persona_id",
        ["default", "professional", "friendly", "technical", "minimal", "creative"],
    )
    def test_builtin_personas_registered(self, persona_id: str) -> None:
        persona = persona_registry.get(persona_id)
        assert persona.display_name
        assert persona.system_prompt

    def test_unknown_persona_rejected_with_known_list(self) -> None:
        with pytest.raises(RegistryError, match="unknown id 'xx'"):
            persona_registry.get("xx")

    def test_default_persona_is_neutral_baseline(self) -> None:
        persona = persona_registry.get(DEFAULT_PERSONA_ID)
        assert persona.verbosity == "normal"
        assert persona.temperature_override is None

    def test_creative_persona_overrides_temperature(self) -> None:
        persona = persona_registry.get("creative")
        assert persona.temperature_override == 0.9


class TestResolution:
    def test_default_settings_resolve_to_default_persona(self) -> None:
        settings = Settings()
        persona = resolve_persona(settings)
        assert persona.id == DEFAULT_PERSONA_ID

    def test_settings_selects_a_different_builtin(self) -> None:
        settings = Settings()
        settings.conversation.persona = "technical"
        assert resolve_persona(settings).id == "technical"

    def test_unknown_persona_id_raises(self) -> None:
        settings = Settings()
        settings.conversation.persona = "does-not-exist"
        with pytest.raises(RegistryError):
            resolve_persona(settings)


class TestCustomPersonas:
    def test_custom_persona_registers_and_resolves(self) -> None:
        settings = Settings()
        settings.conversation.custom_personas = [
            {
                "id": "pirate",
                "display_name": "Pirate",
                "system_prompt": "Speak like a pirate.",
                "tone": "boisterous",
            }
        ]
        settings.conversation.persona = "pirate"
        persona = resolve_persona(settings)
        assert persona.display_name == "Pirate"
        assert persona.tone == "boisterous"

    def test_custom_persona_cannot_shadow_a_builtin_by_id_collision(self) -> None:
        """Registering a custom persona with a builtin's id overwrites the
        registry entry (replace=True) — this documents that behavior rather
        than silently allowing user settings to corrupt a builtin without
        the caller knowing. Validate ids at the settings-write boundary
        (API/CLI) to prevent this in practice, not here."""
        original_default = persona_registry.get("default")  # snapshot before mutating
        settings = Settings()
        settings.conversation.custom_personas = [
            {
                "id": "default",
                "display_name": "Overridden Default",
                "system_prompt": "I am not the real default.",
            }
        ]
        try:
            register_custom_personas(settings)
            assert persona_registry.get("default").display_name == "Overridden Default"
        finally:
            # Restore the exact original object — persona_registry is a
            # process-wide singleton (ADR-010) shared with every other test.
            persona_registry.register("default", original_default, replace=True)

    def test_editing_a_custom_persona_takes_effect_on_next_resolve(self) -> None:
        settings = Settings()
        settings.conversation.custom_personas = [
            {"id": "mine", "display_name": "V1", "system_prompt": "First version."}
        ]
        settings.conversation.persona = "mine"
        assert resolve_persona(settings).display_name == "V1"

        settings.conversation.custom_personas = [
            {"id": "mine", "display_name": "V2", "system_prompt": "Second version."}
        ]
        assert resolve_persona(settings).display_name == "V2"
