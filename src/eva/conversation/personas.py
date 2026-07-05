"""Persona registry: personality profiles (ADR-022).

One `PersonaProfile` per personality — system prompt, verbosity, tone,
reasoning style, and an optional sampling-temperature override. Built-ins
are code, mirroring `eva.conversation.language`'s pattern exactly. Custom,
user-created personas are settings data (`ConversationSettings.
custom_personas`), registered alongside the built-ins at resolve time — a
persona is configuration, not conversation history (ADR-022).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from eva.config.settings import Settings
from eva.core.registry import Registry

DEFAULT_PERSONA_ID = "default"


class PersonaProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    display_name: str
    system_prompt: str
    verbosity: str = "normal"
    tone: str = "neutral"
    reasoning_style: str = "direct"
    temperature_override: float | None = None


persona_registry: Registry[PersonaProfile] = Registry("persona")


def register_builtin_personas() -> None:
    personas = (
        PersonaProfile(
            id=DEFAULT_PERSONA_ID,
            display_name="Default",
            system_prompt=(
                "You are a helpful voice assistant. Answer conversationally and "
                "concisely — one to three short sentences unless asked for detail."
            ),
        ),
        PersonaProfile(
            id="professional",
            display_name="Professional",
            system_prompt=(
                "You are a professional assistant. Respond formally and precisely, "
                "avoiding slang or casual phrasing."
            ),
            verbosity="concise",
            tone="formal",
            reasoning_style="structured",
        ),
        PersonaProfile(
            id="friendly",
            display_name="Friendly",
            system_prompt=(
                "You are a warm, friendly assistant. Respond conversationally, with "
                "encouragement and a positive tone."
            ),
            tone="warm",
        ),
        PersonaProfile(
            id="technical",
            display_name="Technical",
            system_prompt=(
                "You are a technical assistant for an experienced audience. Be precise, "
                "use correct terminology, and do not oversimplify."
            ),
            verbosity="detailed",
            reasoning_style="step-by-step",
        ),
        PersonaProfile(
            id="minimal",
            display_name="Minimal",
            system_prompt="Answer as briefly as possible. One short sentence when you can.",
            verbosity="minimal",
        ),
        PersonaProfile(
            id="creative",
            display_name="Creative",
            system_prompt=(
                "You are an imaginative, creative assistant. Feel free to use vivid "
                "language and offer unconventional ideas."
            ),
            tone="playful",
            reasoning_style="exploratory",
            temperature_override=0.9,
        ),
    )
    for persona in personas:
        if persona.id not in persona_registry:
            persona_registry.register(persona.id, persona)


def register_custom_personas(settings: Settings) -> None:
    """Register user-created personas from settings (ADR-022). Safe to call
    repeatedly — re-registers with `replace=True` so an edit to an existing
    custom persona in settings takes effect without a process restart."""
    for entry in settings.conversation.custom_personas:
        persona = PersonaProfile(
            id=entry.id,
            display_name=entry.display_name,
            system_prompt=entry.system_prompt,
            verbosity=entry.verbosity,
            tone=entry.tone,
            reasoning_style=entry.reasoning_style,
            temperature_override=entry.temperature_override,
        )
        persona_registry.register(persona.id, persona, replace=True)


def resolve_persona(settings: Settings) -> PersonaProfile:
    """The active persona (settings.conversation.persona). Raises
    `RegistryError` for an unknown id — mirrors `resolve_language`'s
    behavior exactly (validate at settings-write time, don't silently
    substitute a default at resolve time)."""
    register_builtin_personas()
    register_custom_personas(settings)
    return persona_registry.get(settings.conversation.persona)
