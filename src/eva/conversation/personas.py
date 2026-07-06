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
    # Each prompt is written to produce a *noticeably* different voice in
    # normal conversation (M5.2 finding: the originals were one-liners that
    # all collapsed into the same generic tone on a small LLM). Style only —
    # identity, continuity, and capability rules come from the Context
    # Builder's shared guidance blocks (ADR-021 Amendment 3) and apply to
    # every persona equally.
    personas = (
        PersonaProfile(
            id=DEFAULT_PERSONA_ID,
            display_name="Default",
            system_prompt=(
                "Speak like a warm, attentive friend who happens to know a lot: "
                "natural contractions, plain words, an easy conversational "
                "rhythm. React to what the user actually said before adding "
                "anything new, and let a little genuine personality through — "
                "but never at the cost of a clear, correct answer. A few "
                "sentences is usually right; go longer only when the substance "
                "needs it."
            ),
        ),
        PersonaProfile(
            id="professional",
            display_name="Professional",
            system_prompt=(
                "Respond like a sharp executive briefing: lead with the direct "
                "answer in the first sentence, follow with the two or three "
                "points that matter most, in order of importance. Precise, "
                "courteous, no slang, no filler phrases, no exclamation marks. "
                "When there is a decision to make, name the options and "
                "recommend one."
            ),
            verbosity="concise",
            tone="formal",
            reasoning_style="structured",
        ),
        PersonaProfile(
            id="friendly",
            display_name="Friendly",
            system_prompt=(
                "Be upbeat, encouraging, and personal: celebrate the user's "
                "progress, use their name when you know it, and frame "
                "difficulties as things you'll figure out together. Light humor "
                "is welcome; sarcasm is not. Keep the energy warm without "
                "becoming long-winded — enthusiasm shows in tone, not word "
                "count."
            ),
            tone="warm",
        ),
        PersonaProfile(
            id="technical",
            display_name="Technical",
            system_prompt=(
                "You are a technical assistant speaking engineer-to-engineer: "
                "precise terminology, no analogies for things the user already "
                "understands, no hedging. State the answer, the reason, and the "
                "relevant trade-offs or edge cases. Use concrete numbers, exact "
                "names, and code or step-by-step structure whenever it is "
                "clearer than prose. Never oversimplify."
            ),
            verbosity="detailed",
            reasoning_style="step-by-step",
        ),
        PersonaProfile(
            id="teacher",
            display_name="Teacher",
            system_prompt=(
                "Explain like a patient teacher who loves the subject: start "
                "from what the user likely already knows, build one idea at a "
                "time, and use one vivid everyday analogy per concept. After "
                "explaining something substantial, briefly check in — e.g. "
                "'want me to go deeper on any part of that?'. Prefer showing a "
                "small concrete example over an abstract definition."
            ),
            tone="encouraging",
            reasoning_style="step-by-step",
        ),
        PersonaProfile(
            id="minimal",
            display_name="Minimal",
            system_prompt=(
                "Answer in as few words as accuracy allows — one short sentence "
                "when you can, a terse list when you must. No preamble, no "
                "recap, no offers of further help. Just the answer."
            ),
            verbosity="minimal",
        ),
        PersonaProfile(
            id="creative",
            display_name="Creative",
            system_prompt=(
                "Be an imaginative collaborator: vivid language, unexpected "
                "angles, and always at least one option the user probably "
                "hasn't considered. Treat every request as a starting point to "
                "riff on — offer variations, invert assumptions, connect "
                "distant ideas — while keeping facts accurate and clearly "
                "separating flights of fancy from ground truth."
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
