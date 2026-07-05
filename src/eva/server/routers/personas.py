"""Persona API (M4, ADR-022): list/get built-in + custom personas, create/
delete custom ones. Personas are configuration, not conversation data — this
router works without a running engine, reading/writing `state.settings`
directly (the same pattern the settings API uses), consistent with
built-ins being code and custom personas being settings data.
"""

from __future__ import annotations

from fastapi import APIRouter

from eva.config.settings import PersonaSettingsEntry, save_settings
from eva.conversation.personas import (
    PersonaProfile,
    persona_registry,
    register_builtin_personas,
    register_custom_personas,
)
from eva.core.errors import ConfigError
from eva.server.deps import StateDep

router = APIRouter(prefix="/personas", tags=["personas"])


def _sync_registry(state: StateDep) -> None:
    register_builtin_personas()
    register_custom_personas(state.settings)


@router.get("", response_model=list[PersonaProfile])
def list_personas(state: StateDep) -> list[PersonaProfile]:
    _sync_registry(state)
    return list(persona_registry.snapshot().values())


@router.get("/{persona_id}", response_model=PersonaProfile)
def get_persona(persona_id: str, state: StateDep) -> PersonaProfile:
    _sync_registry(state)
    return persona_registry.get(persona_id)


@router.post("", response_model=PersonaProfile)
def create_persona(payload: PersonaSettingsEntry, state: StateDep) -> PersonaProfile:
    """Add or replace a custom persona. Rejects reusing a built-in's id —
    settings data must not silently shadow a built-in a user didn't ask to
    override (see ADR-022's note on this exact failure mode)."""
    register_builtin_personas()
    existing = {p.id for p in persona_registry.snapshot().values()}
    custom_ids = {p.id for p in state.settings.conversation.custom_personas}
    if payload.id in existing - custom_ids:
        raise ConfigError(f"'{payload.id}' is a built-in persona id and cannot be overridden")

    remaining = [p for p in state.settings.conversation.custom_personas if p.id != payload.id]
    state.settings.conversation.custom_personas = [*remaining, payload]
    save_settings(state.settings, state.paths.settings_file)
    register_custom_personas(state.settings)
    return persona_registry.get(payload.id)


@router.delete("/{persona_id}")
def delete_persona(persona_id: str, state: StateDep) -> dict[str, str]:
    custom_ids = {p.id for p in state.settings.conversation.custom_personas}
    if persona_id not in custom_ids:
        raise ConfigError(f"'{persona_id}' is not a custom persona (built-ins cannot be deleted)")
    state.settings.conversation.custom_personas = [
        p for p in state.settings.conversation.custom_personas if p.id != persona_id
    ]
    save_settings(state.settings, state.paths.settings_file)
    persona_registry.unregister(persona_id)
    return {"status": "deleted"}
