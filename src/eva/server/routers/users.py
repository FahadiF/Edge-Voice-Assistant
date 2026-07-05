"""User profile API (M4, ADR-022): nickname, preferred language/voice/model,
conversation style, units, timezone. Backed by `UserProfileStore`
(ADR-019's memory database) — a running engine is required.

Named "users", not "profiles": `Settings.profile`/`eva profiles` already
means the hardware/model preset (ADR-015) — a different concept entirely
(see ADR-022's naming note).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from fastapi import APIRouter

from eva.memory.models import UserProfile
from eva.server.deps import StateDep
from eva.server.schemas import CreateUserProfileRequest, UpdateUserProfileRequest

router = APIRouter(prefix="/users", tags=["users"])


@router.get("", response_model=list[UserProfile])
def list_users(state: StateDep) -> list[UserProfile]:
    return state.require_assistant().profiles.list()


@router.get("/{user_id}", response_model=UserProfile)
def get_user(user_id: str, state: StateDep) -> UserProfile:
    return state.require_assistant().profiles.get(user_id)


@router.post("", response_model=UserProfile)
def create_user(payload: CreateUserProfileRequest, state: StateDep) -> UserProfile:
    now = datetime.now(UTC)
    profile = UserProfile(
        id=payload.id or str(uuid.uuid4()),
        nickname=payload.nickname,
        preferred_language=payload.preferred_language,
        preferred_voice=payload.preferred_voice,
        preferred_llm_model=payload.preferred_llm_model,
        conversation_style=payload.conversation_style,
        units=payload.units,
        timezone=payload.timezone,
        extra=payload.extra,
        created_at=now,
        updated_at=now,
    )
    return state.require_assistant().profiles.create(profile)


@router.patch("/{user_id}", response_model=UserProfile)
def update_user(user_id: str, payload: UpdateUserProfileRequest, state: StateDep) -> UserProfile:
    profiles = state.require_assistant().profiles
    current = profiles.get(user_id)
    updates = {k: v for k, v in payload.model_dump().items() if v is not None}
    updated = current.model_copy(update=updates)
    return profiles.update(updated)


@router.post("/{user_id}/activate")
def activate_user(user_id: str, state: StateDep) -> dict[str, str]:
    state.require_assistant().profiles.set_active(user_id)
    return {"status": "activated"}


@router.delete("/{user_id}")
def delete_user(user_id: str, state: StateDep) -> dict[str, str]:
    state.require_assistant().profiles.delete(user_id)
    return {"status": "deleted"}
