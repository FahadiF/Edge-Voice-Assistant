"""Memory subsystem data models (ADR-019).

`MemoryTurn` is one speaker's utterance — one row per turn, not a paired
(user, assistant) record — matching the "speaker/text/timestamp" shape asked
for and matching how chat message lists are already represented
(`eva.llm.base.ChatMessage`). `ConversationTurn` (paired) is kept separately
in `eva.conversation.history` for the pre-M4 `/conversation/*` API contract,
which this milestone does not change the shape of.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Speaker = Literal["user", "assistant"]


class MemoryConversation(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    started_at: datetime
    title: str = ""
    language: str = "en"
    archived: bool = False


class MemoryTurn(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: int | None = None  # None before insert
    conversation_id: str
    created_at: datetime
    speaker: Speaker
    text: str
    language: str = "en"
    metadata: dict[str, Any] = Field(default_factory=dict)
    pinned: bool = False
    favorite: bool = False
    deleted: bool = False


class MemorySummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: int | None = None
    conversation_id: str
    turn_range_start: int
    turn_range_end: int
    text: str
    created_at: datetime
    model_id: str


class UserProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    nickname: str = ""
    preferred_language: str | None = None
    preferred_voice: str | None = None
    preferred_llm_model: str | None = None
    conversation_style: str = ""
    units: Literal["metric", "imperial"] = "metric"
    timezone: str = "UTC"
    extra: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
    active: bool = False


class MemorySearchResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    turn: MemoryTurn
    score: float
    match_reason: Literal["semantic", "keyword", "pinned"] = "semantic"


class MemoryStats(BaseModel):
    model_config = ConfigDict(frozen=True)

    conversation_count: int
    turn_count: int
    embedded_turn_count: int
    summary_count: int
    db_size_bytes: int
    fts_enabled: bool
