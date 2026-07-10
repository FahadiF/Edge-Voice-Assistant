"""Request/response models not already defined by the domain modules.

Domain models (Settings, RuntimeSnapshot, ModelInfo, ConversationTurn, …) are
returned directly where they already exist — no shadow copies. These are the
handful that exist only for the API surface itself.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from eva.conversation.history import ConversationTurn
from eva.llm.base import ChatMessage

CONVERSATION_EXPORT_VERSION = 1


class HealthResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: Literal["ok"] = "ok"
    version: str


class ValidationErrorDetail(BaseModel):
    model_config = ConfigDict(frozen=True)

    loc: list[str | int]
    message: str
    type: str


class SettingsValidationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    valid: bool
    errors: list[ValidationErrorDetail] = Field(default_factory=list)


class ModelActivateRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["llm", "asr", "tts", "vad", "embedding"] | None = Field(
        None, description="Override the catalog kind if a model spans roles; usually omitted"
    )


class DownloadStartedResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_id: str
    status: Literal["started", "already_running", "not_applicable"]


class EngineStatusResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    running: bool
    state: str


class ReadinessResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    ready: bool
    problems: list[str]


class InterruptResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    interrupted: bool


class SayRequest(BaseModel):
    """Typed message from the web UI composer (M5.3)."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(min_length=1, max_length=8000, description="Message to send as a turn")


class RenameConversationRequest(BaseModel):
    """Edit a conversation's title (M5.4 — titles are auto-generated after
    the first exchange and editable afterwards)."""

    model_config = ConfigDict(frozen=True)

    title: str = Field(min_length=1, max_length=120, description="New conversation title")


class PluginStatusResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    version: str
    description: str
    enabled: bool
    healthy: bool
    error: str | None
    contributes: tuple[str, ...]
    permissions: tuple[str, ...]


class ConversationExport(BaseModel):
    model_config = ConfigDict(frozen=True)

    version: int = CONVERSATION_EXPORT_VERSION
    exported_at: datetime
    profile: str
    language: str
    turns: list[ConversationTurn]


class ConversationImportRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    turns: list[ConversationTurn]


class HardwareSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    tier: str
    tier_name: str
    cpu: str
    gpu: str | None
    vram_mb: int
    ram_mb: int


# ──────────────────────── Memory (M4, ADR-019/ADR-021) ────────────────────────


class MemorySearchRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    query: str
    limit: Annotated[int, Field(ge=1, le=200)] = 20
    conversation_id: str | None = None


class MergeConversationsRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_id: str
    target_id: str


class RetrievedMemoryTraceResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    turn_id: int
    score: float
    text_preview: str


class ContextTraceResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    persona_id: str
    profile_id: str | None
    language_code: str
    retrieved_memories: list[RetrievedMemoryTraceResponse]
    summary_included: bool
    summary_text_preview: str | None
    recent_turn_count: int
    trimmed_sections: list[str]


class ContextPreviewResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    messages: list[ChatMessage]
    trace: ContextTraceResponse


# ──────────────────────── User profiles (M4, ADR-022) ────────────────────────


class CreateUserProfileRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str | None = Field(None, description="Auto-generated (uuid4) if omitted")
    nickname: str = ""
    preferred_language: str | None = None
    preferred_voice: str | None = None
    preferred_llm_model: str | None = None
    conversation_style: str = ""
    units: Literal["metric", "imperial"] = "metric"
    timezone: str = "UTC"
    extra: dict[str, Any] = Field(default_factory=dict)


class UpdateUserProfileRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    nickname: str | None = None
    preferred_language: str | None = None
    preferred_voice: str | None = None
    preferred_llm_model: str | None = None
    conversation_style: str | None = None
    units: Literal["metric", "imperial"] | None = None
    timezone: str | None = None
    extra: dict[str, Any] | None = None


# ──────────────────────── Voices (M4, ADR-022) ────────────────────────


class VoicePreviewRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    phrase: str = "Hello, this is a preview of my voice."


# Generic passthrough type for PATCH bodies (arbitrary nested JSON merged
# against the current settings document).
JsonObject = dict[str, Any]
