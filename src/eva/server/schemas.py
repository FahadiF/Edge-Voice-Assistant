"""Request/response models not already defined by the domain modules.

Domain models (Settings, RuntimeSnapshot, ModelInfo, ConversationTurn, …) are
returned directly where they already exist — no shadow copies. These are the
handful that exist only for the API surface itself.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from eva.conversation.history import ConversationTurn

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

    kind: Literal["llm", "asr", "tts", "vad"] | None = Field(
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


# Generic passthrough type for PATCH bodies (arbitrary nested JSON merged
# against the current settings document).
JsonObject = dict[str, Any]
