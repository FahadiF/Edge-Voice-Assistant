"""Settings API: get/update/validate/reset + JSON Schema for UI form generation."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import ValidationError

from eva.config import service as settings_service
from eva.config.settings import Settings
from eva.server.deps import StateDep
from eva.server.schemas import JsonObject, SettingsValidationResult, ValidationErrorDetail

router = APIRouter(prefix="/settings", tags=["settings"])


@router.get("", response_model=Settings)
def get_settings(state: StateDep) -> Settings:
    return state.reload_settings()


@router.get("/schema")
def get_settings_schema() -> dict[str, Any]:
    """JSON Schema for the settings document — the source future UIs render
    their settings pages from (ADR-009)."""
    return settings_service.get_schema()


@router.put("", response_model=Settings)
def replace_settings(payload: JsonObject, state: StateDep) -> Settings:
    updated = settings_service.replace_settings(state.paths, payload)
    state.settings = updated
    return updated


@router.patch("", response_model=Settings)
def patch_settings(payload: JsonObject, state: StateDep) -> Settings:
    updated = settings_service.apply_patch(state.paths, state.settings, payload)
    state.settings = updated
    return updated


@router.post("/validate", response_model=SettingsValidationResult)
def validate_settings(payload: JsonObject) -> SettingsValidationResult:
    """Validate a complete settings document without saving it."""
    try:
        settings_service.validate_full(payload)
    except ValidationError as exc:
        errors = [
            ValidationErrorDetail(**e) for e in settings_service.describe_validation_error(exc)
        ]
        return SettingsValidationResult(valid=False, errors=errors)
    return SettingsValidationResult(valid=True)


@router.post("/reset", response_model=Settings)
def reset_settings(state: StateDep) -> Settings:
    fresh = settings_service.reset_settings(state.paths)
    state.settings = fresh
    return fresh
