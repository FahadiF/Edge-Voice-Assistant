"""Settings read/validate/patch/reset — the one implementation the CLI and the
platform API both call (ADR-017 Part 10: no duplicated business logic).
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from eva.config.paths import AppPaths
from eva.config.settings import Settings, save_settings


def get_schema() -> dict[str, Any]:
    """JSON Schema for the settings document — UIs generate forms from this."""
    return Settings.model_json_schema()


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def validate_patch(current: Settings, patch: dict[str, Any]) -> Settings:
    """Apply `patch` (partial, nested dict) over `current` and validate.

    Raises pydantic.ValidationError on invalid values — callers translate that
    into their own error format (HTTP 422, CLI message, …).
    """
    merged = _deep_merge(current.model_dump(mode="json"), patch)
    return Settings.model_validate(merged)


def validate_full(payload: dict[str, Any]) -> Settings:
    """Validate a complete settings document without merging or saving."""
    return Settings.model_validate(payload)


def apply_patch(paths: AppPaths, current: Settings, patch: dict[str, Any]) -> Settings:
    updated = validate_patch(current, patch)
    save_settings(updated, paths.settings_file)
    return updated


def replace_settings(paths: AppPaths, payload: dict[str, Any]) -> Settings:
    updated = validate_full(payload)
    save_settings(updated, paths.settings_file)
    return updated


def reset_settings(paths: AppPaths) -> Settings:
    """Reset to schema defaults (distinct from `eva profiles set`, which
    resets to a hardware-tier model preset)."""
    fresh = Settings()
    save_settings(fresh, paths.settings_file)
    return fresh


def describe_validation_error(exc: ValidationError) -> list[dict[str, Any]]:
    """Compact, JSON-safe error list for API responses and CLI messages."""
    return [{"loc": list(e["loc"]), "message": e["msg"], "type": e["type"]} for e in exc.errors()]
