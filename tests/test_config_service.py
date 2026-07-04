"""Unit tests for the settings service shared by the CLI and the platform API."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from eva.config import service
from eva.config.paths import AppPaths
from eva.config.settings import Settings, load_settings


class TestSchema:
    def test_schema_has_definitions(self) -> None:
        schema = service.get_schema()
        assert "VADSettings" in schema["$defs"]


class TestValidatePatch:
    def test_merges_nested_without_touching_siblings(self) -> None:
        current = Settings()
        updated = service.validate_patch(current, {"conversation": {"temperature": 0.9}})
        assert updated.conversation.temperature == 0.9
        assert updated.conversation.max_tokens == current.conversation.max_tokens
        assert updated.llm == current.llm

    def test_rejects_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            service.validate_patch(Settings(), {"vad": {"threshold": 9.0}})

    def test_rejects_unknown_field(self) -> None:
        with pytest.raises(ValidationError):
            service.validate_patch(Settings(), {"llm": {"bogus": 1}})


class TestApplyAndReset:
    def test_apply_patch_persists(self, app_paths: AppPaths) -> None:
        current = load_settings(app_paths.settings_file)
        updated = service.apply_patch(app_paths, current, {"tts": {"voice": "af_bella"}})
        assert updated.tts.voice == "af_bella"
        reloaded = load_settings(app_paths.settings_file)
        assert reloaded.tts.voice == "af_bella"

    def test_reset_writes_defaults(self, app_paths: AppPaths) -> None:
        service.apply_patch(app_paths, Settings(), {"conversation": {"temperature": 1.9}})
        fresh = service.reset_settings(app_paths)
        assert fresh.conversation.temperature == 0.4
        assert load_settings(app_paths.settings_file).conversation.temperature == 0.4

    def test_replace_settings_validates_and_persists(self, app_paths: AppPaths) -> None:
        payload = Settings().model_dump(mode="json")
        payload["tts"]["speed"] = 1.5
        updated = service.replace_settings(app_paths, payload)
        assert updated.tts.speed == 1.5
        assert load_settings(app_paths.settings_file).tts.speed == 1.5


class TestErrorFormatting:
    def test_describe_validation_error_shape(self) -> None:
        try:
            service.validate_full({"vad": {"threshold": 9.0}})
        except ValidationError as exc:
            errors = service.describe_validation_error(exc)
        else:
            raise AssertionError("expected ValidationError")
        assert errors[0]["loc"] == ["vad", "threshold"]
        assert "type" in errors[0]
