from __future__ import annotations

import json
from pathlib import Path

import pytest

from eva.config.settings import Settings, load_settings, save_settings
from eva.core.errors import ConfigError


def test_defaults_are_valid() -> None:
    s = Settings()
    assert s.llm.engine == "llamacpp"
    assert s.vad.threshold == 0.5
    assert s.server.host == "127.0.0.1"


def test_missing_file_yields_defaults(tmp_path: Path) -> None:
    s = load_settings(tmp_path / "does-not-exist.json")
    assert s == Settings()


def test_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    s = Settings()
    s.tts.voice = "af_bella"
    s.conversation.temperature = 0.7
    save_settings(s, path)
    loaded = load_settings(path)
    assert loaded == s


def test_partial_file_merges_with_defaults(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"llm": {"model": "custom-model"}}), encoding="utf-8")
    s = load_settings(path)
    assert s.llm.model == "custom-model"
    assert s.llm.engine == "llamacpp"  # untouched default
    assert s.tts.voice == "af_heart"


def test_unknown_keys_rejected(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"llm": {"modle": "typo"}}), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_settings(path)


def test_out_of_range_values_rejected(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"conversation": {"temperature": 9.0}}), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_settings(path)


def test_malformed_json_raises_config_error(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_settings(path)


def test_save_creates_parent_dirs(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dir" / "settings.json"
    save_settings(Settings(), path)
    assert path.exists()


def test_validate_assignment_guards_runtime_changes() -> None:
    s = Settings()
    with pytest.raises(ValueError):
        s.vad.threshold = 5.0  # out of [0, 1]


def test_every_field_has_ui_documentation() -> None:
    """The settings schema is the UI's source of truth (ADR-009): every leaf
    field must carry a description so settings pages can render themselves."""
    from pydantic import BaseModel

    missing: list[str] = []

    def walk(model: type[BaseModel], prefix: str) -> None:
        for name, field in model.model_fields.items():
            annotation = field.annotation
            if isinstance(annotation, type) and issubclass(annotation, BaseModel):
                walk(annotation, f"{prefix}{name}.")
            elif not field.description:
                missing.append(f"{prefix}{name}")

    walk(Settings, "")
    assert not missing, f"settings fields missing descriptions: {missing}"


def test_schema_exports_bounds_for_ui() -> None:
    schema = Settings.model_json_schema()
    vad = schema["$defs"]["VADSettings"]["properties"]["threshold"]
    assert vad["minimum"] == 0.0
    assert vad["maximum"] == 1.0
    assert "description" in vad
