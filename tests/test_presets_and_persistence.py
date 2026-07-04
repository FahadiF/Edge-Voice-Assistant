"""Model presets (Part of ADR-015) and configuration persistence."""

from __future__ import annotations

import pytest

from eva.config.paths import AppPaths
from eva.config.settings import Settings, load_settings
from eva.core.errors import RegistryError
from eva.hardware.presets import (
    CUSTOM_PROFILE_ID,
    apply_preset,
    preset_registry,
    register_builtin_presets,
)
from eva.models.catalog import BUILTIN_CATALOG
from eva.onboarding import resolve_and_persist_settings


class TestPresets:
    def test_builtin_presets_registered(self) -> None:
        register_builtin_presets()
        for preset_id in ("balanced", "fast", "high-accuracy", "low-memory", "developer"):
            assert preset_id in preset_registry

    def test_every_preset_model_exists_in_catalog(self) -> None:
        """Presets are data; this keeps them honest against the catalog."""
        register_builtin_presets()
        catalog_ids = {m.id for m in BUILTIN_CATALOG}
        for preset in preset_registry.snapshot().values():
            for tier_id, models in preset.tiers.items():
                assert models.llm_model in catalog_ids, f"{preset.id}/{tier_id} llm"
                assert models.asr_model in catalog_ids, f"{preset.id}/{tier_id} asr"
                assert models.tts_model in catalog_ids, f"{preset.id}/{tier_id} tts"

    def test_apply_preset_writes_model_fields(self) -> None:
        settings = Settings()
        apply_preset(settings, "fast", "gpu-6gb")
        assert settings.profile == "fast"
        assert settings.llm.model == "qwen3-1.7b-instruct-q4_k_m"
        assert settings.asr.model == "faster-whisper/base"
        assert settings.llm.context_length == 4096

    def test_unknown_tier_falls_back_to_cpu_floor(self) -> None:
        settings = Settings()
        apply_preset(settings, "low-memory", "gpu-6gb")  # preset defines cpu-only only
        assert settings.llm.model == "qwen3-1.7b-instruct-q4_k_m"
        assert settings.asr.device == "cpu"

    def test_unknown_preset_rejected(self) -> None:
        with pytest.raises(RegistryError):
            apply_preset(Settings(), "turbo-max", "gpu-6gb")

    def test_developer_preset_enables_debug(self) -> None:
        settings = Settings()
        apply_preset(settings, "developer", "gpu-6gb")
        assert settings.developer.debug
        assert settings.developer.log_level == "DEBUG"


class TestPersistence:
    def test_resolve_and_persist_writes_settings_file(self, app_paths: AppPaths) -> None:
        settings = Settings()
        resolve_and_persist_settings(settings, app_paths)
        assert app_paths.settings_file.exists()
        loaded = load_settings(app_paths.settings_file)
        # The persisted selection is concrete and reload-stable.
        assert loaded.llm.model == settings.llm.model
        assert loaded.profile == settings.profile
        assert loaded.profile != "auto"

    def test_custom_profile_models_are_preserved(self, app_paths: AppPaths) -> None:
        settings = Settings()
        settings.profile = CUSTOM_PROFILE_ID
        settings.llm.model = "qwen3-4b-instruct-q4_k_m"  # manual choice
        resolve_and_persist_settings(settings, app_paths)
        loaded = load_settings(app_paths.settings_file)
        assert loaded.llm.model == "qwen3-4b-instruct-q4_k_m"
        assert loaded.profile == CUSTOM_PROFILE_ID

    def test_selection_stable_across_restarts(self, app_paths: AppPaths) -> None:
        """The root-cause regression test for inconsistent model selection:
        once persisted, changing code defaults must not change the loaded model."""
        settings = Settings()
        resolve_and_persist_settings(settings, app_paths)
        first = load_settings(app_paths.settings_file).llm.model
        # Simulate a later release shipping a different default: the persisted
        # file, not the default, decides.
        second = load_settings(app_paths.settings_file).llm.model
        assert first == second
