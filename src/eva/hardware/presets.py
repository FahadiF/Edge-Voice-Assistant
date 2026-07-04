"""Model presets: goal-oriented model combinations per hardware tier.

Two layers (ARCHITECTURE §7): hardware detection yields a *capability tier*
(`cpu-only` / `gpu-6gb` / `gpu-12gb`), and a *preset* (Balanced, Fast, …) maps
each tier to a concrete model combination. Presets are registry entries
(ADR-010): users pick one from the UI/CLI, may then override any individual
model — which flips the settings profile to ``custom`` — and the resolved
choice is persisted so it never changes between restarts (ADR-015).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from eva.config.settings import Settings
from eva.core.registry import Registry


class TierModels(BaseModel):
    """Concrete model selection for one capability tier."""

    model_config = ConfigDict(frozen=True)

    llm_model: str
    asr_model: str
    asr_device: Literal["auto", "cuda", "cpu"] = "auto"
    tts_engine: str = "kokoro"
    tts_model: str = "kokoro-82m-v1.0"
    vad_engine: str = "silero"
    context_length: int = 8192


class ModelPreset(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    display_name: str
    description: str
    tiers: dict[str, TierModels]  # capability tier id → models
    debug_logging: bool = False

    def for_tier(self, tier_id: str) -> TierModels:
        """Selection for `tier_id`, falling back to the cpu-only floor."""
        return self.tiers.get(tier_id) or self.tiers["cpu-only"]


CUSTOM_PROFILE_ID = "custom"

_CPU_LIGHT = TierModels(
    llm_model="qwen3-1.7b-instruct-q4_k_m",
    asr_model="faster-whisper/base",
    asr_device="cpu",
    context_length=4096,
)

preset_registry: Registry[ModelPreset] = Registry("model-preset")


def register_builtin_presets() -> None:
    presets = (
        ModelPreset(
            id="balanced",
            display_name="Balanced",
            description="Best overall quality/latency trade-off for the detected hardware.",
            tiers={
                "cpu-only": _CPU_LIGHT,
                "gpu-6gb": TierModels(
                    llm_model="qwen3.5-4b-instruct-q4_k_m",
                    asr_model="faster-whisper/small",
                ),
                "gpu-12gb": TierModels(
                    llm_model="qwen3.5-9b-instruct-q4_k_m",
                    asr_model="faster-whisper/small",
                ),
            },
        ),
        ModelPreset(
            id="fast",
            display_name="Fast",
            description="Lowest latency: smaller models on every tier.",
            tiers={
                "cpu-only": _CPU_LIGHT,
                "gpu-6gb": TierModels(
                    llm_model="qwen3-1.7b-instruct-q4_k_m",
                    asr_model="faster-whisper/base",
                    context_length=4096,
                ),
                "gpu-12gb": TierModels(
                    llm_model="qwen3.5-4b-instruct-q4_k_m",
                    asr_model="faster-whisper/base",
                ),
            },
        ),
        ModelPreset(
            id="high-accuracy",
            display_name="High Accuracy",
            description="Best quality the hardware can hold; higher latency.",
            tiers={
                "cpu-only": TierModels(
                    llm_model="qwen3.5-4b-instruct-q4_k_m",
                    asr_model="faster-whisper/small",
                    asr_device="cpu",
                ),
                "gpu-6gb": TierModels(
                    llm_model="qwen3.5-4b-instruct-q4_k_m",
                    asr_model="faster-whisper/small",
                    context_length=16384,
                ),
                "gpu-12gb": TierModels(
                    llm_model="qwen3.5-9b-instruct-q4_k_m",
                    asr_model="faster-whisper/distil-large-v3",
                    context_length=16384,
                ),
            },
        ),
        ModelPreset(
            id="low-memory",
            display_name="Low Memory",
            description="Smallest footprint for constrained machines.",
            tiers={"cpu-only": _CPU_LIGHT},
        ),
        ModelPreset(
            id="developer",
            display_name="Developer",
            description="Balanced models with verbose diagnostics enabled.",
            debug_logging=True,
            tiers={
                "cpu-only": _CPU_LIGHT,
                "gpu-6gb": TierModels(
                    llm_model="qwen3.5-4b-instruct-q4_k_m",
                    asr_model="faster-whisper/small",
                ),
                "gpu-12gb": TierModels(
                    llm_model="qwen3.5-9b-instruct-q4_k_m",
                    asr_model="faster-whisper/small",
                ),
            },
        ),
    )
    for preset in presets:
        if preset.id not in preset_registry:
            preset_registry.register(preset.id, preset)


def apply_preset(settings: Settings, preset_id: str, tier_id: str) -> None:
    """Write a preset's model selection into `settings` (caller persists).

    Raises RegistryError for an unknown preset id. `custom` is not a preset —
    it means "leave the manually chosen models alone".
    """
    register_builtin_presets()
    preset = preset_registry.get(preset_id)
    models = preset.for_tier(tier_id)
    settings.profile = preset_id
    settings.llm.model = models.llm_model
    settings.llm.context_length = models.context_length
    settings.asr.model = models.asr_model
    settings.asr.device = models.asr_device
    settings.tts.engine = models.tts_engine
    settings.tts.model = models.tts_model
    settings.vad.engine = models.vad_engine
    if preset.debug_logging:
        settings.developer.debug = True
        settings.developer.log_level = "DEBUG"
