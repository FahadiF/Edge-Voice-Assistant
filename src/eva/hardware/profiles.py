"""Hardware profiles: map detected capability to a *capability tier*.

A profile answers one question — "what class of hardware is this?" — and nothing
more. Which models a tier gets is owned solely by `eva.hardware.presets`
(`ModelPreset.tiers`), so there is exactly one place to change a tier's model
selection.

They used to carry their own `llm_model`/`asr_model`/`asr_device`/`tts_engine`
copies, which drifted from the presets: `gpu-12gb` advertised
`distil-large-v3` on `cuda` while applying a preset gave `small` on `auto`.
Deriving the display from the preset removes that failure mode by construction.

Thresholds are deliberately conservative: recommending a model that does not fit
is a much worse experience than recommending one tier lower.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from eva.hardware.detect import HardwareReport


class HardwareProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    display_name: str
    description: str
    min_vram_mb: int  # 0 = CPU profile


PROFILES: dict[str, HardwareProfile] = {
    p.id: p
    for p in (
        HardwareProfile(
            id="cpu-only",
            display_name="CPU only",
            description="No usable GPU; small models, everything on CPU.",
            min_vram_mb=0,
        ),
        HardwareProfile(
            id="gpu-6gb",
            display_name="GPU · 6 GB VRAM",
            description="Mid-range GPU (e.g. RTX 3060 Laptop): 4B LLM on GPU, ASR int8.",
            min_vram_mb=5_500,
        ),
        HardwareProfile(
            id="gpu-12gb",
            display_name="GPU · 12 GB+ VRAM",
            description="High-end GPU: 7-9B class LLM, larger ASR model.",
            min_vram_mb=11_000,
        ),
    )
}


def recommend_profile(report: HardwareReport) -> HardwareProfile:
    """Pick the highest profile whose VRAM floor the best GPU clears."""
    gpu = report.best_gpu
    vram = gpu.vram_total_mb if gpu is not None else 0
    eligible = [p for p in PROFILES.values() if p.min_vram_mb <= vram]
    return max(eligible, key=lambda p: p.min_vram_mb)
