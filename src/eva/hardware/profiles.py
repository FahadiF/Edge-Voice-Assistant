"""Hardware profiles: map detected capability to recommended component defaults.

Profiles are **data**, not behavior — the model manager and settings UI read them
to pre-select components, and users can override any field afterwards. Thresholds
are deliberately conservative: recommending a model that does not fit is a much
worse experience than recommending one tier lower.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from eva.hardware.detect import HardwareReport


class HardwareProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    display_name: str
    description: str
    llm_model: str
    asr_model: str
    asr_device: str
    tts_engine: str
    min_vram_mb: int  # 0 = CPU profile


PROFILES: dict[str, HardwareProfile] = {
    p.id: p
    for p in (
        HardwareProfile(
            id="cpu-only",
            display_name="CPU only",
            description="No usable GPU; small models, everything on CPU.",
            llm_model="qwen3-1.7b-instruct-q4_k_m",
            asr_model="faster-whisper/base",
            asr_device="cpu",
            tts_engine="kokoro",
            min_vram_mb=0,
        ),
        HardwareProfile(
            id="gpu-6gb",
            display_name="GPU · 6 GB VRAM",
            description="Mid-range GPU (e.g. RTX 3060 Laptop): 4B LLM on GPU, ASR int8.",
            llm_model="qwen3-4b-instruct-q4_k_m",
            asr_model="faster-whisper/small",
            asr_device="auto",
            tts_engine="kokoro",
            min_vram_mb=5_500,
        ),
        HardwareProfile(
            id="gpu-12gb",
            display_name="GPU · 12 GB+ VRAM",
            description="High-end GPU: 7-9B class LLM, larger ASR model.",
            llm_model="qwen3-8b-instruct-q4_k_m",
            asr_model="faster-whisper/distil-large-v3",
            asr_device="cuda",
            tts_engine="kokoro",
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
