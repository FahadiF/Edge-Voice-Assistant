"""ASR engine registry: id → factory(settings, paths)."""

from __future__ import annotations

from collections.abc import Callable

from eva.asr.base import ASREngine
from eva.config.paths import AppPaths
from eva.config.settings import Settings
from eva.core.registry import Registry

ASRFactory = Callable[[Settings, AppPaths], ASREngine]

asr_registry: Registry[ASRFactory] = Registry("asr-engine")


def _make_faster_whisper(settings: Settings, paths: AppPaths) -> ASREngine:
    from eva.asr.fasterwhisper import FasterWhisperASR

    # Model ids look like "faster-whisper/small" — the engine consumes the size.
    model = settings.asr.model.split("/", maxsplit=1)[-1]
    return FasterWhisperASR(
        model,
        device=settings.asr.device,
        compute_type=settings.asr.compute_type,
        download_root=paths.models_dir / "asr",
    )


def register_builtins() -> None:
    if "faster-whisper" not in asr_registry:
        asr_registry.register("faster-whisper", _make_faster_whisper)


def create_asr(settings: Settings, paths: AppPaths) -> ASREngine:
    register_builtins()
    return asr_registry.get(settings.asr.engine)(settings, paths)
