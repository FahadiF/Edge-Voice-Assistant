"""TTS engine registry: id → factory(settings, model_files)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from eva.config.settings import Settings
from eva.core.errors import ModelError
from eva.core.registry import Registry
from eva.tts.base import TTSEngine

# Factories receive the resolved file paths of the active TTS model.
TTSFactory = Callable[[Settings, dict[str, Path]], TTSEngine]

tts_registry: Registry[TTSFactory] = Registry("tts-engine")


def _make_kokoro(_settings: Settings, files: dict[str, Path]) -> TTSEngine:
    from eva.tts.kokoro import KokoroTTS

    try:
        return KokoroTTS(files["model"], files["voices"])
    except KeyError as exc:
        raise ModelError(f"Kokoro requires a '{exc.args[0]}' model file") from exc


def register_builtins() -> None:
    if "kokoro" not in tts_registry:
        tts_registry.register("kokoro", _make_kokoro)


def create_tts(settings: Settings, files: dict[str, Path]) -> TTSEngine:
    register_builtins()
    return tts_registry.get(settings.tts.engine)(settings, files)
