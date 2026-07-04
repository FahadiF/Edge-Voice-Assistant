"""Engine assembly: resolve models, build engines, wire the orchestrator.

The single composition root used by the CLI today and the server in M5.
Model ids come from settings, files from the ModelManager, engines from their
registries — nothing here names a concrete implementation.

Load order is deterministic (ADR-015): the LLM loads first and owns the GPU
(architecture §5), then ASR takes what remains, then TTS (CPU). This keeps
device placement — and therefore latency behavior — stable across restarts
instead of depending on which engine grabbed VRAM first.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from eva.asr.base import ASREngine
from eva.asr.registry import create_asr
from eva.audio.system import AudioSystem
from eva.config.paths import AppPaths
from eva.config.settings import Settings
from eva.conversation.orchestrator import Orchestrator
from eva.core.errors import ModelNotInstalledError
from eva.core.events import EventBus
from eva.llm.base import LLMEngine
from eva.llm.registry import create_llm
from eva.models.manager import ModelManager
from eva.tts.base import TTSEngine
from eva.tts.registry import create_tts

logger = logging.getLogger(__name__)


@dataclass
class Assistant:
    settings: Settings
    bus: EventBus
    audio: AudioSystem
    orchestrator: Orchestrator
    asr: ASREngine
    llm: LLMEngine
    tts: TTSEngine

    def start_audio(self) -> None:
        self.audio.start()

    def stop(self) -> None:
        self.orchestrator.request_shutdown()
        self.audio.stop()

    def preload(self) -> None:
        """Load all models up front so the first turn has no load latency.

        Order matters: LLM first (owns the GPU), then ASR (uses leftover VRAM
        or falls back to CPU — visibly), then TTS (CPU). See ADR-015.
        """
        logger.info("Loading models (first run may download weights)...")
        self.llm.load()
        self.asr.load()
        self.tts.load()

    def active_models(self) -> dict[str, str]:
        """kind → model id actually configured (for banners and diagnostics)."""
        return {
            "llm": self.settings.llm.model,
            "asr": self.settings.asr.model,
            "tts": self.settings.tts.model,
            "vad": self.settings.vad.engine,
        }


def required_models(settings: Settings) -> list[str]:
    """Model ids the current settings need (for preflight checks)."""
    return [settings.llm.model, settings.tts.model]


def build_assistant(settings: Settings, paths: AppPaths, bus: EventBus | None = None) -> Assistant:
    """Build a fully wired (but not yet started) assistant.

    Raises ModelNotInstalledError with an actionable message when required
    model files are missing — callers surface it, they don't half-start.
    """
    bus = bus or EventBus()
    manager = ModelManager(paths)

    llm_path = manager.files_for(settings.llm.model)["model"]
    llm = create_llm(settings, llm_path)
    tts = create_tts(settings, manager.files_for(settings.tts.model))
    asr = create_asr(settings, paths)  # engine-managed weights (downloads on first load)

    orchestrator: Orchestrator | None = None

    def on_audio_event(event: object) -> None:
        assert orchestrator is not None
        orchestrator.feed_audio_event(event)  # type: ignore[arg-type]

    audio = AudioSystem(settings, on_audio_event)
    orchestrator = Orchestrator(settings, bus, audio, asr, llm, tts)
    return Assistant(
        settings=settings,
        bus=bus,
        audio=audio,
        orchestrator=orchestrator,
        asr=asr,
        llm=llm,
        tts=tts,
    )


__all__ = ["Assistant", "ModelNotInstalledError", "build_assistant", "required_models"]
