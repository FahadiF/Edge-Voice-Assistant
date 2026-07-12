"""Engine assembly: resolve models, build engines, wire the orchestrator.

The single composition root used by the CLI today and the server in M5.
Model ids come from settings, files from the ModelManager, engines from their
registries — nothing here names a concrete implementation.

Load order is deterministic (ADR-015): the LLM loads first and owns the GPU
(architecture §5), then ASR takes what remains, then TTS (CPU), then the
embedding model (M4, ADR-020 — also CPU, optional). This keeps device
placement — and therefore latency behavior — stable across restarts instead
of depending on which engine grabbed VRAM first.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from eva.asr.base import ASREngine
from eva.asr.registry import create_asr
from eva.audio.system import AudioSystem
from eva.config.paths import AppPaths
from eva.config.settings import Settings
from eva.conversation.context_builder import ContextBuilder
from eva.conversation.orchestrator import Orchestrator
from eva.core.errors import ModelNotInstalledError
from eva.core.events import ComponentLoadFinished, ComponentLoadStarted, EventBus
from eva.embedding.base import EmbeddingProvider
from eva.embedding.registry import create_embedding_provider
from eva.llm.base import LLMEngine
from eva.llm.registry import create_llm
from eva.memory.base import MemoryStore, UserProfileStore
from eva.memory.registry import create_stores
from eva.memory.retriever import NumpyMemoryRetriever
from eva.models.manager import ModelManager
from eva.tts.base import TTSEngine
from eva.tts.registry import create_tts
from eva.tts.voices import register_voices_for_engine

logger = logging.getLogger(__name__)

_COMPONENT_LABELS = {
    "llm": "Loading language model…",
    "asr": "Loading speech recognition…",
    "tts": "Loading speech synthesis…",
    "embedding": "Loading memory embeddings…",
}


@dataclass
class Assistant:
    settings: Settings
    bus: EventBus
    audio: AudioSystem
    orchestrator: Orchestrator
    asr: ASREngine
    llm: LLMEngine
    tts: TTSEngine
    memory: MemoryStore
    profiles: UserProfileStore
    embedding: EmbeddingProvider | None
    """None when the embedding model isn't installed or is disabled in
    settings — memory search still works via keyword/FTS (ADR-020)."""

    _audio_started: bool = False
    """Whether audio capture was actually opened (False when the microphone
    permission is off — ADR-025 regroup); stop() must not stop what never
    started."""

    def start_audio(self) -> None:
        """Open the microphone/speaker pipeline — unless the user revoked
        the microphone permission (ADR-025 regroup, M5.4), in which case the
        assistant runs typed-chat-only: the web composer still works and TTS
        still speaks (playback-only stream, M5.6 — no input device is ever
        opened), but no audio is captured. Before M5.6 mic-off skipped audio
        entirely, so the playback queue never drained and every typed turn
        wedged in the "speaking" state."""
        if not self.settings.permissions.devices.microphone:
            logger.info("Microphone permission is off — audio capture disabled (typed chat only)")
            self.audio.start_playback_only()
            self._audio_started = True
            return
        self.audio.start()
        self._audio_started = True

    def stop(self) -> None:
        self.orchestrator.request_shutdown()
        if self._audio_started:
            self.audio.stop()
        self.memory.close()

    def preload(self) -> None:
        """Load all models up front so the first turn has no load latency.

        GPU components stay strictly ordered — LLM first (owns the GPU,
        ADR-015 §5), then ASR (leftover VRAM or CPU fallback) — while the
        CPU-resident components (TTS, embedding) load concurrently on worker
        threads (M5.5, ADR-026): total startup ≈ LLM+ASR time instead of the
        sum of everything. Each component publishes ComponentLoadStarted/
        Finished on the bus (a no-op when the bus is unbound, e.g. `eva run`
        before its loop starts) so the UI renders per-component progress.
        With `tts.lazy_load` the TTS engine is skipped here and loads on
        first use (the Kokoro adapter self-loads; the voices API registers
        voices on demand).
        """
        logger.info("Loading models (first run may download weights)...")
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="preload-cpu") as pool:
            cpu_futures = []
            if not self.settings.tts.lazy_load:
                cpu_futures.append(pool.submit(self._load_component, "tts", self._load_tts))
            if self.embedding is not None:
                embedding = self.embedding
                cpu_futures.append(pool.submit(self._load_component, "embedding", embedding.load))
            self._load_component("llm", self.llm.load)
            self._load_component("asr", self.asr.load)
            for future in cpu_futures:
                future.result()  # re-raise the first CPU-side load failure

    def _load_tts(self) -> None:
        self.tts.load()
        register_voices_for_engine(self.settings.tts.engine, self.tts)

    def _load_component(self, name: str, load: Callable[[], None]) -> None:
        """Load one component, bracketing it with progress events; a failure
        is reported before it propagates so the UI never shows a bar stuck
        at 'loading'."""
        self.bus.publish_threadsafe(
            ComponentLoadStarted(component=name, label=_COMPONENT_LABELS.get(name, name))
        )
        start = time.perf_counter()
        try:
            load()
        except Exception as exc:
            self.bus.publish_threadsafe(
                ComponentLoadFinished(
                    component=name,
                    ms=int((time.perf_counter() - start) * 1000),
                    error=str(exc),
                )
            )
            raise
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        self.bus.publish_threadsafe(ComponentLoadFinished(component=name, ms=elapsed_ms))
        logger.info("Component '%s' loaded in %d ms", name, elapsed_ms)

    def active_models(self) -> dict[str, str]:
        """kind → model id actually configured (for banners and diagnostics)."""
        models = {
            "llm": self.settings.llm.model,
            "asr": self.settings.asr.model,
            "tts": self.settings.tts.model,
            "vad": self.settings.vad.engine,
        }
        if self.embedding is not None:
            models["embedding"] = self.settings.memory.embedding_model
        return models


def required_models(settings: Settings) -> list[str]:
    """Model ids the current settings need (for preflight checks).

    The embedding model is intentionally excluded — semantic memory search
    is optional; keyword/FTS search works without it (ADR-020).
    """
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
    memory, profiles = create_stores(settings, paths)

    embedding: EmbeddingProvider | None = None
    retriever: NumpyMemoryRetriever | None = None
    if settings.memory.embedding_enabled and manager.is_installed(settings.memory.embedding_model):
        embedding = create_embedding_provider(
            settings, manager.files_for(settings.memory.embedding_model)
        )
        retriever = NumpyMemoryRetriever(
            memory,
            recency_half_life_days=settings.memory.recency_half_life_days,
            pinned_boost=settings.memory.pinned_boost,
            favorite_boost=settings.memory.favorite_boost,
            scan_limit=settings.memory.retrieval_scan_limit,
        )

    context_builder = ContextBuilder(
        settings,
        memory,
        retriever=retriever,
        embedding_provider=embedding,
        profile_store=profiles,
    )

    orchestrator: Orchestrator | None = None

    def on_audio_event(event: object) -> None:
        assert orchestrator is not None
        orchestrator.feed_audio_event(event)  # type: ignore[arg-type]

    audio = AudioSystem(settings, on_audio_event)
    orchestrator = Orchestrator(
        settings,
        bus,
        audio,
        asr,
        llm,
        tts,
        memory,
        context_builder=context_builder,
        embedding=embedding,
    )
    return Assistant(
        settings=settings,
        bus=bus,
        audio=audio,
        orchestrator=orchestrator,
        asr=asr,
        llm=llm,
        tts=tts,
        memory=memory,
        profiles=profiles,
        embedding=embedding,
    )


__all__ = ["Assistant", "ModelNotInstalledError", "build_assistant", "required_models"]
