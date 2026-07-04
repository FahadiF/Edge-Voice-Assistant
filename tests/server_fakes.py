"""Fake engine components shared by server tests needing a "running" engine
without real models or audio hardware — same fakes used in test_orchestrator.py,
combined into an Assistant-shaped object for `ServerState.start_engine()`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np

from eva.asr.base import ASREngine, TranscriptionResult
from eva.audio.frames import Frame
from eva.config.paths import AppPaths
from eva.config.settings import Settings
from eva.conversation.orchestrator import Orchestrator
from eva.core.events import EventBus
from eva.engine import Assistant
from eva.llm.base import ChatMessage, GenerationParams, LLMEngine
from eva.tts.base import TTSEngine


class FakeASR(ASREngine):
    device = "cpu"

    def __init__(self, text: str = "hello") -> None:
        self.text = text

    def load(self) -> None: ...
    def unload(self) -> None: ...

    def transcribe(self, audio: Frame, language: str | None = None) -> TranscriptionResult:
        return TranscriptionResult(text=self.text)


class FakeLLM(LLMEngine):
    device = "cpu"

    def load(self) -> None: ...
    def unload(self) -> None: ...

    def stream(
        self,
        messages: list[ChatMessage],
        params: GenerationParams,
        should_abort: Callable[[], bool],
    ) -> Iterator[str]:
        for token in ["Hello", " there."]:
            if should_abort():
                return
            yield token


class FakeTTS(TTSEngine):
    device = "cpu"

    def load(self) -> None: ...
    def unload(self) -> None: ...

    def synthesize(self, text: str, *, voice: str, speed: float = 1.0) -> Frame:
        return np.ones(1600, dtype=np.int16)

    def voices(self) -> list[str]:
        return ["test-voice"]


class _FakePipeline:
    level_dbfs = -40.0


class _FakeRing:
    dropped = 0

    def __len__(self) -> int:
        return 0


class _FakePlayback:
    def queued_seconds(self) -> float:
        return 0.0


class FakeAudioSystem:
    """Satisfies both the orchestrator's AudioOutput protocol and the
    lifecycle/diagnostics surface `Assistant.audio` needs."""

    def __init__(self) -> None:
        self.spoken: list[Frame] = []
        self.pipeline = _FakePipeline()
        self.capture_ring = _FakeRing()
        self.playback = _FakePlayback()
        self._speaking = False

    # AudioOutput protocol (used by the orchestrator)
    def say(self, pcm: Frame) -> None:
        self.spoken.append(pcm)

    def finish_utterance(self) -> None:
        pass

    def stop_speaking(self) -> None:
        self._speaking = False

    @property
    def is_speaking(self) -> bool:
        return self._speaking

    # Lifecycle (used by Assistant.start_audio/stop)
    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


def build_fake_assistant(
    settings: Settings, _paths: AppPaths, bus: EventBus | None = None
) -> Assistant:
    """Drop-in replacement for `eva.engine.build_assistant` in server tests."""
    bus = bus or EventBus()
    audio = FakeAudioSystem()
    asr, llm, tts = FakeASR(), FakeLLM(), FakeTTS()
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
