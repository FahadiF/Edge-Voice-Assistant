"""Pipeline benchmark tests — fake engines, no models required.

Confirms the M3 TTFA breakdown (asr_ms/ttft_ms/first_chunk_ms/ttfa_ms) is
computed from the streaming TTS path (ADR-018), not full-sentence synthesis.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np

from eva.asr.base import ASREngine, TranscriptionResult
from eva.audio.frames import Frame
from eva.benchmark.pipeline import PipelineBenchmark
from eva.llm.base import ChatMessage, GenerationParams, LLMEngine
from eva.tts.base import TTSEngine


class _FakeASR(ASREngine):
    def load(self) -> None: ...
    def unload(self) -> None: ...

    def transcribe(self, audio: Frame, language: str | None = None) -> TranscriptionResult:
        return TranscriptionResult(text="what is the weather today")


class _FakeLLM(LLMEngine):
    def load(self) -> None: ...
    def unload(self) -> None: ...

    def stream(
        self,
        messages: list[ChatMessage],
        params: GenerationParams,
        should_abort: Callable[[], bool],
    ) -> Iterator[str]:
        yield from ["It ", "is ", "sunny. ", "Enjoy ", "your ", "day!"]


class _StreamingFakeTTS(TTSEngine):
    """Yields two chunks per call so first_chunk_ms and full-synthesis time differ."""

    def load(self) -> None: ...
    def unload(self) -> None: ...

    def synthesize(self, text: str, *, voice: str, speed: float = 1.0) -> Frame:
        return np.ones(3200, dtype=np.int16)

    def synthesize_stream(
        self, text: str, *, voice: str, speed: float = 1.0
    ) -> Iterator[Frame]:
        yield np.ones(1600, dtype=np.int16)
        yield np.ones(1600, dtype=np.int16)

    def voices(self) -> list[str]:
        return ["bench-voice"]


def _make_bench(tts: TTSEngine | None = None) -> PipelineBenchmark:
    return PipelineBenchmark(
        _FakeASR(),
        _FakeLLM(),
        tts or _StreamingFakeTTS(),
        voice="bench-voice",
        system_prompt="You are a test assistant.",
    )


def test_run_produces_a_complete_report() -> None:
    report = _make_bench().run("What's the weather?")
    assert report.transcript == "what is the weather today"
    assert report.reply == "It is sunny. Enjoy your day!"
    assert report.tokens == 6
    assert report.asr_ms >= 0
    assert report.ttft_ms >= report.asr_ms
    assert report.first_chunk_ms >= 0
    assert report.ttfa_ms >= report.asr_ms
    assert report.tokens_per_s >= 0
    assert report.tts_rtf >= 0


def test_first_chunk_stage_uses_streaming_synthesis() -> None:
    report = _make_bench().run("What's the weather?")
    stage_names = [s.name for s in report.stages]
    assert "TTS (first chunk ready)" in stage_names
    assert "TTS (full first sentence)" in stage_names
    # The default fallback TTSEngine.synthesize_stream() yields exactly one
    # chunk via synthesize(); this fake yields two — confirms run() actually
    # drives synthesize_stream() rather than a single blocking synthesize().
    full_stage = next(s for s in report.stages if s.name == "TTS (full first sentence)")
    chunk_stage = next(s for s in report.stages if s.name == "TTS (first chunk ready)")
    assert full_stage.duration_ms >= chunk_stage.duration_ms


def test_render_includes_ttfa_breakdown() -> None:
    text = _make_bench().run("What's the weather?").render()
    for label in ("ASR", "Time to first token", "Time to first TTS chunk", "Time to first audio"):
        assert label in text
