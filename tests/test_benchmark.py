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

    def transcribe(
        self, audio: Frame, language: str | None = None, *, prompt: str | None = None
    ) -> TranscriptionResult:
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

    def synthesize(
        self, text: str, *, voice: str, speed: float = 1.0, language: str | None = None
    ) -> Frame:
        return np.ones(3200, dtype=np.int16)

    def synthesize_stream(
        self, text: str, *, voice: str, speed: float = 1.0, language: str | None = None
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


class TestAsTurnMetrics:
    """M8: the benchmark re-projects onto `TurnMetrics` so the report
    generator consumes one record type regardless of source. A pure mapping
    of already-measured numbers — no second collection path."""

    def test_every_overlapping_field_is_carried_over_unchanged(self) -> None:
        report = _make_bench().run("What's the weather?")
        turn = report.as_turn_metrics(epoch=7)
        assert turn.epoch == 7
        assert turn.asr_ms == report.asr_ms
        assert turn.ttft_ms == report.ttft_ms
        assert turn.llm_ms == report.llm_ms
        assert turn.tokens == report.tokens
        assert turn.tts_first_ms == report.first_chunk_ms
        assert turn.ttfa_ms == report.ttfa_ms

    def test_llm_ms_is_exposed_and_agrees_with_its_stage_timing(self) -> None:
        """`llm_ms` was always measured — it was just only reachable as a
        `StageTiming` string. Promoting it to a field must not change it."""
        report = _make_bench().run("What's the weather?")
        stage = next(s for s in report.stages if s.name == "LLM (full generation)")
        assert report.llm_ms == stage.duration_ms

    def test_stages_the_benchmark_never_runs_stay_at_their_defaults(self) -> None:
        """`PipelineBenchmark` composes messages directly and never calls
        `ContextBuilder`, so retrieval/context genuinely did not happen here.
        They must read 0 (which the report renders as "not measured"), never
        an invented value."""
        turn = _make_bench().run("What's the weather?").as_turn_metrics()
        assert turn.retrieval_ms == 0
        assert turn.context_ms == 0
        assert turn.retrieval_score_top1 is None
        assert turn.retrieval_scan_count == 0
        assert turn.resources is None

    def test_projection_feeds_the_aggregator_end_to_end(self) -> None:
        """The whole M8 chain in one test: benchmark run → TurnMetrics →
        aggregate → render, with no live engine anywhere."""
        from eva.benchmark.report import aggregate, render
        from eva.core.provenance import capture_environment

        bench = _make_bench()
        turns = [bench.run("What's the weather?").as_turn_metrics(epoch=i) for i in range(1, 4)]
        report = aggregate(
            turns, label="test", source="benchmark", environment=capture_environment("cpu")
        )
        assert report.turn_count == 3
        assert report.completed_count == 3
        assert report.cancelled_count == 0
        # Fakes return instantly, so the sub-millisecond stages honestly read
        # "not measured" — that is the ambiguity rule doing its job, not a
        # gap. What matters here is that every format renders the chain.
        for fmt in ("json", "md", "html"):
            assert render(report, fmt)
