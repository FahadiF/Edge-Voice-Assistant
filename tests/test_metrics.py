"""MetricsCollector: bounded in-memory history with lifetime counters.

EVA is designed for long-running sessions, so per-turn samples must not
accumulate without limit — but lifetime counts (used by diagnostics and the
CLI summary) must stay accurate past the window.
"""

from __future__ import annotations

from eva.metrics.turn import _MAX_TURN_SAMPLES, MetricsCollector, TurnMetrics


def _turn(epoch: int, *, cancelled: bool = False) -> TurnMetrics:
    return TurnMetrics(
        epoch=epoch,
        asr_ms=100,
        ttft_ms=200,
        tts_first_ms=150,
        ttfa_ms=0 if cancelled else 900,
        llm_ms=1000,
        tokens=50,
        cancelled=cancelled,
    )


def test_samples_are_bounded() -> None:
    collector = MetricsCollector()
    for i in range(_MAX_TURN_SAMPLES + 500):
        collector.record(_turn(i))
    # The window is capped; memory does not grow without limit.
    assert len(collector.turns) == _MAX_TURN_SAMPLES
    # The most recent turn is retained (diagnostics reads turns[-1]).
    assert collector.turns[-1].epoch == _MAX_TURN_SAMPLES + 499


def test_lifetime_counters_survive_the_window() -> None:
    collector = MetricsCollector()
    total = _MAX_TURN_SAMPLES + 200
    cancelled = 0
    for i in range(total):
        is_cancelled = i % 5 == 0
        cancelled += is_cancelled
        collector.record(_turn(i, cancelled=is_cancelled))
    assert collector.total_recorded == total
    assert collector.non_cancelled_count == total - cancelled
    # The summary reports the lifetime total, not the window size.
    assert f"Turns: {total}" in collector.summary()


def test_summary_medians_use_recent_completed_turns() -> None:
    collector = MetricsCollector()
    collector.record(_turn(1))
    summary = collector.summary()
    assert "Time to first audio:" in summary
    assert "900 ms" in summary  # ttfa median


def test_no_completed_turns_message() -> None:
    collector = MetricsCollector()
    collector.record(_turn(1, cancelled=True))
    assert collector.summary() == "No completed turns."
    assert collector.total_recorded == 1
    assert collector.non_cancelled_count == 0


class TestHistoricalRetrievalAttribution:
    """Batch 7 (M4): retrieval/context timing must be reconstructable per
    turn from the bounded history, not just readable as a single latest
    value — the roadmap's own acceptance criterion for this batch."""

    def test_reconstructs_retrieval_and_context_timing_for_every_turn(self) -> None:
        collector = MetricsCollector()
        for i in range(3):
            collector.record(
                TurnMetrics(
                    epoch=i,
                    retrieval_ms=10 + i,
                    context_ms=20 + i,
                    retrieval_score_top1=0.1 * i,
                    retrieval_scan_count=100 + i,
                )
            )
        turns = collector.turns
        assert [t.retrieval_ms for t in turns] == [10, 11, 12]
        assert [t.context_ms for t in turns] == [20, 21, 22]
        assert [round(t.retrieval_score_top1, 2) for t in turns] == [0.0, 0.1, 0.2]  # type: ignore[arg-type]
        assert [t.retrieval_scan_count for t in turns] == [100, 101, 102]

    def test_new_fields_default_safely_for_a_turn_that_never_built_context(self) -> None:
        """An early-return `TurnMetrics` (stale-after-ASR, empty transcript)
        must never report a retrieval that did not happen."""
        m = TurnMetrics(epoch=1)
        assert m.retrieval_ms == 0
        assert m.context_ms == 0
        assert m.retrieval_score_top1 is None
        assert m.retrieval_scan_count == 0
        assert m.resources is None
