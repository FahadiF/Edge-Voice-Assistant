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
