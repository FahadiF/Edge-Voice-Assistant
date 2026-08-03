"""M8 benchmark reporting: aggregation correctness and the three renderers.

Pure over `Sequence[TurnMetrics]` — no models, no audio, no engine. That is
the point of the design: the report generator cannot tell whether its samples
came from `PipelineBenchmark` or from a live `MetricsCollector`, so both are
testable with plain constructed records.
"""

from __future__ import annotations

import json

import pytest

from eva.benchmark.report import (
    NOT_MEASURED,
    WER_DEFERRAL_NOTE,
    BenchmarkReport,
    StageStats,
    aggregate,
    render,
    summary_line,
    to_html,
    to_json,
    to_markdown,
)
from eva.core.provenance import Environment
from eva.metrics.turn import MetricsCollector, ResourceUsage, TurnMetrics

_ENV = Environment(
    timestamp_utc="2026-08-03T12:00:00+00:00",
    eva_version="0.7.0a1",
    git_commit="abc1234",
    git_dirty=False,
    backend="cuda",
    gpu_name="NVIDIA GeForce RTX 3060 Laptop GPU",
    gpu_vram_mb=6144,
    cuda_device_count=1,
    faster_whisper_version="1.1.0",
    ctranslate2_version="4.8.1",
    python_version="3.12.0",
    platform="Windows 11 (win32)",
)


def _turn(epoch: int, **overrides: object) -> TurnMetrics:
    base: dict[str, object] = {
        "epoch": epoch,
        "asr_ms": 100,
        "ttft_ms": 300,
        "llm_ms": 1000,
        "tokens": 50,
        "tts_first_ms": 150,
        "ttfa_ms": 900,
        "total_ms": 2000,
    }
    base.update(overrides)
    return TurnMetrics(**base)


def _report(
    turns: list[TurnMetrics],
    *,
    label: str = "test run",
    source: str = "benchmark",
    scan_limit: int | None = None,
) -> BenchmarkReport:
    return aggregate(turns, label=label, source=source, environment=_ENV, scan_limit=scan_limit)


def _stage(report: BenchmarkReport, name: str) -> StageStats:
    return next(s for s in report.stages if s.name == name)


class TestAggregation:
    def test_percentiles_match_hand_computed_values(self) -> None:
        turns = [_turn(i, asr_ms=ms) for i, ms in enumerate([100, 200, 300, 400, 500], start=1)]
        asr = _stage(_report(turns), "Speech recognition")
        assert asr.samples == 5
        assert asr.p50 == 300.0
        assert asr.minimum == 100.0
        assert asr.maximum == 500.0
        assert asr.p95 == 480.0  # numpy linear interpolation over 5 points

    def test_cancelled_turns_are_excluded_from_latency_but_still_counted(self) -> None:
        """A barged-in turn stopped early by design; folding its truncated
        timings into a median would understate real latency. It must still
        appear in the totals, or the report misrepresents what happened."""
        turns = [_turn(1, asr_ms=100), _turn(2, asr_ms=9999, cancelled=True), _turn(3, asr_ms=200)]
        report = _report(turns)
        assert report.turn_count == 3
        assert report.completed_count == 2
        assert report.cancelled_count == 1
        asr = _stage(report, "Speech recognition")
        assert asr.samples == 2
        assert asr.maximum == 200.0  # the cancelled 9999 never entered the stats

    def test_llm_speed_derived_from_tokens_and_llm_ms(self) -> None:
        turns = [_turn(1, tokens=100, llm_ms=1000)]  # exactly 100 tok/s
        speed = _stage(_report(turns), "LLM speed")
        assert speed.unit == "tok/s"
        assert speed.p50 == 100.0

    def test_turns_without_llm_time_are_excluded_from_speed(self) -> None:
        turns = [_turn(1, tokens=0, llm_ms=0), _turn(2, tokens=50, llm_ms=1000)]
        speed = _stage(_report(turns), "LLM speed")
        assert speed.samples == 1  # the zero-duration turn would divide by zero


class TestEmptyAndDegenerateInput:
    def test_empty_input_produces_a_valid_report_not_a_crash(self) -> None:
        report = _report([])
        assert report.turn_count == 0
        assert report.completed_count == 0
        assert all(not s.measured for s in report.stages)
        assert all(s.p50 == 0.0 for s in report.stages)  # no divide-by-zero

    def test_all_cancelled_produces_a_valid_report(self) -> None:
        report = _report([_turn(1, cancelled=True)])
        assert report.turn_count == 1
        assert report.completed_count == 0
        assert all(not s.measured for s in report.stages)

    def test_empty_report_renders_in_every_format(self) -> None:
        report = _report([])
        for fmt in ("json", "md", "html"):
            assert render(report, fmt)


class TestNotMeasuredHonesty:
    """A stage that never ran must not be reported as 0 ms — that reads as
    'instant' when it means 'absent'."""

    def test_structurally_absent_stage_is_marked_not_measured(self) -> None:
        # PipelineBenchmark never calls ContextBuilder, so these stay 0.
        report = _report([_turn(1, retrieval_ms=0, context_ms=0)])
        assert not _stage(report, "Memory retrieval").measured
        assert not _stage(report, "Context composition").measured
        assert _stage(report, "Speech recognition").measured  # this one did run

    def test_not_measured_surfaces_in_markdown_and_html(self) -> None:
        report = _report([_turn(1, retrieval_ms=0)])
        assert NOT_MEASURED in to_markdown(report)
        assert NOT_MEASURED in to_html(report)

    def test_a_measured_stage_is_never_labelled_not_measured(self) -> None:
        report = _report([_turn(1, retrieval_ms=12, context_ms=15)])
        assert _stage(report, "Memory retrieval").measured
        assert _stage(report, "Context composition").p50 == 15.0


class TestResources:
    def test_peaks_taken_across_turns_that_carry_a_sample(self) -> None:
        turns = [
            _turn(
                1,
                resources=ResourceUsage(cpu_percent=10.0, ram_used_mb=4000, ram_total_mb=16384),
            ),
            _turn(
                2,
                resources=ResourceUsage(
                    cpu_percent=55.5,
                    ram_used_mb=6000,
                    ram_total_mb=16384,
                    vram_used_mb=3900,
                    vram_total_mb=6144,
                    gpu_percent=88.0,
                ),
            ),
        ]
        r = _report(turns).resources
        assert r is not None
        assert r.samples == 2
        assert r.peak_ram_used_mb == 6000
        assert r.peak_cpu_percent == 55.5
        assert r.peak_vram_used_mb == 3900
        assert r.peak_gpu_percent == 88.0

    def test_no_resource_samples_reports_none_not_zeros(self) -> None:
        assert _report([_turn(1)]).resources is None


class TestScanSaturation:
    """The M1(a) visibility metric — the evidence that decides whether the
    deferred ANN index is ever needed."""

    def test_saturation_computed_against_the_configured_limit(self) -> None:
        turns = [_turn(1, retrieval_scan_count=2000), _turn(2, retrieval_scan_count=500)]
        sat = _report(turns, scan_limit=2000).scan_saturation
        assert sat is not None
        assert sat.scan_limit == 2000
        assert sat.max_scanned == 2000
        assert sat.saturated_turns == 1
        assert sat.saturated_percent == 50.0

    def test_omitted_when_no_limit_is_supplied(self) -> None:
        """A raw scan count without the limit it was bounded by is not
        actionable, so the section is omitted rather than shown bare."""
        assert _report([_turn(1, retrieval_scan_count=2000)]).scan_saturation is None

    def test_omitted_when_retrieval_never_ran(self) -> None:
        assert _report([_turn(1, retrieval_scan_count=0)], scan_limit=2000).scan_saturation is None


class TestRenderers:
    def test_json_round_trips_to_the_same_model_values(self) -> None:
        report = _report([_turn(1, asr_ms=123)])
        data = json.loads(to_json(report))
        assert data["label"] == "test run"
        assert data["environment"]["git_commit"] == "abc1234"
        asr = next(s for s in data["stages"] if s["name"] == "Speech recognition")
        assert asr["p50"] == 123.0

    def test_markdown_contains_every_required_section(self) -> None:
        text = to_markdown(_report([_turn(1)], scan_limit=2000))
        for heading in ("# Benchmark report", "## Latency", "## Resources", "## Environment"):
            assert heading in text, f"missing {heading!r}"

    def test_markdown_reports_a_dirty_checkout(self) -> None:
        """Measurements taken with uncommitted changes are not reproducible
        from the commit alone; the report has to say so."""
        from dataclasses import replace

        report = aggregate(
            [_turn(1)],
            label="x",
            source="benchmark",
            environment=replace(_ENV, git_dirty=True),
        )
        assert "dirty" in to_markdown(report)
        assert "dirty" in to_html(report)

    def test_wer_section_is_present_and_states_why_it_is_empty(self) -> None:
        """M8 lists WER, the fixture corpus does not exist yet, and inventing
        a number would be worse than omitting one."""
        report = _report([_turn(1)])
        assert report.wer is None
        assert WER_DEFERRAL_NOTE in to_markdown(report)
        assert WER_DEFERRAL_NOTE in to_html(report)

    def test_render_rejects_an_unknown_format(self) -> None:
        with pytest.raises(ValueError, match="Unknown report format"):
            render(_report([_turn(1)]), "pdf")

    def test_summary_line_reports_ttfa_when_measured(self) -> None:
        assert "TTFA p50 900 ms" in summary_line(_report([_turn(1, ttfa_ms=900)]))


class TestHtmlIsSelfContained:
    """EVA runs offline by construction. A report that fetched a stylesheet or
    chart library from a CDN would break that the moment it was opened on an
    air-gapped machine — and would violate the §10 offline invariant that
    Batch 6 will make an automated test."""

    def test_no_external_references_of_any_kind(self) -> None:
        html_out = to_html(_report([_turn(1, retrieval_ms=5)], scan_limit=2000))
        for forbidden in ("http://", "https://", "//cdn", "src=", "@import"):
            assert forbidden not in html_out, f"HTML report reaches outside: {forbidden!r}"

    def test_styling_and_charts_are_inline(self) -> None:
        html_out = to_html(_report([_turn(1)]))
        assert "<style>" in html_out  # CSS inline, not <link rel=stylesheet>
        assert "<svg" in html_out  # bars inline, not an image request
        assert "<script" not in html_out  # no JS at all

    def test_report_text_is_html_escaped(self) -> None:
        report = _report([_turn(1)], label="<script>alert(1)</script>")
        html_out = to_html(report)
        assert "<script>alert(1)</script>" not in html_out
        assert "&lt;script&gt;" in html_out


class TestSourceAgnostic:
    """The design claim: one generator, either source, no branching."""

    def test_live_metrics_collector_history_is_accepted_directly(self) -> None:
        collector = MetricsCollector()
        for i in range(3):
            collector.record(_turn(i + 1, asr_ms=100 * (i + 1)))
        report = _report(collector.turns, source="live-session")
        assert report.source == "live-session"
        assert report.turn_count == 3
        assert _stage(report, "Speech recognition").p50 == 200.0

    def test_identical_samples_produce_identical_stats_regardless_of_source(self) -> None:
        turns = [_turn(1), _turn(2)]
        from_bench = _report(turns, source="benchmark")
        from_live = _report(turns, source="live-session")
        assert [s.model_dump() for s in from_bench.stages] == [
            s.model_dump() for s in from_live.stages
        ]
