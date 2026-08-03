"""Benchmark suite (M2: end-to-end pipeline benchmark; reporting added in M8).

`report` aggregates and presents; it never collects. Both sample sources —
`PipelineBenchmark` rounds and a live `MetricsCollector` history — hand over
`Sequence[TurnMetrics]`, so one report generator serves both.
"""

from eva.benchmark.pipeline import PipelineBenchmark
from eva.benchmark.report import BenchmarkReport, aggregate, render

__all__ = ["BenchmarkReport", "PipelineBenchmark", "aggregate", "render"]
