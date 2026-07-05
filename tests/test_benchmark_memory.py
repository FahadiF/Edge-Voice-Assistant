"""Memory benchmark smoke tests — correctness of the harness itself, not
performance assertions (timing thresholds would be flaky in CI; the actual
measured numbers live in the M4 deliverables report, captured by hand on
real hardware).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from eva.benchmark.memory import run_memory_benchmark
from eva.memory import db
from eva.memory.sqlite_store import SQLiteMemoryStore


@pytest.fixture
def store(tmp_path: Path) -> Iterator[SQLiteMemoryStore]:
    conn = db.connect(tmp_path / "bench.db")
    s = SQLiteMemoryStore(conn)
    yield s
    s.close()


def test_benchmark_produces_a_complete_report(store: SQLiteMemoryStore) -> None:
    report = run_memory_benchmark(store, turn_count=50, turns_per_conversation=10, search_rounds=3)
    assert report.turn_count == 50
    assert report.embedding_dim == 384
    assert report.db_size_bytes > 0
    assert report.search_text_ms_p50 >= 0
    assert report.retrieval_ms_p50 >= 0
    assert report.context_build_ms_p50 >= 0


def test_benchmark_spreads_turns_across_multiple_conversations(
    store: SQLiteMemoryStore,
) -> None:
    run_memory_benchmark(store, turn_count=100, turns_per_conversation=10, search_rounds=1)
    conversations = store.all_conversations(include_archived=True)
    assert len(conversations) == 10  # 100 turns / 10 per conversation


def test_render_includes_all_sections() -> None:
    from eva.benchmark.memory import MemoryBenchmarkReport

    report = MemoryBenchmarkReport(
        turn_count=100,
        embedding_dim=384,
        db_size_bytes=1_048_576,
        search_text_ms_p50=1.0,
        search_text_ms_p95=2.0,
        retrieval_ms_p50=3.0,
        retrieval_ms_p95=4.0,
        context_build_ms_p50=5.0,
        context_build_ms_p95=6.0,
    )
    text = report.render()
    for label in ("Synthetic turns", "Keyword search", "Semantic retrieval", "Context composition"):
        assert label in text
