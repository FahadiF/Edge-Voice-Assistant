"""NumpyMemoryRetriever tests (ADR-020): retrieval accuracy on synthetic
vectors with known nearest neighbors, using a real SQLiteMemoryStore (no
mocking the store — only the vectors are synthetic, not the persistence).
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from eva.memory import db
from eva.memory.models import MemoryTurn
from eva.memory.retriever import NumpyMemoryRetriever
from eva.memory.sqlite_store import SQLiteMemoryStore


def _vec(values: list[float]) -> bytes:
    arr = np.array(values, dtype=np.float32)
    arr = arr / np.linalg.norm(arr)  # embeddings are always L2-normalized in practice
    return arr.tobytes()


@pytest.fixture
def store(tmp_path: Path) -> Iterator[SQLiteMemoryStore]:
    conn = db.connect(tmp_path / "memory.db")
    s = SQLiteMemoryStore(conn)
    yield s
    s.close()


class TestBasicRetrieval:
    def test_exact_match_ranks_first(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation()
        t_a = store.add_turn(conv.id, "user", "about cats")
        t_b = store.add_turn(conv.id, "user", "about dogs")
        t_c = store.add_turn(conv.id, "user", "about cars")
        store.store_embedding(t_a.id, "m", _vec([1, 0, 0]), dim=3)
        store.store_embedding(t_b.id, "m", _vec([0, 1, 0]), dim=3)
        store.store_embedding(t_c.id, "m", _vec([0, 0, 1]), dim=3)

        retriever = NumpyMemoryRetriever(store, recency_half_life_days=0)  # disable decay
        results = retriever.retrieve(_vec([1, 0, 0]), top_k=3)

        assert results[0].turn.id == t_a.id
        assert results[0].score == pytest.approx(1.0, abs=1e-5)
        # b and c are both orthogonal to the query — genuinely tied at 0, not
        # a ranking bug.
        assert results[1].score == pytest.approx(0.0, abs=1e-5)
        assert results[2].score == pytest.approx(0.0, abs=1e-5)

    def test_top_k_limits_results(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation()
        for i in range(10):
            turn = store.add_turn(conv.id, "user", f"turn {i}")
            store.store_embedding(turn.id, "m", _vec([1, i * 0.01, 0]), dim=3)
        retriever = NumpyMemoryRetriever(store, recency_half_life_days=0)
        results = retriever.retrieve(_vec([1, 0, 0]), top_k=3)
        assert len(results) == 3

    def test_ranking_matches_cosine_similarity_order(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation()
        # Vectors at increasing angular distance from the query [1, 0].
        specs = [("near", [0.99, 0.14]), ("mid", [0.5, 0.87]), ("far", [-1, 0.05])]
        ids = {}
        for name, vec in specs:
            turn = store.add_turn(conv.id, "user", name)
            store.store_embedding(turn.id, "m", _vec(vec), dim=2)
            ids[name] = turn.id
        retriever = NumpyMemoryRetriever(store, recency_half_life_days=0)
        results = retriever.retrieve(_vec([1, 0]), top_k=3)
        assert [r.turn.id for r in results] == [ids["near"], ids["mid"], ids["far"]]

    def test_empty_store_returns_empty(self, store: SQLiteMemoryStore) -> None:
        retriever = NumpyMemoryRetriever(store)
        assert retriever.retrieve(_vec([1, 0, 0]), top_k=5) == []

    def test_conversation_scoping(self, store: SQLiteMemoryStore) -> None:
        conv_a = store.start_conversation()
        conv_b = store.start_conversation()
        t_a = store.add_turn(conv_a.id, "user", "in a")
        t_b = store.add_turn(conv_b.id, "user", "in b")
        store.store_embedding(t_a.id, "m", _vec([1, 0]), dim=2)
        store.store_embedding(t_b.id, "m", _vec([1, 0]), dim=2)
        retriever = NumpyMemoryRetriever(store, recency_half_life_days=0)
        results = retriever.retrieve(_vec([1, 0]), top_k=10, conversation_id=conv_a.id)
        assert [r.turn.id for r in results] == [t_a.id]


class TestScoringAdjustments:
    def test_pinned_boost_can_change_ranking(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation()
        t_close = store.add_turn(conv.id, "user", "closer match")
        t_pinned = store.add_turn(conv.id, "user", "pinned but slightly further")
        store.store_embedding(t_close.id, "m", _vec([1.0, 0.05]), dim=2)
        store.store_embedding(t_pinned.id, "m", _vec([1.0, 0.2]), dim=2)
        store.pin(t_pinned.id)

        retriever = NumpyMemoryRetriever(
            store, recency_half_life_days=0, pinned_boost=0.5, favorite_boost=0.0
        )
        results = retriever.retrieve(_vec([1.0, 0.0]), top_k=2)
        assert results[0].turn.id == t_pinned.id  # boost overcomes the small similarity gap

    def test_favorite_boost_applied(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation()
        turn = store.add_turn(conv.id, "user", "favorited")
        store.store_embedding(turn.id, "m", _vec([1, 0]), dim=2)
        store.favorite(turn.id)
        retriever_boosted = NumpyMemoryRetriever(
            store, recency_half_life_days=0, pinned_boost=0.0, favorite_boost=0.4
        )
        retriever_plain = NumpyMemoryRetriever(
            store, recency_half_life_days=0, pinned_boost=0.0, favorite_boost=0.0
        )
        boosted = retriever_boosted.retrieve(_vec([1, 0]), top_k=1)[0]
        plain = retriever_plain.retrieve(_vec([1, 0]), top_k=1)[0]
        assert boosted.score > plain.score

    def test_recency_decay_prefers_newer_memory(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation()
        old_turn = store.add_turn(conv.id, "user", "old but identical vector")
        new_turn = store.add_turn(conv.id, "user", "new and identical vector")
        store.store_embedding(old_turn.id, "m", _vec([1, 0]), dim=2)
        store.store_embedding(new_turn.id, "m", _vec([1, 0]), dim=2)

        # Backdate the old turn well past the retriever's half-life.
        old_timestamp = (datetime.now(UTC) - timedelta(days=60)).isoformat()
        store._conn.execute(
            "UPDATE turns SET created_at = ? WHERE id = ?", (old_timestamp, old_turn.id)
        )
        store._conn.commit()

        retriever = NumpyMemoryRetriever(store, recency_half_life_days=14.0)
        results = retriever.retrieve(_vec([1, 0]), top_k=2)
        assert results[0].turn.id == new_turn.id
        assert results[0].score > results[1].score

    def test_zero_half_life_disables_decay(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation()
        turn = store.add_turn(conv.id, "user", "identical vector")
        store.store_embedding(turn.id, "m", _vec([1, 0]), dim=2)
        old_timestamp = (datetime.now(UTC) - timedelta(days=3650)).isoformat()
        store._conn.execute(
            "UPDATE turns SET created_at = ? WHERE id = ?", (old_timestamp, turn.id)
        )
        store._conn.commit()
        retriever = NumpyMemoryRetriever(store, recency_half_life_days=0)
        result = retriever.retrieve(_vec([1, 0]), top_k=1)[0]
        assert result.score == pytest.approx(1.0, abs=1e-5)  # pure cosine similarity, no decay


class TestScanLimit:
    def test_scan_limit_bounds_candidates_scored(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation()
        for i in range(10):
            turn = store.add_turn(conv.id, "user", f"turn {i}")
            store.store_embedding(turn.id, "m", _vec([1, i * 0.001]), dim=2)

        retriever = NumpyMemoryRetriever(store, recency_half_life_days=0, scan_limit=3)
        results = retriever.retrieve(_vec([1, 0]), top_k=10)
        # Only the 3 most-recently-created embeddings were ever candidates.
        assert len(results) == 3

    def test_scan_limit_keeps_most_recent_embeddings(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation()
        oldest = store.add_turn(conv.id, "user", "oldest")
        store.store_embedding(oldest.id, "m", _vec([1, 0]), dim=2)
        newest = store.add_turn(conv.id, "user", "newest")
        store.store_embedding(newest.id, "m", _vec([1, 0]), dim=2)

        retriever = NumpyMemoryRetriever(store, scan_limit=1)
        results = retriever.retrieve(_vec([1, 0]), top_k=10)
        assert [r.turn.id for r in results] == [newest.id]


class TestRobustness:
    def test_mismatched_dimension_embeddings_are_skipped(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation()
        stale = store.add_turn(conv.id, "user", "embedded with an old, different-size model")
        current = store.add_turn(conv.id, "user", "embedded with the current model")
        store.store_embedding(stale.id, "old-model", b"\x00" * (128 * 4), dim=128)
        store.store_embedding(current.id, "new-model", _vec([1, 0, 0]), dim=3)

        retriever = NumpyMemoryRetriever(store, recency_half_life_days=0)
        results = retriever.retrieve(_vec([1, 0, 0]), top_k=10)
        assert [r.turn.id for r in results] == [current.id]  # stale-dim entry silently excluded

    def test_forgotten_turn_between_lookup_and_fetch_is_skipped(
        self, store: SQLiteMemoryStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The FK constraint (`embeddings.turn_id REFERENCES turns(id)`) makes
        a dangling embedding row impossible in SQLite itself — but a
        cross-connection race is still real: another connection's `forget()`
        could remove a turn between this retriever's `embeddings_for()` read
        and its `get_turns()` bulk-fetch. Simulated here by making
        `get_turns` omit the id, exactly what that race would look like from
        the retriever's point of view."""
        conv = store.start_conversation()
        turn = store.add_turn(conv.id, "user", "vanishes mid-retrieval")
        store.store_embedding(turn.id, "m", _vec([1, 0]), dim=2)

        real_get_turns = store.get_turns

        def flaky_get_turns(turn_ids: list[int]) -> list[MemoryTurn]:
            return [t for t in real_get_turns(turn_ids) if t.id != turn.id]

        monkeypatch.setattr(store, "get_turns", flaky_get_turns)

        retriever = NumpyMemoryRetriever(store)
        results = retriever.retrieve(_vec([1, 0]), top_k=10)
        assert results == []  # must not raise, must not crash on the gap
