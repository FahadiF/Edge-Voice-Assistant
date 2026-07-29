"""Memory recall quality: supersession, provenance rendering, retrieval hygiene.

When a later statement corrects an earlier one, both remain relevant and both
are recalled. Retrieval order alone cannot express which is current, so the
memory block states who said each thing and how long ago, newest first.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from eva.config.settings import Settings
from eva.conversation.context_builder import ContextBuilder, _age_phrase
from eva.memory import db
from eva.memory.base import MemoryRetriever
from eva.memory.models import MemorySearchResult, MemoryTurn
from eva.memory.retriever import NumpyMemoryRetriever
from eva.memory.sqlite_store import SQLiteMemoryStore


def _vec(values: list[float]) -> bytes:
    arr = np.array(values, dtype=np.float32)
    return (arr / np.linalg.norm(arr)).tobytes()


@pytest.fixture
def store(tmp_path: Path) -> Iterator[SQLiteMemoryStore]:
    conn = db.connect(tmp_path / "memory.db")
    s = SQLiteMemoryStore(conn)
    yield s
    s.close()


def _turn_at(
    store: SQLiteMemoryStore, conv: str, text: str, days_ago: float, *, speaker: str = "user"
) -> MemoryTurn:
    """Add a turn with a controlled age (add_turn always stamps 'now')."""
    turn = store.add_turn(conv, speaker, text)
    when = datetime.now(UTC) - timedelta(days=days_ago)
    store._conn.execute("UPDATE turns SET created_at = ? WHERE id = ?", (when.isoformat(), turn.id))
    store._conn.commit()
    return turn.model_copy(update={"created_at": when})


def _at(store: SQLiteMemoryStore, conv: str, speaker: str, text: str, days_ago: float) -> int:
    return int(_turn_at(store, conv, text, days_ago, speaker=speaker).id or 0)


class _StubEmbedding:
    def embed(self, text: str) -> Any:
        return np.zeros(2, dtype=np.float32)


class _OrderedRetriever(MemoryRetriever):
    """Returns a fixed list in a fixed order.

    Yields the superseded statement first, since similarity ranking may place
    it there. The block must still present the newest first — that is why the
    renderer orders by timestamp rather than by score.
    """

    def __init__(self, turns: list[MemoryTurn]) -> None:
        self._turns = turns

    def retrieve(
        self, query_vector: bytes, *, top_k: int, conversation_id: str | None = None
    ) -> list[MemorySearchResult]:
        return [
            MemorySearchResult(turn=t, score=1.0 - i * 0.1, match_reason="semantic")
            for i, t in enumerate(self._turns)
        ][:top_k]


def _semantic_builder(store: SQLiteMemoryStore, turns: list[MemoryTurn]) -> ContextBuilder:
    return ContextBuilder(
        Settings(),
        store,
        retriever=_OrderedRetriever(turns),
        embedding_provider=_StubEmbedding(),  # type: ignore[arg-type]
    )


def _memory_block(builder: ContextBuilder, conv: str, query: str) -> str:
    system = builder.build(conv, query).messages[0].content
    for section in system.split("\n\n"):
        if section.startswith("You remember"):
            return section
    return ""


class TestSupersession:
    """A correction must be distinguishable from what it corrects."""

    def test_newer_statement_appears_before_the_stale_one(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation().id
        stale = _turn_at(store, conv, "I live in Vaasa.", 30)
        current = _turn_at(store, conv, "I moved to Helsinki.", 1)
        block = _memory_block(_semantic_builder(store, [stale, current]), conv, "Where do I live?")
        assert block.index("Helsinki") < block.index("Vaasa"), (
            "the current fact must come first; position is how recency is communicated"
        )

    def test_each_memory_carries_when_it_was_said(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation().id
        stale = _turn_at(store, conv, "I live in Vaasa.", 30)
        current = _turn_at(store, conv, "I moved to Helsinki.", 1)
        block = _memory_block(_semantic_builder(store, [stale, current]), conv, "Where do I live?")
        assert "You said yesterday: I moved to Helsinki." in block
        assert "You said 30 days ago: I live in Vaasa." in block

    def test_several_corrections_put_the_latest_first(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation().id
        a = _turn_at(store, conv, "My dog's name is Milo.", 20)
        b = _turn_at(store, conv, "Actually my dog's name is Max.", 10)
        c = _turn_at(store, conv, "Sorry, the dog is called Rex.", 1)
        block = _memory_block(_semantic_builder(store, [a, b, c]), conv, "dog name")
        assert block.index("Rex") < block.index("Max") < block.index("Milo")

    def test_an_old_uncontested_fact_is_still_recalled(self, store: SQLiteMemoryStore) -> None:
        """Ordering by recency must not mean forgetting what is merely old."""
        conv = store.start_conversation().id
        old = _turn_at(store, conv, "My laptop has an RTX 3060.", 200)
        recent = _turn_at(store, conv, "I moved to Helsinki.", 1)
        block = _memory_block(_semantic_builder(store, [old, recent]), conv, "laptop GPU")
        assert "RTX 3060" in block
        assert "You said 6 months ago: My laptop has an RTX 3060." in block


class TestProvenanceRendering:
    def test_user_and_assistant_are_distinguishable(self, store: SQLiteMemoryStore) -> None:
        """The speaker filter keeps assistant turns out of the semantic path,
        but the renderer must still attribute correctly for any caller that
        does surface one (the keyword path, an imported store)."""
        conv = store.start_conversation().id
        builder = ContextBuilder(Settings(), store)
        now = datetime.now(UTC)
        user_turn = store.add_turn(conv, "user", "I like Python.")
        assistant_turn = store.add_turn(conv, "assistant", "Noted.")
        assert builder._render_memory(user_turn, now).startswith("You said")
        assert builder._render_memory(assistant_turn, now).startswith("You told the user")

    def test_a_naive_timestamp_renders_without_an_age(self, store: SQLiteMemoryStore) -> None:
        """Imported or pre-migration rows can carry a naive `created_at`.
        Rendering must degrade to an undated statement, never guess an age
        and never raise."""
        conv = store.start_conversation().id
        turn = store.add_turn(conv, "user", "I like Python.")
        naive = turn.model_copy(update={"created_at": datetime(2020, 1, 1)})  # no tzinfo
        rendered = ContextBuilder(Settings(), store)._render_memory(naive, datetime.now(UTC))
        assert rendered == "You said: I like Python."

    @pytest.mark.parametrize(
        ("days_ago", "expected"),
        [
            (0.0, "just now"),
            (0.01, "14 minutes ago"),
            (0.5, "12 hours ago"),
            (1.2, "yesterday"),
            (5.0, "5 days ago"),
            (90.0, "3 months ago"),
            (400.0, "a year ago"),
        ],
    )
    def test_age_phrases(self, days_ago: float, expected: str) -> None:
        now = datetime.now(UTC)
        assert _age_phrase(now - timedelta(days=days_ago), now) == expected

    def test_no_internal_identifiers_leak_into_the_prompt(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation().id
        _at(store, conv, "user", "I like Python.", days_ago=2)
        block = _memory_block(ContextBuilder(Settings(), store), conv, "language")
        for leaked in ("turn_id", "score=", "cosine", "similarity", "conversation_id", "embedding"):
            assert leaked not in block


class TestRetrievalHygiene:
    """Filtering that keeps the memory block small enough to be useful."""

    def test_similarity_floor_drops_weak_matches(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation().id
        near = store.add_turn(conv, "user", "near")
        far = store.add_turn(conv, "user", "far")
        store.store_embedding(near.id, "m", _vec([1.0, 0.02]), dim=2)
        store.store_embedding(far.id, "m", _vec([0.2, 1.0]), dim=2)  # cosine ~0.2
        retriever = NumpyMemoryRetriever(store, recency_half_life_days=0, min_similarity=0.30)
        results = retriever.retrieve(_vec([1.0, 0.0]), top_k=5)
        assert [r.turn.id for r in results] == [near.id]

    def test_the_floor_is_applied_to_raw_cosine_not_the_aged_score(
        self, store: SQLiteMemoryStore
    ) -> None:
        """The blended score scales relevance by age, so a single cutoff
        applied to it would reject old-but-relevant memories. The floor must
        therefore act on raw similarity."""
        conv = store.start_conversation().id
        old_relevant = _at(store, conv, "user", "old but exactly on topic", days_ago=365)
        store.store_embedding(old_relevant, "m", _vec([1.0, 0.0]), dim=2)
        retriever = NumpyMemoryRetriever(store, recency_half_life_days=14.0, min_similarity=0.30)
        results = retriever.retrieve(_vec([1.0, 0.0]), top_k=5)
        assert [r.turn.id for r in results] == [old_relevant]
        assert results[0].score < 0.30, "aged score is below the floor, yet it survived"

    def test_assistant_turns_are_excluded_from_personal_memory(
        self, store: SQLiteMemoryStore
    ) -> None:
        """The assistant's own replies are not an authoritative source for
        personal memory; only user-authored turns are recalled."""
        conv = store.start_conversation().id
        user_turn = store.add_turn(conv, "user", "My dog is Milo.")
        ack = store.add_turn(conv, "assistant", "Got it, I'll remember that.")
        for turn in (user_turn, ack):
            store.store_embedding(turn.id, "m", _vec([1.0, 0.0]), dim=2)
        retriever = NumpyMemoryRetriever(store, recency_half_life_days=0, speakers=("user",))
        results = retriever.retrieve(_vec([1.0, 0.0]), top_k=5)
        assert [r.turn.id for r in results] == [user_turn.id]

    def test_duplicate_texts_collapse_to_one(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation().id
        first = store.add_turn(conv, "user", "I prefer short answers.")
        again = store.add_turn(conv, "user", "  I prefer   SHORT answers.  ")
        other = store.add_turn(conv, "user", "Something else entirely.")
        for turn, vec in ((first, [1.0, 0.0]), (again, [0.99, 0.1]), (other, [0.9, 0.3])):
            store.store_embedding(turn.id, "m", _vec(vec), dim=2)
        retriever = NumpyMemoryRetriever(store, recency_half_life_days=0)
        texts = [r.turn.text for r in retriever.retrieve(_vec([1.0, 0.0]), top_k=5)]
        assert len(texts) == 2, f"whitespace/case duplicate survived: {texts}"

    def test_an_unanswerable_question_receives_nothing(self, store: SQLiteMemoryStore) -> None:
        """A question whose answer was never stated must recall nothing,
        rather than handing the model weak matches to ignore."""
        conv = store.start_conversation().id
        turn = store.add_turn(conv, "user", "unrelated content")
        store.store_embedding(turn.id, "m", _vec([0.1, 1.0]), dim=2)
        retriever = NumpyMemoryRetriever(store, recency_half_life_days=0, min_similarity=0.30)
        assert retriever.retrieve(_vec([1.0, 0.0]), top_k=5) == []

    def test_a_strong_match_still_survives_the_floor(self, store: SQLiteMemoryStore) -> None:
        """The floor must reject weak matches without costing recall."""
        conv = store.start_conversation().id
        turn = store.add_turn(conv, "user", "My laptop has an RTX 3060.")
        store.store_embedding(turn.id, "m", _vec([1.0, 0.05]), dim=2)
        retriever = NumpyMemoryRetriever(store, recency_half_life_days=0, min_similarity=0.30)
        assert len(retriever.retrieve(_vec([1.0, 0.0]), top_k=5)) == 1

    def test_the_adapter_filters_nothing_by_default(self, store: SQLiteMemoryStore) -> None:
        """Policy comes from settings; the port stays neutral so a caller that
        constructs it plainly gets pure similarity ranking."""
        conv = store.start_conversation().id
        turn = store.add_turn(conv, "user", "opposite")
        store.store_embedding(turn.id, "m", _vec([-1.0, 0.0]), dim=2)
        retriever = NumpyMemoryRetriever(store, recency_half_life_days=0)
        assert len(retriever.retrieve(_vec([1.0, 0.0]), top_k=5)) == 1


class TestDegradedModes:
    def test_no_embeddings_still_recalls_by_keyword(self, store: SQLiteMemoryStore) -> None:
        """No embedding model installed: ContextBuilder falls back to FTS
        rather than losing memory entirely (M5.4)."""
        conv = store.start_conversation().id
        store.add_turn(conv, "user", "My robotics project is called Atlas.")
        store.add_turn(conv, "assistant", "Understood.")
        block = _memory_block(ContextBuilder(Settings(), store), conv, "robotics project")
        assert "Atlas" in block

    def test_the_keyword_path_applies_the_same_speaker_policy(
        self, store: SQLiteMemoryStore
    ) -> None:
        """Otherwise behaviour would change depending on whether the optional
        embedding model happens to be installed."""
        conv = store.start_conversation().id
        store.add_turn(conv, "user", "My robotics project is called Atlas.")
        store.add_turn(conv, "assistant", "Your robotics project sounds interesting.")
        block = _memory_block(ContextBuilder(Settings(), store), conv, "robotics project")
        assert "You told the user" not in block

    def test_no_memories_produces_no_block_at_all(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation().id
        assert _memory_block(ContextBuilder(Settings(), store), conv, "anything") == ""
