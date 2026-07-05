"""Context Builder tests (ADR-021): deterministic ordering, budget trimming,
trace correctness. No LLM or real embedding model involved — pure
composition logic over a real SQLiteMemoryStore plus fakes for retrieval.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from eva.config.settings import Settings
from eva.conversation.context_builder import ContextBuilder
from eva.embedding.base import EmbeddingProvider
from eva.llm.base import ChatMessage
from eva.memory import db
from eva.memory.base import MemoryRetriever, UserProfileStore
from eva.memory.models import MemorySearchResult, MemorySummary, MemoryTurn, UserProfile
from eva.memory.sqlite_store import SQLiteMemoryStore


@pytest.fixture
def store(tmp_path: Path) -> Iterator[SQLiteMemoryStore]:
    conn = db.connect(tmp_path / "memory.db")
    s = SQLiteMemoryStore(conn)
    yield s
    s.close()


class _FixedRetriever(MemoryRetriever):
    def __init__(self, results: list[MemorySearchResult]) -> None:
        self._results = results
        self.last_top_k: int | None = None
        self.last_conversation_id: str | None = "not-called"

    def retrieve(
        self, query_vector: bytes, *, top_k: int, conversation_id: str | None = None
    ) -> list[MemorySearchResult]:
        self.last_top_k = top_k
        self.last_conversation_id = conversation_id
        return self._results


class _FakeEmbeddingProvider(EmbeddingProvider):
    def load(self) -> None: ...
    def unload(self) -> None: ...

    def embed(self, text: str) -> npt.NDArray[np.float32]:
        return np.zeros(4, dtype=np.float32)


class _FixedProfileStore(UserProfileStore):
    def __init__(self, profile: UserProfile | None) -> None:
        self._profile = profile

    def create(self, profile: UserProfile) -> UserProfile:
        return profile

    def get(self, profile_id: str) -> UserProfile:
        assert self._profile is not None
        return self._profile

    def list(self) -> list[UserProfile]:
        return [self._profile] if self._profile else []

    def update(self, profile: UserProfile) -> UserProfile:
        return profile

    def set_active(self, profile_id: str) -> None:
        pass

    def active(self) -> UserProfile | None:
        return self._profile

    def delete(self, profile_id: str) -> None:
        pass


def _make_result(text: str, score: float, turn_id: int = 1) -> MemorySearchResult:
    turn = MemoryTurn(
        id=turn_id, conversation_id="c1", created_at=datetime.now(UTC), speaker="user", text=text
    )
    return MemorySearchResult(turn=turn, score=score, match_reason="semantic")


class TestDeterministicOrder:
    def test_message_order_system_memory_summary_recent_user(
        self, store: SQLiteMemoryStore
    ) -> None:
        conv = store.start_conversation()
        store.add_turn(conv.id, "user", "earlier question")
        store.add_turn(conv.id, "assistant", "earlier answer")

        store.add_summary(
            MemorySummary(
                conversation_id=conv.id,
                turn_range_start=1,
                turn_range_end=2,
                text="They discussed the weather.",
                created_at=datetime.now(UTC),
                model_id="test",
            )
        )

        retriever = _FixedRetriever([_make_result("relevant fact", 0.9)])
        builder = ContextBuilder(
            Settings(), store, retriever=retriever, embedding_provider=_FakeEmbeddingProvider()
        )
        result = builder.build(conv.id, "new question")

        contents = [m.content for m in result.messages]
        assert "relevant fact" in contents[1]  # memory block right after system prompt
        assert "weather" in contents[2]  # summary next
        assert contents[3] == "earlier question"
        assert contents[4] == "earlier answer"
        assert contents[-1] == "new question"  # current utterance always last

    def test_no_memories_or_summary_still_produces_valid_messages(
        self, store: SQLiteMemoryStore
    ) -> None:
        conv = store.start_conversation()
        builder = ContextBuilder(Settings(), store)
        result = builder.build(conv.id, "hello")
        assert result.messages[0].role == "system"
        assert result.messages[-1] == ChatMessage(role="user", content="hello")


class TestPersonaAndLanguage:
    def test_persona_system_prompt_used(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation()
        settings = Settings()
        settings.conversation.persona = "technical"
        builder = ContextBuilder(settings, store)
        result = builder.build(conv.id, "hi")
        assert "technical assistant" in result.messages[0].content.lower()
        assert result.trace.persona_id == "technical"

    def test_language_note_appended(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation()
        settings = Settings()
        settings.conversation.language = "fi"
        builder = ContextBuilder(settings, store)
        result = builder.build(conv.id, "hei")
        assert "suomeksi" in result.messages[0].content
        assert result.trace.language_code == "fi"


class TestUserProfile:
    def test_profile_preferences_applied(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation()
        profile = UserProfile(
            id="u1",
            nickname="Fahad",
            units="imperial",
            conversation_style="casual",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        builder = ContextBuilder(Settings(), store, profile_store=_FixedProfileStore(profile))
        result = builder.build(conv.id, "hi")
        assert "Fahad" in result.messages[0].content
        assert "imperial" in result.messages[0].content
        assert "casual" in result.messages[0].content
        assert result.trace.profile_id == "u1"

    def test_no_profile_store_means_no_preferences(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation()
        builder = ContextBuilder(Settings(), store)
        result = builder.build(conv.id, "hi")
        assert result.trace.profile_id is None

    def test_profile_store_with_no_active_profile(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation()
        builder = ContextBuilder(Settings(), store, profile_store=_FixedProfileStore(None))
        result = builder.build(conv.id, "hi")
        assert result.trace.profile_id is None


class TestRetrievalGating:
    def test_missing_retriever_or_embedding_provider_skips_retrieval_silently(
        self, store: SQLiteMemoryStore
    ) -> None:
        conv = store.start_conversation()
        builder = ContextBuilder(Settings(), store)  # no retriever/embedding_provider
        result = builder.build(conv.id, "hi")
        assert result.trace.retrieved_memories == ()

    def test_embedding_disabled_in_settings_skips_retrieval(
        self, store: SQLiteMemoryStore
    ) -> None:
        conv = store.start_conversation()
        settings = Settings()
        settings.memory.embedding_enabled = False
        retriever = _FixedRetriever([_make_result("should not appear", 0.9)])
        builder = ContextBuilder(
            settings, store, retriever=retriever, embedding_provider=_FakeEmbeddingProvider()
        )
        result = builder.build(conv.id, "hi")
        assert result.trace.retrieved_memories == ()
        assert all("should not appear" not in m.content for m in result.messages)

    def test_retrieval_uses_settings_top_k(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation()
        settings = Settings()
        settings.memory.retrieval_top_k = 7
        retriever = _FixedRetriever([])
        builder = ContextBuilder(
            settings, store, retriever=retriever, embedding_provider=_FakeEmbeddingProvider()
        )
        builder.build(conv.id, "hi")
        assert retriever.last_top_k == 7

    def test_retrieval_searches_across_all_conversations_not_just_active(
        self, store: SQLiteMemoryStore
    ) -> None:
        """Semantic memory recalls *past* conversations (Part 3: "related
        conversations", "similar memories") — it must not be scoped to only
        the currently active one, unlike `recent_turns`."""
        conv = store.start_conversation()
        retriever = _FixedRetriever([])
        builder = ContextBuilder(
            Settings(), store, retriever=retriever, embedding_provider=_FakeEmbeddingProvider()
        )
        builder.build(conv.id, "hi")
        assert retriever.last_conversation_id is None


class TestBudgetTrimming:
    def test_memory_block_trimmed_when_over_budget(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation()
        settings = Settings()
        settings.memory.max_memory_chars = 100
        long_results = [_make_result("x" * 500, 0.9, turn_id=1)]
        retriever = _FixedRetriever(long_results)
        builder = ContextBuilder(
            settings, store, retriever=retriever, embedding_provider=_FakeEmbeddingProvider()
        )
        result = builder.build(conv.id, "hi")
        assert "relevant_memories" in result.trace.trimmed_sections
        memory_message = result.messages[1]
        assert len(memory_message.content) <= 100

    def test_summary_trimmed_when_over_budget(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation()
        store.add_summary(
            MemorySummary(
                conversation_id=conv.id,
                turn_range_start=1,
                turn_range_end=1,
                text="y" * 500,
                created_at=datetime.now(UTC),
                model_id="test",
            )
        )
        settings = Settings()
        settings.memory.max_summary_chars = 100
        builder = ContextBuilder(settings, store)
        result = builder.build(conv.id, "hi")
        assert "summary" in result.trace.trimmed_sections

    def test_within_budget_not_marked_trimmed(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation()
        retriever = _FixedRetriever([_make_result("short", 0.9)])
        builder = ContextBuilder(
            Settings(), store, retriever=retriever, embedding_provider=_FakeEmbeddingProvider()
        )
        result = builder.build(conv.id, "hi")
        assert result.trace.trimmed_sections == ()


class TestTrace:
    def test_trace_reflects_retrieved_memory_scores(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation()
        results = [_make_result("first", 0.9, turn_id=10), _make_result("second", 0.5, turn_id=20)]
        retriever = _FixedRetriever(results)
        builder = ContextBuilder(
            Settings(), store, retriever=retriever, embedding_provider=_FakeEmbeddingProvider()
        )
        result = builder.build(conv.id, "hi")
        assert [m.turn_id for m in result.trace.retrieved_memories] == [10, 20]
        assert [m.score for m in result.trace.retrieved_memories] == [0.9, 0.5]

    def test_trace_recent_turn_count(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation()
        for i in range(4):
            store.add_turn(conv.id, "user", f"turn {i}")
        builder = ContextBuilder(Settings(), store)
        result = builder.build(conv.id, "current")
        assert result.trace.recent_turn_count == 4

    def test_trace_summary_preview_truncated(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation()
        store.add_summary(
            MemorySummary(
                conversation_id=conv.id,
                turn_range_start=1,
                turn_range_end=1,
                text="z" * 200,
                created_at=datetime.now(UTC),
                model_id="test",
            )
        )
        builder = ContextBuilder(Settings(), store)
        result = builder.build(conv.id, "hi")
        assert result.trace.summary_included is True
        assert result.trace.summary_text_preview is not None
        assert len(result.trace.summary_text_preview) <= 80
