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


class _ScannedRetriever(MemoryRetriever):
    """Exposes a mutable `last_scan_count` like the real
    `NumpyMemoryRetriever` (Batch 7 decision 10.2), so `ContextBuilder`'s
    `getattr` wiring — and per-call independence from any prior call's
    reading — can be tested without a real embedding index."""

    def __init__(self, results: list[MemorySearchResult], scan_count: int) -> None:
        self._results = results
        self.last_scan_count = scan_count

    def retrieve(
        self, query_vector: bytes, *, top_k: int, conversation_id: str | None = None
    ) -> list[MemorySearchResult]:
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

        # Exactly one system message (identity + persona + language +
        # technical facts + memory + summary, all merged) — a chat-template
        # requirement (ADR-021 amendment), not just a style choice.
        assert result.messages[0].role == "system"
        assert sum(1 for m in result.messages if m.role == "system") == 1

        system_content = result.messages[0].content
        assert "relevant fact" in system_content
        assert "weather" in system_content

        contents = [m.content for m in result.messages]
        assert contents[1] == "earlier question"
        assert contents[2] == "earlier answer"
        assert contents[-1] == "new question"  # current utterance always last

    def test_no_memories_or_summary_still_produces_valid_messages(
        self, store: SQLiteMemoryStore
    ) -> None:
        conv = store.start_conversation()
        builder = ContextBuilder(Settings(), store)
        result = builder.build(conv.id, "hello")
        assert result.messages[0].role == "system"
        assert result.messages[-1] == ChatMessage(role="user", content="hello")


class TestIdentity:
    def test_identity_present_regardless_of_persona(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation()
        settings = Settings()
        settings.conversation.persona = "creative"
        builder = ContextBuilder(settings, store)
        result = builder.build(conv.id, "hi")
        assert "Edge Voice Assistant" in result.messages[0].content

    def test_technical_facts_block_contains_configured_models(
        self, store: SQLiteMemoryStore
    ) -> None:
        conv = store.start_conversation()
        settings = Settings()
        builder = ContextBuilder(settings, store)
        result = builder.build(conv.id, "hi")
        facts = result.messages[0].content
        assert settings.llm.model in facts
        assert settings.asr.model in facts
        assert settings.tts.model in facts
        assert "only if explicitly asked" in facts.lower()

    def test_runtime_device_still_resolves_for_a_local_llm_adapter(
        self, store: SQLiteMemoryStore
    ) -> None:
        """Regression test for the C1 port-split risk named in the review: a
        careless split could make EVA silently unable to answer "are you
        using the GPU?", with no test failure — because `engine_device()`
        now sits between `runtime_devices` and the raw `.device` read,
        exactly mirroring `build_assistant`'s real wiring
        (`lambda: {"llm": engine_device(llm), ...}`)."""
        from eva.llm.base import engine_device

        class _FakeLocalLLM:
            device = "cuda"

            def load(self) -> None: ...
            def unload(self) -> None: ...

        conv = store.start_conversation()
        settings = Settings()
        llm = _FakeLocalLLM()
        builder = ContextBuilder(
            settings,
            store,
            runtime_devices=lambda: {"llm": engine_device(llm)},  # type: ignore[arg-type]
        )
        facts = builder.build(conv.id, "hi").messages[0].content
        assert "running on the GPU (CUDA)" in facts

    def test_runtime_device_reports_remote_for_a_non_local_provider(
        self, store: SQLiteMemoryStore
    ) -> None:
        """The mirror image: a remote/API-backed provider has no `device` to
        report, so the prompt must say so honestly rather than guessing or
        raising `AttributeError` mid-turn."""
        from eva.llm.base import engine_device

        class _FakeRemoteLLM:
            def stream(self) -> None: ...

        conv = store.start_conversation()
        settings = Settings()
        llm = _FakeRemoteLLM()
        builder = ContextBuilder(
            settings,
            store,
            runtime_devices=lambda: {"llm": engine_device(llm)},  # type: ignore[arg-type]
        )
        facts = builder.build(conv.id, "hi").messages[0].content
        assert "running on remote" in facts

    def test_only_one_system_message_ever_emitted(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation()
        store.add_turn(conv.id, "user", "hi")
        store.add_turn(conv.id, "assistant", "hello")
        store.add_summary(
            MemorySummary(
                conversation_id=conv.id,
                turn_range_start=1,
                turn_range_end=2,
                text="greeting exchanged",
                created_at=datetime.now(UTC),
                model_id="test",
            )
        )
        retriever = _FixedRetriever([_make_result("a fact", 0.9)])
        builder = ContextBuilder(
            Settings(), store, retriever=retriever, embedding_provider=_FakeEmbeddingProvider()
        )
        result = builder.build(conv.id, "another question")
        system_messages = [m for m in result.messages if m.role == "system"]
        assert len(system_messages) == 1
        assert result.messages[0].role == "system"


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


class TestSessionName:
    """A name stated during the session is a durable fact in the system prompt,
    so the assistant stays consistent about it regardless of query or history
    depth (the within-session contradiction this fixes)."""

    def test_session_name_added_to_system_prompt(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation()
        builder = ContextBuilder(Settings(), store)
        result = builder.build(conv.id, "anything", session_name="Fahad")
        assert "The user's name is Fahad." in result.messages[0].content

    def test_session_name_takes_precedence_over_profile_nickname(
        self, store: SQLiteMemoryStore
    ) -> None:
        conv = store.start_conversation()
        profile = UserProfile(
            id="u1", nickname="OldName", created_at=datetime.now(UTC), updated_at=datetime.now(UTC)
        )
        builder = ContextBuilder(Settings(), store, profile_store=_FixedProfileStore(profile))
        result = builder.build(conv.id, "hi", session_name="NewName")
        system = result.messages[0].content
        assert "The user's name is NewName." in system
        assert "OldName" not in system

    def test_profile_nickname_used_when_no_session_name(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation()
        profile = UserProfile(
            id="u1", nickname="Fahad", created_at=datetime.now(UTC), updated_at=datetime.now(UTC)
        )
        builder = ContextBuilder(Settings(), store, profile_store=_FixedProfileStore(profile))
        result = builder.build(conv.id, "hi")  # no session_name
        assert "The user's name is Fahad." in result.messages[0].content

    def test_no_name_line_when_neither_present(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation()
        builder = ContextBuilder(Settings(), store)
        result = builder.build(conv.id, "hi")
        assert "The user's name is" not in result.messages[0].content


class TestRetrievalGating:
    def test_missing_retriever_or_embedding_provider_skips_retrieval_silently(
        self, store: SQLiteMemoryStore
    ) -> None:
        conv = store.start_conversation()
        builder = ContextBuilder(Settings(), store)  # no retriever/embedding_provider
        result = builder.build(conv.id, "hi")
        assert result.trace.retrieved_memories == ()

    def test_embedding_disabled_in_settings_skips_retrieval(self, store: SQLiteMemoryStore) -> None:
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
        # Trimming still bounds the memory *section* even though it's now
        # merged into the single system message rather than its own message.
        system_content = result.messages[0].content
        assert "x" * 100 not in system_content

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


class TestAlternationNormalization:
    def test_consecutive_same_role_turns_are_merged_not_rejected(
        self, store: SQLiteMemoryStore
    ) -> None:
        """A malformed import or a dangling unanswered turn could leave two
        same-speaker turns adjacent in storage — `build()` must still
        produce a valid, single-system-message, strictly-alternating list
        rather than raising (ADR-021 amendment)."""
        conv = store.start_conversation()
        store.add_turn(conv.id, "user", "first thing")
        store.add_turn(conv.id, "user", "second thing")  # no assistant reply between
        store.add_turn(conv.id, "assistant", "a reply")

        builder = ContextBuilder(Settings(), store)
        result = builder.build(conv.id, "current question")

        roles = [m.role for m in result.messages]
        assert roles == ["system", "user", "assistant", "user"]
        assert "first thing" in result.messages[1].content
        assert "second thing" in result.messages[1].content

    def test_dangling_unanswered_user_turn_merges_with_current_question(
        self, store: SQLiteMemoryStore
    ) -> None:
        conv = store.start_conversation()
        store.add_turn(conv.id, "user", "unanswered question")

        builder = ContextBuilder(Settings(), store)
        result = builder.build(conv.id, "follow-up question")

        roles = [m.role for m in result.messages]
        assert roles == ["system", "user"]
        assert "unanswered question" in result.messages[1].content
        assert "follow-up question" in result.messages[1].content


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
        # Real conversations always alternate speakers (the orchestrator
        # only ever writes a user+assistant pair together) — strict
        # alternation is also what the chat-template contract requires
        # (ADR-021 amendment), so test data must reflect that shape too.
        for i in range(2):
            store.add_turn(conv.id, "user", f"user turn {i}")
            store.add_turn(conv.id, "assistant", f"assistant turn {i}")
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


class TestRetrievalAndContextTiming:
    """Batch 7 (M4): retrieval_ms/context_ms/retrieval_score_top1/
    retrieval_scan_count are per-call values carried on THIS build's trace —
    not the mutable `self._last_retrieval_ms`-style instance state the prior
    design used, which let a later build's read attribute the wrong turn's
    numbers (see `.dev/EVA — Principal Architecture Review.md`)."""

    def test_context_ms_covers_the_whole_build_including_retrieval(
        self, store: SQLiteMemoryStore
    ) -> None:
        conv = store.start_conversation()
        retriever = _FixedRetriever([_make_result("a fact", 0.9)])
        builder = ContextBuilder(
            Settings(), store, retriever=retriever, embedding_provider=_FakeEmbeddingProvider()
        )
        result = builder.build(conv.id, "hi")
        assert result.trace.context_ms >= result.trace.retrieval_ms
        assert result.trace.retrieval_ms >= 0
        assert result.trace.context_ms >= 0

    def test_retrieval_score_top1_matches_the_highest_scored_result(
        self, store: SQLiteMemoryStore
    ) -> None:
        conv = store.start_conversation()
        results = [_make_result("first", 0.9, turn_id=10), _make_result("second", 0.5, turn_id=20)]
        retriever = _FixedRetriever(results)
        builder = ContextBuilder(
            Settings(), store, retriever=retriever, embedding_provider=_FakeEmbeddingProvider()
        )
        result = builder.build(conv.id, "hi")
        assert result.trace.retrieval_score_top1 == 0.9

    def test_retrieval_score_top1_is_none_when_nothing_is_retrieved(
        self, store: SQLiteMemoryStore
    ) -> None:
        conv = store.start_conversation()
        builder = ContextBuilder(Settings(), store)  # no retriever configured
        result = builder.build(conv.id, "hi")
        assert result.trace.retrieval_score_top1 is None
        assert result.trace.retrieval_ms == 0
        assert result.trace.retrieval_scan_count == 0

    def test_scan_count_is_read_from_the_concrete_retriever_not_the_port(
        self, store: SQLiteMemoryStore
    ) -> None:
        """The `MemoryRetriever` port stays unchanged (decision 10.2) — the
        count is read via `getattr`, so it must reflect whatever the
        concrete retriever reports, and degrade to 0 for one that doesn't
        expose it at all (`_FixedRetriever`, below)."""
        conv = store.start_conversation()
        retriever = _ScannedRetriever([_make_result("x", 0.5)], scan_count=1847)
        builder = ContextBuilder(
            Settings(), store, retriever=retriever, embedding_provider=_FakeEmbeddingProvider()
        )
        result = builder.build(conv.id, "hi")
        assert result.trace.retrieval_scan_count == 1847

    def test_retriever_without_last_scan_count_degrades_to_zero(
        self, store: SQLiteMemoryStore
    ) -> None:
        conv = store.start_conversation()
        retriever = _FixedRetriever([_make_result("x", 0.5)])  # no last_scan_count attribute
        builder = ContextBuilder(
            Settings(), store, retriever=retriever, embedding_provider=_FakeEmbeddingProvider()
        )
        result = builder.build(conv.id, "hi")
        assert result.trace.retrieval_scan_count == 0

    def test_two_calls_never_cross_contaminate_each_others_trace(
        self, store: SQLiteMemoryStore
    ) -> None:
        """The regression this batch fixes: retrieval numbers must belong to
        the call that produced them, not to whichever call happened to run
        (or finish) most recently on the shared retriever instance. Proven
        here by changing the retriever's state BETWEEN two calls on the SAME
        builder and confirming each call's trace matches its own moment,
        never the other's — the mechanism that makes this safe is that
        `ContextBuilder` no longer holds any of this as instance state at
        all (verified structurally: no `_last_retrieval_ms`-style attribute
        exists post-construction)."""
        conv = store.start_conversation()
        retriever = _ScannedRetriever([_make_result("first", 0.9)], scan_count=100)
        builder = ContextBuilder(
            Settings(), store, retriever=retriever, embedding_provider=_FakeEmbeddingProvider()
        )
        assert not hasattr(builder, "_last_retrieval_ms")
        assert not hasattr(builder, "_last_retrieval_top_score")

        first = builder.build(conv.id, "hi")
        assert first.trace.retrieval_scan_count == 100
        assert first.trace.retrieval_score_top1 == 0.9

        retriever.last_scan_count = 999
        retriever._results = [_make_result("second", 0.1)]
        second = builder.build(conv.id, "hi again")
        assert second.trace.retrieval_scan_count == 999
        assert second.trace.retrieval_score_top1 == 0.1

        # The first call's already-returned trace is unaffected by the
        # retriever's later state change — proof there is no shared mutable
        # attribute a later call could have corrupted it through.
        assert first.trace.retrieval_scan_count == 100
        assert first.trace.retrieval_score_top1 == 0.9
