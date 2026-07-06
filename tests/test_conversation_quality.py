"""Conversational-quality regression tests (M5.2, ADR-021 Amendment 3).

What a unit test CAN pin down about conversation quality: the exact prompt
the LLM receives — its hierarchy, its guidance, what context reaches it, and
in what order. (Whether the model then *behaves* is validated live against
the real LLM in MANUAL_TESTING §15.)
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from eva.config.settings import Settings
from eva.conversation.context_builder import ContextBuilder
from eva.conversation.personas import persona_registry, register_builtin_personas
from eva.memory import db
from eva.memory.base import MemoryRetriever
from eva.memory.models import MemorySearchResult, MemorySummary, MemoryTurn
from eva.memory.sqlite_store import SQLiteMemoryStore

BUILTIN_IDS = ["default", "professional", "friendly", "technical", "teacher", "minimal", "creative"]


@pytest.fixture
def store(tmp_path: Path) -> Iterator[SQLiteMemoryStore]:
    conn = db.connect(tmp_path / "memory.db")
    s = SQLiteMemoryStore(conn)
    yield s
    s.close()


def _system(store: SQLiteMemoryStore, settings: Settings | None = None, text: str = "hi") -> str:
    conv = store.start_conversation()
    built = ContextBuilder(settings or Settings(), store).build(conv.id, text)
    return built.messages[0].content


class _FixedRetriever(MemoryRetriever):
    def __init__(self, results: list[MemorySearchResult]) -> None:
        self._results = results

    def retrieve(
        self, query_vector: bytes, *, top_k: int, conversation_id: str | None = None
    ) -> list[MemorySearchResult]:
        return self._results


def _result(text: str, score: float, turn_id: int) -> MemorySearchResult:
    turn = MemoryTurn(
        id=turn_id, conversation_id="c", created_at=datetime.now(UTC), speaker="user", text=text
    )
    return MemorySearchResult(turn=turn, score=score, match_reason="semantic")


class TestPromptHierarchy:
    def test_identity_named_exactly_once(self, store: SQLiteMemoryStore) -> None:
        """Natural identity (M5.2 §5): the name appears once — not hammered
        into the prompt in a way that makes the model repeat it."""
        assert _system(store).count("Edge Voice Assistant") == 1

    def test_continuity_guidance_present(self, store: SQLiteMemoryStore) -> None:
        """Fragments/pronouns must be treated as continuations (M5.2 §1)."""
        system = _system(store)
        assert "continue the current topic" in system
        assert "pronouns" in system.lower()

    def test_helpfulness_over_literalness_guidance(self, store: SQLiteMemoryStore) -> None:
        """'I am not a spreadsheet' class of reply (M5.2 §3): the prompt must
        steer toward doing the task, not defending identity."""
        system = _system(store)
        assert "never refuse on the grounds of not being that kind of tool" in system
        assert "what the user is trying to do" in system

    def test_capability_messaging_is_build_scoped_not_permanent(
        self, store: SQLiteMemoryStore
    ) -> None:
        """Image capability (M5.2 §2): 'not enabled in this build', never
        'impossible forever'."""
        system = _system(store)
        assert "not enabled" in system
        assert "planned" in system
        assert "image understanding" in system

    def test_behavior_guidance_precedes_persona_style(self, store: SQLiteMemoryStore) -> None:
        """Continuity/helpfulness rules hold for every persona — they come
        before (and therefore apply regardless of) the persona's voice."""
        settings = Settings()
        settings.conversation.persona = "minimal"
        system = _system(store, settings)
        assert system.index("continue the current topic") < system.index("as few words")

    def test_technical_facts_are_the_last_section(self, store: SQLiteMemoryStore) -> None:
        """Low salience by position: backend details must trail everything
        else so the model doesn't volunteer them (M5.2 §5/§8)."""
        conv = store.start_conversation()
        store.add_summary(
            MemorySummary(
                conversation_id=conv.id,
                turn_range_start=1,
                turn_range_end=1,
                text="talked about planets",
                created_at=datetime.now(UTC),
                model_id="t",
            )
        )
        built = ContextBuilder(Settings(), store).build(conv.id, "hi")
        sections = built.messages[0].content.split("\n\n")
        assert sections[-1].startswith("Technical backend details")

    def test_summary_section_precedes_memory_section(self, store: SQLiteMemoryStore) -> None:
        """This conversation's summary (continuity) outranks cross-
        conversation memories (background) — ADR-021 Amendment 3 order."""
        conv = store.start_conversation()
        store.add_summary(
            MemorySummary(
                conversation_id=conv.id,
                turn_range_start=1,
                turn_range_end=1,
                text="SUMMARYMARKER",
                created_at=datetime.now(UTC),
                model_id="t",
            )
        )
        retriever = _FixedRetriever([_result("MEMORYMARKER", 0.9, 1)])

        class _NullEmbedding:
            def embed(self, text: str):
                import numpy as np

                return np.zeros(4, dtype=np.float32)

        builder = ContextBuilder(
            Settings(),
            store,
            retriever=retriever,
            embedding_provider=_NullEmbedding(),  # type: ignore[arg-type]
        )
        system = builder.build(conv.id, "hi").messages[0].content
        assert system.index("SUMMARYMARKER") < system.index("MEMORYMARKER")


class TestMultiTurnContext:
    def test_fragment_followup_has_its_antecedent_in_the_prompt(
        self, store: SQLiteMemoryStore
    ) -> None:
        """The 'with rows and columns.' scenario (M5.2 §1): the turn being
        continued must be present in the message list so the model CAN
        resolve the fragment."""
        conv = store.start_conversation()
        store.add_turn(conv.id, "user", "Create a markdown table of two planets.")
        store.add_turn(conv.id, "assistant", "| Planet | Diameter |\n|---|---|\n| Mars | 6779 km |")
        built = ContextBuilder(Settings(), store).build(conv.id, "with rows and columns.")
        contents = [m.content for m in built.messages]
        assert any("markdown table of two planets" in c for c in contents)
        assert contents[-1] == "with rows and columns."

    def test_pronoun_followup_has_referent_in_window(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation()
        store.add_turn(conv.id, "user", "Tell me about the Eiffel Tower.")
        store.add_turn(conv.id, "assistant", "The Eiffel Tower is a landmark in Paris.")
        built = ContextBuilder(Settings(), store).build(conv.id, "how tall is it?")
        contents = [m.content for m in built.messages]
        assert any("Eiffel Tower" in c for c in contents)

    def test_default_window_keeps_twenty_turns(self, store: SQLiteMemoryStore) -> None:
        """max_history_turns=20 by default — enough that recent context is
        never the reason a follow-up fails (M5.2 §6)."""
        conv = store.start_conversation()
        for i in range(15):
            store.add_turn(conv.id, "user", f"question {i}")
            store.add_turn(conv.id, "assistant", f"answer {i}")
        built = ContextBuilder(Settings(), store).build(conv.id, "next")
        assert built.trace.recent_turn_count == 20


class TestRetrievedMemoryOrdering:
    def test_memory_block_preserves_score_order(self, store: SQLiteMemoryStore) -> None:
        """Highest-scored memory first in the block (the retriever sorts;
        the formatter must not reorder)."""
        conv = store.start_conversation()
        retriever = _FixedRetriever(
            [_result("BEST", 0.9, 1), _result("MIDDLE", 0.5, 2), _result("WORST", 0.2, 3)]
        )

        class _NullEmbedding:
            def embed(self, text: str):
                import numpy as np

                return np.zeros(4, dtype=np.float32)

        builder = ContextBuilder(
            Settings(),
            store,
            retriever=retriever,
            embedding_provider=_NullEmbedding(),  # type: ignore[arg-type]
        )
        system = builder.build(conv.id, "hi").messages[0].content
        assert system.index("BEST") < system.index("MIDDLE") < system.index("WORST")

    def test_memory_block_framed_as_natural_recall(self, store: SQLiteMemoryStore) -> None:
        """M5.2 §7: memories are 'things you remember, use naturally' — not a
        document to recite."""
        conv = store.start_conversation()
        retriever = _FixedRetriever([_result("user likes teal", 0.9, 1)])

        class _NullEmbedding:
            def embed(self, text: str):
                import numpy as np

                return np.zeros(4, dtype=np.float32)

        builder = ContextBuilder(
            Settings(),
            store,
            retriever=retriever,
            embedding_provider=_NullEmbedding(),  # type: ignore[arg-type]
        )
        system = builder.build(conv.id, "hi").messages[0].content
        assert "You remember" in system
        assert "don't announce" in system


class TestPersonaDistinctness:
    def test_teacher_persona_registered(self) -> None:
        register_builtin_personas()
        teacher = persona_registry.get("teacher")
        assert teacher.display_name == "Teacher"
        assert "analogy" in teacher.system_prompt

    def test_all_builtin_prompts_are_pairwise_distinct_and_substantial(self) -> None:
        """M5.2 §4: personas must be *noticeably* different — one-liner
        prompts collapsed into the same generic voice on a small LLM, so
        every prompt must now carry real, distinct style instructions."""
        register_builtin_personas()
        prompts = {pid: persona_registry.get(pid).system_prompt for pid in BUILTIN_IDS}
        assert len(set(prompts.values())) == len(prompts)  # pairwise distinct
        for pid, prompt in prompts.items():
            assert len(prompt) > 100, f"persona '{pid}' prompt is too thin to shape style"

    def test_each_persona_produces_a_different_system_message(
        self, store: SQLiteMemoryStore
    ) -> None:
        systems = set()
        for pid in BUILTIN_IDS:
            settings = Settings()
            settings.conversation.persona = pid
            systems.add(_system(store, settings))
        assert len(systems) == len(BUILTIN_IDS)

    def test_identity_and_continuity_shared_by_every_persona(
        self, store: SQLiteMemoryStore
    ) -> None:
        """Style varies; identity and behavior rules must not."""
        for pid in BUILTIN_IDS:
            settings = Settings()
            settings.conversation.persona = pid
            system = _system(store, settings)
            assert system.count("Edge Voice Assistant") == 1, pid
            assert "continue the current topic" in system, pid
