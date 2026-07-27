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
        system = _system(store).lower()
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


class TestLongTermRecall:
    """M5.4 §1: remembered facts must actually reach the prompt in LATER
    conversations — the nickname acceptance case."""

    def test_keyword_fallback_recalls_across_conversations(self, store: SQLiteMemoryStore) -> None:
        """Without an embedding model, recall degrades to keyword search —
        not to nothing (the pre-M5.4 behavior)."""
        earlier = store.start_conversation()
        store.add_turn(earlier.id, "user", "My nickname is Fahad.")
        store.add_turn(earlier.id, "assistant", "Nice to meet you, Fahad!")

        later = store.start_conversation()
        builder = ContextBuilder(Settings(), store)  # no retriever/embedding
        built = builder.build(later.id, "What is my nickname?")
        system = built.messages[0].content
        assert "Fahad" in system
        assert "You remember" in system

    def test_fallback_produces_retrieval_trace(self, store: SQLiteMemoryStore) -> None:
        earlier = store.start_conversation()
        store.add_turn(earlier.id, "user", "My favorite color is teal.")
        later = store.start_conversation()
        built = ContextBuilder(Settings(), store).build(later.id, "Suggest a color for my office")
        assert any("teal" in m.text_preview for m in built.trace.retrieved_memories)

    def test_fallback_never_breaks_the_turn_on_store_errors(
        self, store: SQLiteMemoryStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(*a: object, **k: object) -> list[MemorySearchResult]:
            raise RuntimeError("index corrupted")

        monkeypatch.setattr(store, "search_text", boom)
        conv = store.start_conversation()
        built = ContextBuilder(Settings(), store).build(conv.id, "What is my nickname?")
        assert built.messages  # composed fine, just without memories
        assert built.trace.retrieved_memories == ()


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


def _system_spoken(
    store: SQLiteMemoryStore,
    settings: Settings | None = None,
    *,
    spoken: bool,
    devices: dict[str, str] | None = None,
) -> str:
    conv = store.start_conversation()
    builder = ContextBuilder(
        settings or Settings(),
        store,
        runtime_devices=(lambda: devices) if devices is not None else None,
    )
    return builder.build(conv.id, "hi", spoken=spoken).messages[0].content


class TestRuntimeAwareness:
    """EVA must never guess where it is running (M7.3).

    Measured before this: five phrasings of "are you using my GPU?" produced
    five different answers, three of them wrong — including "your RTX 3060 is
    free to do what it wants" while both the LLM and ASR were resident on
    CUDA. The prompt listed model names and the hardware the machine
    contains, but never which device each component executes on.
    """

    def test_live_devices_reach_the_prompt(self, store: SQLiteMemoryStore) -> None:
        system = _system_spoken(
            store, spoken=False, devices={"llm": "cuda", "asr": "cuda", "tts": "cpu"}
        )
        assert "running on the GPU (CUDA)" in system
        assert "running on the CPU" in system

    def test_each_component_reports_its_own_device(self, store: SQLiteMemoryStore) -> None:
        """A single shared device string would be a different wrong answer:
        TTS is CPU-resident even when the LLM and ASR are on the GPU."""
        settings = Settings()
        system = _system_spoken(
            store, settings, spoken=False, devices={"llm": "cuda", "asr": "cuda", "tts": "cpu"}
        )
        assert f"LLM model = {settings.llm.model}, running on the GPU (CUDA)" in system
        assert f"ASR model = {settings.asr.model}, running on the GPU (CUDA)" in system
        assert f"TTS model = {settings.tts.model}, running on the CPU" in system

    def test_unloaded_components_say_so_rather_than_going_silent(
        self, store: SQLiteMemoryStore
    ) -> None:
        """`tts.lazy_load` leaves TTS "unloaded" for the first turns. Omitting
        the device outright is worse than naming the state: with the line
        simply absent, the model invented one ("text-to-speech is running on
        the GPU" while it was not loaded at all)."""
        system = _system_spoken(
            store, spoken=False, devices={"llm": "cuda", "asr": "cuda", "tts": "unloaded"}
        )
        assert f"TTS model = {Settings().tts.model}, running on not loaded yet" in system

    def test_locality_is_stated_not_implied(self, store: SQLiteMemoryStore) -> None:
        assert "no request is sent to any remote service" in _system(store)

    def test_no_provider_keeps_the_previous_output(self, store: SQLiteMemoryStore) -> None:
        """Backward compatibility: callers that pass no provider (context
        preview, benchmarks) get exactly the model-name-only block."""
        system = _system(store)
        assert "running on" not in system
        assert f"LLM model = {Settings().llm.model}" in system


class TestWorldFactHonesty:
    """Uncertainty about the world is handled like uncertainty about the user.

    Measured before this: asked for the weather, EVA answered "it's likely
    warm and sunny in many parts of Europe" — inferred from the timezone in
    its own system-facts block. Spoken aloud, the trailing hedge does not
    undo the assertion.
    """

    def test_volatile_categories_are_named(self, store: SQLiteMemoryStore) -> None:
        system = _system(store).lower()
        for category in ("weather", "news", "prices", "scores", "standings"):
            assert category in system, category

    def test_inference_and_hedging_are_both_forbidden(self, store: SQLiteMemoryStore) -> None:
        """Naming the failure modes matters: "don't guess" alone still let
        the model estimate from the date and hedge with "probably"."""
        system = _system(store).lower()
        assert "do not estimate it" in system
        assert "infer it from the date or location" in system
        assert "probably" in system

    def test_it_must_not_hand_the_question_back(self, store: SQLiteMemoryStore) -> None:
        """It asked "who is leading the standings right now?" — pushing the
        unknown onto the user instead of admitting it."""
        assert "do not ask the user to supply it" in _system(store).lower()


class TestDeliveryStyle:
    """Spoken and on-screen turns get different formatting rules."""

    def test_spoken_turns_ask_for_brevity_and_prose(self, store: SQLiteMemoryStore) -> None:
        system = _system_spoken(store, spoken=True).lower()
        assert "read aloud" in system
        assert "one to three sentences" in system
        assert "no markdown" in system

    def test_spoken_turns_forbid_screen_only_phrasing(self, store: SQLiteMemoryStore) -> None:
        """ "Copy the text and paste it here" is an instruction the user
        physically cannot follow in a voice conversation."""
        assert "paste, click, type or look at" in _system_spoken(store, spoken=True)

    def test_spoken_turns_damp_the_closing_question_reflex(self, store: SQLiteMemoryStore) -> None:
        """Every one of twelve probe replies ended with a follow-up question."""
        system = _system_spoken(store, spoken=True).lower()
        assert "only when you genuinely need the answer" in system

    def test_text_turns_still_allow_markdown_and_tables(self, store: SQLiteMemoryStore) -> None:
        system = _system_spoken(store, spoken=False).lower()
        assert "markdown is welcome" in system
        assert "tables" in system
        assert "read aloud" not in system

    def test_exactly_one_delivery_style_per_turn(self, store: SQLiteMemoryStore) -> None:
        spoken = _system_spoken(store, spoken=True).lower()
        text = _system_spoken(store, spoken=False).lower()
        assert "markdown is welcome" not in spoken
        assert "no markdown" not in text

    def test_default_is_the_on_screen_style(self, store: SQLiteMemoryStore) -> None:
        """`spoken` defaults to False so the context-preview endpoint and the
        benchmarks keep the behaviour they had."""
        assert "read on screen" in _system(store)

    def test_delivery_style_follows_the_persona(self, store: SQLiteMemoryStore) -> None:
        """A persona sets tone; it must not talk the model out of the length
        and formatting limits the output channel imposes."""
        settings = Settings()
        settings.conversation.persona = "technical"  # asks for detail and structure
        system = _system_spoken(store, settings, spoken=True)
        register_builtin_personas()
        persona_prompt = persona_registry.get("technical").system_prompt
        assert system.index(persona_prompt) < system.index("Your reply will be read aloud")
