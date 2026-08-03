"""Generation limits and truncation honesty (M7.3 Tier 1).

Traced from a real session: a request for a single-file HTML page was cut off
at the 512-token ceiling three times in a row - byte-identically, which is what
proved it was a deterministic cap rather than cancellation or transport. The
adapter discarded llama.cpp's `finish_reason`, so the truncated reply was
stored, replayed as history, and described by the model as "complete and ready
to run".
"""

from __future__ import annotations

import inspect
import json
import logging
import threading
from collections.abc import Generator, Iterator
from pathlib import Path
from typing import Any

import pytest

from eva.config.settings import SETTINGS_SCHEMA_VERSION, Settings, load_settings
from eva.conversation.chunker import SentenceChunker
from eva.conversation.context_builder import ContextBuilder
from eva.conversation.markdown import speakable_end
from eva.core.tools import ToolDefinition
from eva.llm.base import ChatMessage, GenerationParams, LLMEngine
from eva.llm.llamacpp import LlamaCppLLM
from eva.memory import db
from eva.memory.sqlite_store import SQLiteMemoryStore

CODE_FENCE = "```"


@pytest.fixture
def store(tmp_path: Path) -> Any:
    conn = db.connect(tmp_path / "memory.db")
    s = SQLiteMemoryStore(conn)
    yield s
    s.close()


def _chunker() -> SentenceChunker:
    return SentenceChunker(min_chars=12, max_chars=350, first_chunk_min_chars=6)


def _drain(gen: Generator[str, None, Any]) -> tuple[str, str]:
    """Consume a stream and report its text and finish reason.

    `stream()` returns a `GenerationOutcome`; an adapter that returns nothing
    is read as an ordinary completion.
    """
    parts: list[str] = []
    while True:
        try:
            parts.append(next(gen))
        except StopIteration as done:
            outcome = done.value
            return "".join(parts), (outcome.reason if outcome is not None else "stop")


def _engine_with(chunks: list[dict[str, Any]]) -> LlamaCppLLM:
    """A LlamaCppLLM wired to a stub completion - no weights required."""
    engine = LlamaCppLLM.__new__(LlamaCppLLM)

    class _Fake:
        def create_chat_completion(self, **_: Any) -> Iterator[dict[str, Any]]:
            yield from chunks

    engine._llama = _Fake()  # type: ignore[attr-defined]
    engine._infer_lock = threading.Lock()  # type: ignore[attr-defined]
    return engine


def _estimate(text: str) -> int:
    return max(1, len(text) // 4)


class TestFinishReason:
    """The adapter must distinguish "finished" from "ran out of room"."""

    def test_length_is_reported_not_swallowed(self) -> None:
        engine = _engine_with(
            [
                {"choices": [{"delta": {"content": "def f():"}, "finish_reason": None}]},
                {"choices": [{"delta": {"content": " return"}, "finish_reason": "length"}]},
            ]
        )
        text, reason = _drain(
            engine.stream(
                [ChatMessage(role="system", content="s")], GenerationParams(), lambda: False
            )
        )
        assert text == "def f(): return"
        assert reason == "length"

    def test_normal_completion_reports_stop(self) -> None:
        engine = _engine_with(
            [{"choices": [{"delta": {"content": "Hello."}, "finish_reason": "stop"}]}]
        )
        _text, reason = _drain(
            engine.stream(
                [ChatMessage(role="system", content="s")], GenerationParams(), lambda: False
            )
        )
        assert reason == "stop"

    def test_an_adapter_that_reports_nothing_is_treated_as_complete(self) -> None:
        """Backward compatibility: a stub or older adapter that never sets a
        reason must not make every reply look truncated."""
        engine = _engine_with([{"choices": [{"delta": {"content": "Hi."}}]}])
        _text, reason = _drain(
            engine.stream(
                [ChatMessage(role="system", content="s")], GenerationParams(), lambda: False
            )
        )
        assert reason == "stop"

    def test_abort_is_distinct_from_completion(self) -> None:
        engine = _engine_with(
            [{"choices": [{"delta": {"content": "x"}, "finish_reason": None}]} for _ in range(10)]
        )
        calls = {"n": 0}

        def abort() -> bool:
            calls["n"] += 1
            return calls["n"] > 3

        _text, reason = _drain(
            engine.stream([ChatMessage(role="system", content="s")], GenerationParams(), abort)
        )
        assert reason == "abort", "a barge-in must not look like a finished answer"


class TestToolAvailability:
    """The LLM port can be *offered* tools, not handed executable ones.

    Batch 1 could report a tool call through `GenerationOutcome` but had no
    way to tell an adapter which tools existed, so the contract could only
    ever describe an answer, never a question.
    """

    def _recording_engine(self) -> tuple[LlamaCppLLM, dict[str, Any]]:
        engine = LlamaCppLLM.__new__(LlamaCppLLM)
        seen: dict[str, Any] = {}

        class _Fake:
            def create_chat_completion(self, **kwargs: Any) -> Iterator[dict[str, Any]]:
                seen.update(kwargs)
                yield {"choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}]}

        engine._llama = _Fake()  # type: ignore[attr-defined]
        engine._infer_lock = threading.Lock()  # type: ignore[attr-defined]
        return engine, seen

    def test_omitting_tools_sends_the_payload_it_always_did(self) -> None:
        """The correlation field must not leak into the chat template as a
        null: an ordinary turn has to serialize exactly as before."""
        engine, seen = self._recording_engine()
        text, reason = _drain(
            engine.stream(
                [ChatMessage(role="system", content="s")], GenerationParams(), lambda: False
            )
        )
        assert (text, reason) == ("hi", "stop")
        # Exactly the two keys the template saw before correlation existed,
        # plus the already-present `tool_calls`; no `call_id: None` alongside.
        assert seen["messages"] == [{"role": "system", "content": "s", "tool_calls": ()}]

    def test_definitions_reach_the_adapter(self, caplog: pytest.LogCaptureFixture) -> None:
        """This adapter cannot offer tools yet, so it must say so rather than
        answer as though the model declined to use them."""
        engine, _seen = self._recording_engine()
        definition = ToolDefinition(
            name="get_weather", description="Look up the weather.", parameters={"type": "object"}
        )
        with caplog.at_level(logging.WARNING):
            _drain(
                engine.stream(
                    [ChatMessage(role="system", content="s")],
                    GenerationParams(),
                    lambda: False,
                    tools=(definition,),
                )
            )
        assert "get_weather" in caplog.text
        assert "cannot offer tools" in caplog.text

    def test_the_port_takes_descriptions_not_executables(self) -> None:
        """Dependency direction: `eva.llm` must not reach into `eva.tools`,
        and an adapter must not receive anything it could invoke."""
        annotation = inspect.signature(LLMEngine.stream).parameters["tools"].annotation
        assert "ToolDefinition" in str(annotation)
        assert "eva.tools" not in Path("src/eva/llm/base.py").read_text(encoding="utf-8")

    def test_tools_is_keyword_only_with_an_empty_default(self) -> None:
        """Existing adapters and fakes keep working untouched."""
        parameter = inspect.signature(LLMEngine.stream).parameters["tools"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default == ()


class TestTruncationReachesTheModel:
    def test_a_truncated_prior_turn_is_marked_in_context(self, store: SQLiteMemoryStore) -> None:
        """Without this the model reads its own half-finished code as
        deliberate and answers "yes, that is complete"."""
        conv = store.start_conversation().id
        store.add_turn(conv, "user", "generate an HTML page")
        store.add_turn(
            conv,
            "assistant",
            f"{CODE_FENCE}html\n<!DOCTYPE html>\n<div class=",
            metadata={"finish_reason": "length"},
        )
        built = ContextBuilder(Settings(), store).build(conv, "is it complete?")
        assistant_msg = [m for m in built.messages if m.role == "assistant"][-1]
        assert "cut off by the length limit" in assistant_msg.content
        assert "incomplete" in assistant_msg.content

    def test_a_normal_prior_turn_is_untouched(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation().id
        store.add_turn(conv, "user", "hi")
        store.add_turn(conv, "assistant", "Hello there.")
        built = ContextBuilder(Settings(), store).build(conv, "how are you?")
        assistant_msg = [m for m in built.messages if m.role == "assistant"][-1]
        assert assistant_msg.content == "Hello there."

    def test_the_marker_carries_no_instruction_to_apologise(self) -> None:
        """A note that tells the model to apologise produces a turn that is
        mostly apology; this one states a fact and stops."""
        from eva.conversation.context_builder import _TRUNCATION_NOTE

        lowered = _TRUNCATION_NOTE.lower()
        assert "sorry" not in lowered
        assert "apolog" not in lowered


class TestHistoryTokenBudget:
    """`max_history_turns` bounds the turn COUNT, not their size. Raising
    max_tokens to 2048 made a handful of code artifacts able to overflow an
    8192-token window, which is not graceful degradation - it corrupts or
    drops the system prompt, where every behavioural rule lives.
    """

    def test_ordinary_dialogue_is_not_trimmed(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation().id
        for i in range(8):
            store.add_turn(conv, "user", f"question {i}")
            store.add_turn(conv, "assistant", f"answer {i}")
        trace = ContextBuilder(Settings(), store).build(conv, "next?").trace
        assert trace.history_turns_dropped == 0
        assert trace.recent_turn_count == 16

    def test_consecutive_large_artifacts_stay_inside_the_window(
        self, store: SQLiteMemoryStore
    ) -> None:
        settings = Settings()
        conv = store.start_conversation().id
        artifact = "x" * 8000  # ~2000 tokens under the chars/4 estimate
        for i in range(6):
            store.add_turn(conv, "user", f"generate artifact {i}")
            store.add_turn(conv, "assistant", artifact)

        built = ContextBuilder(settings, store).build(conv, "one more please")
        total = sum(_estimate(m.content) for m in built.messages)
        ceiling = settings.llm.providers.local.context_length - settings.conversation.max_tokens
        assert total <= ceiling, (
            f"prompt is {total} tokens, leaving no room for "
            f"{settings.conversation.max_tokens} of generation"
        )
        assert built.trace.history_turns_dropped > 0
        assert "history" in built.trace.trimmed_sections

    def test_the_newest_exchange_always_survives(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation().id
        artifact = "y" * 8000
        for i in range(6):
            store.add_turn(conv, "user", f"old request {i}")
            store.add_turn(conv, "assistant", artifact)
        store.add_turn(conv, "user", "THE NEWEST QUESTION")
        store.add_turn(conv, "assistant", "THE NEWEST ANSWER")

        built = ContextBuilder(Settings(), store).build(conv, "follow up")
        joined = "\n".join(m.content for m in built.messages)
        assert "THE NEWEST QUESTION" in joined
        assert "THE NEWEST ANSWER" in joined
        assert "old request 0" not in joined

    def test_trimming_never_leaves_a_leading_assistant_turn(self, store: SQLiteMemoryStore) -> None:
        """Every chat template rejects it, and trimming an odd number of
        turns can expose one. `validate_chat_messages` would hard-fail the
        turn, so this must hold by construction."""
        conv = store.start_conversation().id
        big = "z" * 8000
        store.add_turn(conv, "user", "first")
        for i in range(5):
            store.add_turn(conv, "assistant", big)
            store.add_turn(conv, "user", f"q{i}")
        built = ContextBuilder(Settings(), store).build(conv, "now")
        non_system = [m for m in built.messages if m.role != "system"]
        assert non_system[0].role == "user"

    def test_the_injected_tokenizer_is_used(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation().id
        for i in range(4):
            store.add_turn(conv, "user", f"q{i}")
            store.add_turn(conv, "assistant", "a" * 400)
        seen: list[str] = []

        def counter(text: str) -> int:
            seen.append(text)
            return len(text) // 4

        ContextBuilder(Settings(), store, token_counter=counter).build(conv, "next")
        assert seen, "the injected counter must actually be consulted"


class TestSpeakableEnd:
    """Where display-only trailing content begins, so a speech-paced view can
    reveal it without ever showing unspoken prose ahead of playback."""

    @pytest.mark.parametrize(
        ("label", "text", "expect_hidden_tail"),
        [
            ("normal prose", "Here you go. That should work.", False),
            (
                "prose then code",
                f"Here is the file.\n\n{CODE_FENCE}html\n<p>hi</p>\n{CODE_FENCE}",
                True,
            ),
            ("code only", f"{CODE_FENCE}html\n<!DOCTYPE html>\n{CODE_FENCE}", True),
            (
                "prose code prose",
                f"Here it is.\n\n{CODE_FENCE}py\nx=1\n{CODE_FENCE}\n\nLet me know.",
                False,
            ),
            ("table after prose", "Two planets.\n\n| P | D |\n|---|---|\n| Mars | 6779 |", False),
        ],
    )
    def test_boundary(self, label: str, text: str, expect_hidden_tail: bool) -> None:
        end = speakable_end(text, _chunker)
        assert 0 <= end <= len(text)
        assert (end < len(text)) is expect_hidden_tail, label

    def test_code_only_reply_has_no_spoken_content_at_all(self) -> None:
        """No sentence marker will ever fire, so the UI must be able to
        reveal the whole thing from the boundary alone."""
        assert speakable_end(f"{CODE_FENCE}html\n<!DOCTYPE html>\n{CODE_FENCE}", _chunker) == 0

    def test_boundary_matches_the_web_cursor_arithmetic(self) -> None:
        """The store advances its cursor with indexOf-from-last-position, and
        the chunker strips whitespace between segments - summing segment
        lengths drifts out of alignment with the source text."""
        text = f"Here it is.\n\n{CODE_FENCE}py\nx=1\n{CODE_FENCE}\n\nLet me know."
        assert speakable_end(text, _chunker) == len(text)


class TestGenerationCapMigration:
    def test_the_old_default_is_raised(self, tmp_path: Path) -> None:
        """`save_settings` writes every field, so an existing install carries
        512 explicitly and would keep truncating regardless of the new
        default."""
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"schema_version": 3, "conversation": {"max_tokens": 512}}))
        loaded = load_settings(path)
        assert loaded.conversation.max_tokens == 2048
        assert loaded.schema_version == SETTINGS_SCHEMA_VERSION

    def test_a_hand_tuned_value_is_left_alone(self, tmp_path: Path) -> None:
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"schema_version": 3, "conversation": {"max_tokens": 256}}))
        assert load_settings(path).conversation.max_tokens == 256

    def test_the_cap_still_leaves_room_for_prompt_and_history(self) -> None:
        settings = Settings()
        assert settings.conversation.max_tokens < settings.llm.providers.local.context_length // 2
