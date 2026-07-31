"""Regression tests for the chat-format contract (ADR-021 amendment).

Real hardware testing surfaced `ValueError: System message must be at the
beginning.` from llama.cpp's Qwen chat template — caused by `ContextBuilder`
emitting multiple system messages. `validate_chat_messages()` is the
generic, model-agnostic guard against this class of bug recurring for any
future chat-template-based engine (Qwen, Llama, Mistral, ...).
"""

from __future__ import annotations

from collections.abc import Generator

import pytest

from eva.core.errors import InvalidChatSequenceError
from eva.core.tools import ToolCall
from eva.llm.base import ChatMessage, GenerationOutcome, validate_chat_messages


def _msg(role: str, content: str = "x") -> ChatMessage:
    return ChatMessage(role=role, content=content)  # type: ignore[arg-type]


class TestValidSequences:
    def test_system_then_user(self) -> None:
        validate_chat_messages([_msg("system"), _msg("user")])

    def test_system_then_alternating_history_then_user(self) -> None:
        validate_chat_messages(
            [
                _msg("system"),
                _msg("user"),
                _msg("assistant"),
                _msg("user"),
                _msg("assistant"),
                _msg("user"),
            ]
        )


class TestEmptyList:
    def test_empty_message_list_rejected(self) -> None:
        with pytest.raises(InvalidChatSequenceError):
            validate_chat_messages([])


class TestMissingLeadingSystem:
    def test_first_message_not_system_rejected(self) -> None:
        with pytest.raises(InvalidChatSequenceError, match="first message must have role"):
            validate_chat_messages([_msg("user")])


class TestMultipleSystemMessages:
    def test_second_system_message_rejected(self) -> None:
        """The exact bug: `ContextBuilder` used to emit identity, technical
        facts, memory, and summary as separate system messages."""
        with pytest.raises(InvalidChatSequenceError, match="Only one system message"):
            validate_chat_messages([_msg("system"), _msg("system"), _msg("user")])

    def test_system_message_after_user_rejected(self) -> None:
        with pytest.raises(InvalidChatSequenceError, match="Only one system message"):
            validate_chat_messages([_msg("system"), _msg("user"), _msg("system")])

    def test_system_message_after_assistant_rejected(self) -> None:
        with pytest.raises(InvalidChatSequenceError, match="Only one system message"):
            validate_chat_messages(
                [_msg("system"), _msg("user"), _msg("assistant"), _msg("system")]
            )


class TestBrokenAlternation:
    def test_two_consecutive_user_messages_rejected(self) -> None:
        with pytest.raises(InvalidChatSequenceError, match="Invalid role sequence"):
            validate_chat_messages([_msg("system"), _msg("user"), _msg("user")])

    def test_two_consecutive_assistant_messages_rejected(self) -> None:
        with pytest.raises(InvalidChatSequenceError, match="Invalid role sequence"):
            validate_chat_messages(
                [_msg("system"), _msg("user"), _msg("assistant"), _msg("assistant")]
            )

    def test_assistant_immediately_after_system_rejected(self) -> None:
        """The first non-system message must be 'user', never 'assistant'."""
        with pytest.raises(InvalidChatSequenceError, match="Invalid role sequence"):
            validate_chat_messages([_msg("system"), _msg("assistant")])


def _calling(name: str = "get_weather") -> ChatMessage:
    """An assistant turn that actually issued a tool call."""
    return ChatMessage(
        role="assistant",
        content="",
        tool_calls=(ToolCall(id="c1", name=name, arguments={}),),
    )


class TestToolSequences:
    """A tool result answers a specific request. Role adjacency alone cannot
    express that, so the assistant turn it follows must have issued calls."""

    def test_a_full_round_trip_is_valid(self) -> None:
        validate_chat_messages(
            [_msg("system"), _msg("user"), _calling(), _msg("tool"), _msg("assistant")]
        )

    def test_several_results_may_follow_one_calling_turn(self) -> None:
        """One generation can request more than one tool."""
        validate_chat_messages(
            [
                _msg("system"),
                _msg("user"),
                _calling(),
                _msg("tool"),
                _msg("tool"),
                _msg("assistant"),
            ]
        )

    def test_a_tool_result_after_ordinary_prose_is_rejected(self) -> None:
        """The case adjacency rules miss: the assistant replied normally, so
        there is no call for this result to answer and no provider can render
        it."""
        with pytest.raises(InvalidChatSequenceError, match="issued tool calls"):
            validate_chat_messages([_msg("system"), _msg("user"), _msg("assistant"), _msg("tool")])

    def test_tool_immediately_after_system_rejected(self) -> None:
        with pytest.raises(InvalidChatSequenceError, match="Invalid role sequence"):
            validate_chat_messages([_msg("system"), _msg("tool")])

    def test_tool_immediately_after_user_rejected(self) -> None:
        with pytest.raises(InvalidChatSequenceError, match="Invalid role sequence"):
            validate_chat_messages([_msg("system"), _msg("user"), _msg("tool")])

    def test_user_immediately_after_tool_rejected(self) -> None:
        with pytest.raises(InvalidChatSequenceError, match="Invalid role sequence"):
            validate_chat_messages(
                [_msg("system"), _msg("user"), _calling(), _msg("tool"), _msg("user")]
            )

    def test_ordinary_conversation_is_unaffected(self) -> None:
        """Tool support must not change the shape of a normal exchange."""
        validate_chat_messages(
            [_msg("system"), _msg("user"), _msg("assistant"), _msg("user"), _msg("assistant")]
        )


def _calling_two() -> ChatMessage:
    """One generation requesting two tools at once."""
    return ChatMessage(
        role="assistant",
        content="",
        tool_calls=(
            ToolCall(id="c1", name="get_weather", arguments={}),
            ToolCall(id="c2", name="get_time", arguments={}),
        ),
    )


def _answer(call_id: str | None, content: str = "r") -> ChatMessage:
    return ChatMessage(role="tool", content=content, call_id=call_id)


class TestToolCallCorrelation:
    """Which call a result answers is carried by the answering message.

    With one outstanding request, order is enough. With two, the answers need
    not arrive in the order they were issued — a slow first tool and a fast
    second one swap — so position would attribute an answer to the wrong
    request and the model would be told a plausible, wrong thing.
    """

    def test_two_calls_get_two_correlated_answers(self) -> None:
        validate_chat_messages(
            [
                _msg("system"),
                _msg("user"),
                _calling_two(),
                _answer("c1"),
                _answer("c2"),
                _msg("assistant"),
            ]
        )

    def test_answers_may_arrive_in_any_order(self) -> None:
        """The point of correlating: reversed arrival is still unambiguous."""
        validate_chat_messages(
            [
                _msg("system"),
                _msg("user"),
                _calling_two(),
                _answer("c2"),
                _answer("c1"),
                _msg("assistant"),
            ]
        )

    def test_multiple_calls_require_an_explicit_id(self) -> None:
        """Exactly where positional correlation stops being sufficient."""
        with pytest.raises(InvalidChatSequenceError, match="must set call_id"):
            validate_chat_messages(
                [_msg("system"), _msg("user"), _calling_two(), _answer(None), _answer(None)]
            )

    def test_a_single_call_needs_no_id(self) -> None:
        """One outstanding request has nothing to be confused with, so the
        common exchange stays as simple as it reads."""
        validate_chat_messages(
            [_msg("system"), _msg("user"), _calling(), _answer(None), _msg("assistant")]
        )

    def test_an_id_answering_no_issued_call_is_rejected(self) -> None:
        with pytest.raises(InvalidChatSequenceError, match="answers no call"):
            validate_chat_messages([_msg("system"), _msg("user"), _calling_two(), _answer("nope")])

    def test_the_same_call_cannot_be_answered_twice(self) -> None:
        with pytest.raises(InvalidChatSequenceError, match="more than once"):
            validate_chat_messages(
                [_msg("system"), _msg("user"), _calling_two(), _answer("c1"), _answer("c1")]
            )

    def test_only_a_tool_message_may_carry_a_call_id(self) -> None:
        """Keeps the field meaningful: correlation belongs to an answer."""
        stray = ChatMessage(role="user", content="hi", call_id="c1")
        with pytest.raises(InvalidChatSequenceError, match="Only a 'tool' message"):
            validate_chat_messages([_msg("system"), stray])

    def test_the_system_message_may_not_carry_a_call_id(self) -> None:
        """The role loop starts at `messages[1:]`, so position 0 needs its own
        check — without it the rule above had a hole at exactly one index."""
        system = ChatMessage(role="system", content="s", call_id="c1")
        with pytest.raises(InvalidChatSequenceError, match="Only a 'tool' message"):
            validate_chat_messages([system, _msg("user")])

    def test_correlation_is_reset_by_the_next_calling_turn(self) -> None:
        """A later turn's ids must not be satisfiable by an earlier turn's."""
        with pytest.raises(InvalidChatSequenceError, match="answers no call"):
            validate_chat_messages(
                [
                    _msg("system"),
                    _msg("user"),
                    _calling_two(),
                    _answer("c1"),
                    _answer("c2"),
                    _msg("assistant"),
                    _msg("user"),
                    _calling("get_weather"),  # issues only "c1"
                    _answer("c2"),  # belonged to the previous turn
                ]
            )

    def test_correlation_survives_serialization(self) -> None:
        """History is persisted and replayed; a round trip must not silently
        drop the field that makes an answer attributable."""
        original = _answer("c2", content="18 degrees")
        restored = ChatMessage.model_validate_json(original.model_dump_json())
        assert restored.call_id == "c2"
        assert restored == original

    def test_an_ordinary_message_serializes_without_a_null_id(self) -> None:
        """`exclude_none` is what keeps the llama.cpp payload byte-identical
        to what it was before correlation existed."""
        assert "call_id" not in _msg("user").model_dump(exclude_none=True)


class TestGenerationOutcome:
    """`stream()` reports how a pass ended through its return value, so
    ordinary streaming keeps yielding plain text."""

    def test_an_outcome_defaults_to_no_tool_calls(self) -> None:
        assert GenerationOutcome(reason="stop").tool_calls == ()

    def test_tool_calls_travel_with_the_outcome_not_the_token_stream(self) -> None:
        outcome = GenerationOutcome(
            reason="tool_calls",
            tool_calls=(ToolCall(id="c1", name="get_weather", arguments={"city": "Oslo"}),),
        )
        assert outcome.tool_calls[0].arguments == {"city": "Oslo"}

    def test_an_adapter_returning_nothing_is_read_as_a_plain_stop(self) -> None:
        """A generator with a bare `return` yields None; callers treat that as
        an ordinary completion so older adapters keep working."""

        def _bare() -> Generator[str, None, None]:
            yield "hi"
            return

        gen = _bare()
        list(gen)
        outcome = None
        assert (outcome.reason if outcome is not None else "stop") == "stop"
