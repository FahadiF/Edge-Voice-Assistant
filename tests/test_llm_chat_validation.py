"""Regression tests for the chat-format contract (ADR-021 amendment).

Real hardware testing surfaced `ValueError: System message must be at the
beginning.` from llama.cpp's Qwen chat template — caused by `ContextBuilder`
emitting multiple system messages. `validate_chat_messages()` is the
generic, model-agnostic guard against this class of bug recurring for any
future chat-template-based engine (Qwen, Llama, Mistral, ...).
"""

from __future__ import annotations

import pytest

from eva.core.errors import InvalidChatSequenceError
from eva.llm.base import ChatMessage, validate_chat_messages


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
        with pytest.raises(InvalidChatSequenceError, match="strictly alternate"):
            validate_chat_messages([_msg("system"), _msg("user"), _msg("user")])

    def test_two_consecutive_assistant_messages_rejected(self) -> None:
        with pytest.raises(InvalidChatSequenceError, match="strictly alternate"):
            validate_chat_messages(
                [_msg("system"), _msg("user"), _msg("assistant"), _msg("assistant")]
            )

    def test_assistant_immediately_after_system_rejected(self) -> None:
        """The first non-system message must be 'user', never 'assistant'."""
        with pytest.raises(InvalidChatSequenceError, match="strictly alternate"):
            validate_chat_messages([_msg("system"), _msg("assistant")])
