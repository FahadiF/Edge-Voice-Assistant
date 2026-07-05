"""LLMSummarizer tests (ADR-019 §9) — fake LLM, no models required."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import UTC, datetime

from eva.llm.base import ChatMessage, GenerationParams, LLMEngine
from eva.memory.models import MemoryTurn
from eva.memory.summarizer import LLMSummarizer


class _FakeLLM(LLMEngine):
    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.received_messages: list[ChatMessage] | None = None

    def load(self) -> None: ...
    def unload(self) -> None: ...

    def stream(
        self,
        messages: list[ChatMessage],
        params: GenerationParams,
        should_abort: Callable[[], bool],
    ) -> Iterator[str]:
        self.received_messages = messages
        yield from self.tokens


def _turn(speaker: str, text: str) -> MemoryTurn:
    return MemoryTurn(
        conversation_id="c1", created_at=datetime.now(UTC), speaker=speaker, text=text
    )


def test_summarize_joins_streamed_tokens() -> None:
    llm = _FakeLLM(["A ", "short ", "summary."])
    summarizer = LLMSummarizer(llm)
    result = summarizer.summarize([_turn("user", "hi"), _turn("assistant", "hello")])
    assert result == "A short summary."


def test_summarize_empty_turns_returns_empty_without_calling_llm() -> None:
    llm = _FakeLLM(["should not be used"])
    summarizer = LLMSummarizer(llm)
    result = summarizer.summarize([])
    assert result == ""
    assert llm.received_messages is None


def test_summarize_includes_transcript_with_speaker_labels() -> None:
    llm = _FakeLLM(["ok"])
    summarizer = LLMSummarizer(llm)
    summarizer.summarize([_turn("user", "what's the weather"), _turn("assistant", "sunny")])
    assert llm.received_messages is not None
    transcript_message = llm.received_messages[1]
    assert transcript_message.role == "user"
    assert "user: what's the weather" in transcript_message.content
    assert "assistant: sunny" in transcript_message.content


def test_summarize_never_mutates_input_turns() -> None:
    llm = _FakeLLM(["summary"])
    summarizer = LLMSummarizer(llm)
    turns = [_turn("user", "original text")]
    snapshot = turns[0].model_copy()
    summarizer.summarize(turns)
    assert turns[0] == snapshot


def test_summarize_strips_whitespace() -> None:
    llm = _FakeLLM(["  padded summary  \n"])
    summarizer = LLMSummarizer(llm)
    result = summarizer.summarize([_turn("user", "hi")])
    assert result == "padded summary"
