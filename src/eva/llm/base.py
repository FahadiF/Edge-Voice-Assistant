"""LLM engine port.

`stream()` is a blocking generator executed in a worker thread by the
orchestrator; tokens are handed to the asyncio side as they arrive.
Cancellation contract: implementations MUST call `should_abort()` at least once
per generated token and stop promptly when it returns True — this is what makes
barge-in cut generation mid-sentence instead of finishing a stale reply.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from typing import Literal

from pydantic import BaseModel, ConfigDict

from eva.core.errors import InvalidChatSequenceError
from eva.core.events import FinishReason as FinishReason
from eva.core.tools import ToolCall


class ChatMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    """Calls this assistant message issued, if any.

    Neutral values, not a provider's wire fields: adapters render them into
    whatever markup their template expects. Present on the message rather
    than tracked separately because it is what makes a `tool` message
    verifiable — without it, a tool result is only positionally plausible.
    """


def validate_chat_messages(messages: list[ChatMessage]) -> None:
    """Enforce the chat-format contract every template-based chat engine
    needs — Qwen, Llama, and Mistral's GGUF-embedded Jinja templates all
    reject a message list that isn't: exactly one system message, first,
    then strictly alternating user/assistant turns. This is a generic
    contract (no model-specific logic), so one validator call at message
    composition time (`ContextBuilder.build()`) protects every current and
    future `LLMEngine` adapter without each adapter needing its own check.
    """
    if not messages:
        raise InvalidChatSequenceError("Message list must not be empty")
    if messages[0].role != "system":
        raise InvalidChatSequenceError(
            f"The first message must have role 'system', got '{messages[0].role}'"
        )
    expected: tuple[Literal["user", "assistant", "tool"], ...] = ("user",)
    previous: ChatMessage | None = None
    for message in messages[1:]:
        if message.role == "system":
            raise InvalidChatSequenceError(
                "Only one system message is allowed, and it must be first"
            )
        if message.role not in expected:
            expected_str = " or ".join(f"'{e}'" for e in expected)
            raise InvalidChatSequenceError(
                f"Invalid role sequence; expected {expected_str} but got '{message.role}'"
            )
        # Adjacency alone would accept a tool result after ordinary assistant
        # prose, which no provider can render: there is no call for it to
        # answer. The assistant turn it follows must actually have issued one.
        if message.role == "tool" and previous is not None:
            issuing = previous.role == "assistant" and bool(previous.tool_calls)
            if not (issuing or previous.role == "tool"):
                raise InvalidChatSequenceError(
                    "A 'tool' message must follow an assistant message that issued "
                    "tool calls, or another 'tool' message"
                )

        if message.role == "user":
            expected = ("assistant",)
        elif message.role == "assistant":
            expected = ("user", "tool")
        else:
            expected = ("tool", "assistant")
        previous = message


class GenerationParams(BaseModel):
    model_config = ConfigDict(frozen=True)

    temperature: float = 0.4
    top_p: float = 0.9
    max_tokens: int = 512
    stop: tuple[str, ...] = ()


class GenerationOutcome(BaseModel):
    """How one generation pass ended.

    Returned rather than yielded so ordinary streaming keeps its `str` element
    type. That is also what keeps provider markup out of the speech path: an
    adapter that recognises a tool call yields no text at all and reports it
    here, instead of emitting markup a downstream filter would have to strip.
    """

    model_config = ConfigDict(frozen=True)

    reason: FinishReason
    tool_calls: tuple[ToolCall, ...] = ()
    """Non-empty only when `reason` is `tool_calls`."""


class LLMEngine(ABC):
    device: str = "unloaded"
    """Device the model actually landed on ("cuda"/"cpu"); set by load()."""

    @abstractmethod
    def load(self) -> None:
        """Load model weights. Idempotent."""

    @abstractmethod
    def unload(self) -> None:
        """Release model resources (hot-swap support)."""

    def count_tokens(self, text: str) -> int:
        """Token count for `text` under this model's tokenizer.

        The default is a coarse chars/4 estimate so adapters need not
        implement it; engines that can tokenize exactly should override.
        Used by the Context Builder to keep the assembled prompt plus the
        generation allowance inside the model's context window.
        """
        return max(1, len(text) // 4)

    @abstractmethod
    def stream(
        self,
        messages: list[ChatMessage],
        params: GenerationParams,
        should_abort: Callable[[], bool],
    ) -> Generator[str, None, GenerationOutcome]:
        """Yield response text incrementally; honor `should_abort` per token.

        RETURNS (via `StopIteration.value`, i.e. `return` in the generator
        body) the `GenerationOutcome`. Callers that only iterate keep working;
        a generator with a bare `return` yields `None`, which callers treat as
        an ordinary `stop` with no tool calls.
        """
