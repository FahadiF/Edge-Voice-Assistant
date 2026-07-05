"""LLM engine port.

`stream()` is a blocking generator executed in a worker thread by the
orchestrator; tokens are handed to the asyncio side as they arrive.
Cancellation contract: implementations MUST call `should_abort()` at least once
per generated token and stop promptly when it returns True — this is what makes
barge-in cut generation mid-sentence instead of finishing a stale reply.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from typing import Literal

from pydantic import BaseModel, ConfigDict

from eva.core.errors import InvalidChatSequenceError


class ChatMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Literal["system", "user", "assistant", "tool"]
    content: str


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
    expected: Literal["user", "assistant"] = "user"
    for message in messages[1:]:
        if message.role == "system":
            raise InvalidChatSequenceError(
                "Only one system message is allowed, and it must be first"
            )
        if message.role != expected:
            raise InvalidChatSequenceError(
                "Messages after the system message must strictly alternate "
                f"user/assistant starting with 'user'; expected '{expected}' "
                f"but got '{message.role}'"
            )
        expected = "assistant" if expected == "user" else "user"


class GenerationParams(BaseModel):
    model_config = ConfigDict(frozen=True)

    temperature: float = 0.4
    top_p: float = 0.9
    max_tokens: int = 512
    stop: tuple[str, ...] = ()


class LLMEngine(ABC):
    device: str = "unloaded"
    """Device the model actually landed on ("cuda"/"cpu"); set by load()."""

    @abstractmethod
    def load(self) -> None:
        """Load model weights. Idempotent."""

    @abstractmethod
    def unload(self) -> None:
        """Release model resources (hot-swap support)."""

    @abstractmethod
    def stream(
        self,
        messages: list[ChatMessage],
        params: GenerationParams,
        should_abort: Callable[[], bool],
    ) -> Iterator[str]:
        """Yield response text incrementally; honor `should_abort` per token."""
