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


class ChatMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Literal["system", "user", "assistant", "tool"]
    content: str


class GenerationParams(BaseModel):
    model_config = ConfigDict(frozen=True)

    temperature: float = 0.4
    top_p: float = 0.9
    max_tokens: int = 512
    stop: tuple[str, ...] = ()


class LLMEngine(ABC):
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
