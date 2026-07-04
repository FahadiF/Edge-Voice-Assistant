"""In-memory conversation history with turn windowing.

M2 scope: system prompt + a sliding window of recent turns, composed into the
LLM message list. Persistence and summarization arrive in M4 behind the
MemoryStore port; the composition point is already isolated here so that change
will not touch the orchestrator.
"""

from __future__ import annotations

from eva.llm.base import ChatMessage


class ConversationHistory:
    def __init__(self, system_prompt: str, max_turns: int = 20) -> None:
        self._system_prompt = system_prompt
        self._max_turns = max_turns
        self._turns: list[tuple[str, str]] = []  # (user, assistant)

    def add_turn(self, user_text: str, assistant_text: str) -> None:
        self._turns.append((user_text, assistant_text))
        if len(self._turns) > self._max_turns:
            del self._turns[: len(self._turns) - self._max_turns]

    def messages(self, user_text: str) -> list[ChatMessage]:
        """Compose the message list for the next generation."""
        messages = [ChatMessage(role="system", content=self._system_prompt)]
        for user, assistant in self._turns:
            messages.append(ChatMessage(role="user", content=user))
            messages.append(ChatMessage(role="assistant", content=assistant))
        messages.append(ChatMessage(role="user", content=user_text))
        return messages

    def clear(self) -> None:
        self._turns.clear()

    @property
    def turn_count(self) -> int:
        return len(self._turns)
