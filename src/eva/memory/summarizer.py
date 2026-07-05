"""LLM-backed conversation summarization (ADR-019 §9).

Reuses the existing `LLMEngine` port — no new ML dependency for
summarization. `summarize()` never mutates or deletes the turns it
summarizes; the caller is responsible for storing the result via
`MemoryStore.add_summary()` (summaries are additive, originals are kept).
"""

from __future__ import annotations

from eva.llm.base import ChatMessage, GenerationParams, LLMEngine
from eva.memory.base import Summarizer
from eva.memory.models import MemoryTurn

_SUMMARIZER_SYSTEM_PROMPT = (
    "You summarize a conversation transcript concisely for future reference. "
    "Capture key facts, decisions, and preferences the user expressed. "
    "Write two to four sentences. Do not add commentary, apologies, or "
    "meta-remarks about summarizing."
)


class LLMSummarizer(Summarizer):
    def __init__(self, llm: LLMEngine, *, max_tokens: int = 200) -> None:
        self._llm = llm
        self._params = GenerationParams(temperature=0.2, max_tokens=max_tokens)

    def summarize(self, turns: list[MemoryTurn]) -> str:
        if not turns:
            return ""
        transcript = "\n".join(f"{turn.speaker}: {turn.text}" for turn in turns)
        messages = [
            ChatMessage(role="system", content=_SUMMARIZER_SYSTEM_PROMPT),
            ChatMessage(role="user", content=transcript),
        ]
        parts = list(self._llm.stream(messages, self._params, should_abort=lambda: False))
        return "".join(parts).strip()
