"""Context Builder (ADR-021): deterministic prompt composition.

A single, fixed-order composition class — not a registry (see ADR-021's
rationale: swappable composition policy would undermine the determinism and
inspectability this milestone explicitly asks for). It is built *from*
registry-resolved parts (persona, language, a `MemoryStore`, optionally a
`MemoryRetriever` + `EmbeddingProvider` + `UserProfileStore`), which
individually stay swappable.

Composition order, always: system prompt (persona + language + profile) →
retrieved relevant memories → latest conversation summary → recent-turn
window → the current utterance. Every build returns a `ContextTrace`
alongside the message list so the result is inspectable without spending a
generation on it (the API's context-preview endpoint, Part 12).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from eva.config.settings import Settings
from eva.conversation.language import resolve_language
from eva.conversation.personas import resolve_persona
from eva.embedding.base import EmbeddingProvider
from eva.llm.base import ChatMessage
from eva.memory.base import MemoryRetriever, MemoryStore, UserProfileStore
from eva.memory.models import MemorySearchResult, UserProfile


@dataclass(frozen=True)
class RetrievedMemoryTrace:
    turn_id: int
    score: float
    text_preview: str


@dataclass(frozen=True)
class ContextTrace:
    persona_id: str
    profile_id: str | None
    language_code: str
    retrieved_memories: tuple[RetrievedMemoryTrace, ...]
    summary_included: bool
    summary_text_preview: str | None
    recent_turn_count: int
    trimmed_sections: tuple[str, ...]


@dataclass(frozen=True)
class BuiltContext:
    messages: list[ChatMessage]
    trace: ContextTrace


class ContextBuilder:
    def __init__(
        self,
        settings: Settings,
        memory: MemoryStore,
        *,
        retriever: MemoryRetriever | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        profile_store: UserProfileStore | None = None,
    ) -> None:
        self._settings = settings
        self._memory = memory
        self._retriever = retriever
        self._embedding_provider = embedding_provider
        self._profile_store = profile_store
        self._last_retrieval_ms: int | None = None
        self._last_retrieval_top_score: float | None = None

    @property
    def last_retrieval_ms(self) -> int | None:
        """Wall-clock time of the most recent semantic retrieval (embed +
        search) — diagnostics (ADR-019 §11). None until the first retrieval."""
        return self._last_retrieval_ms

    @property
    def last_retrieval_top_score(self) -> float | None:
        """Top result's score from the most recent retrieval, or None if
        nothing was retrieved (no retriever configured, or no matches)."""
        return self._last_retrieval_top_score

    def build(self, conversation_id: str, user_text: str) -> BuiltContext:
        language = resolve_language(self._settings)
        persona = resolve_persona(self._settings)
        profile = self._profile_store.active() if self._profile_store is not None else None
        trimmed_sections: list[str] = []

        system_prompt = self._compose_system_prompt(persona.system_prompt, language.prompt_note)
        if profile is not None:
            system_prompt = self._apply_profile_preferences(system_prompt, profile)

        results, memory_trace = self._retrieve_memories(user_text)
        memory_block, memory_trimmed = self._format_memory_block(results)
        if memory_trimmed:
            trimmed_sections.append("relevant_memories")

        summary = self._memory.latest_summary(conversation_id)
        summary_text: str | None = None
        if summary is not None:
            summary_text = summary.text
            budget = self._settings.memory.max_summary_chars
            if len(summary_text) > budget:
                summary_text = summary_text[:budget]
                trimmed_sections.append("summary")

        recent_turns = self._memory.recent_turns(
            conversation_id, self._settings.conversation.max_history_turns
        )

        messages: list[ChatMessage] = [ChatMessage(role="system", content=system_prompt)]
        if memory_block:
            messages.append(ChatMessage(role="system", content=memory_block))
        if summary_text:
            messages.append(
                ChatMessage(
                    role="system", content=f"Earlier conversation summary: {summary_text}"
                )
            )
        for turn in recent_turns:
            messages.append(ChatMessage(role=turn.speaker, content=turn.text))
        messages.append(ChatMessage(role="user", content=user_text))

        trace = ContextTrace(
            persona_id=persona.id,
            profile_id=profile.id if profile is not None else None,
            language_code=language.code,
            retrieved_memories=tuple(memory_trace),
            summary_included=summary_text is not None,
            summary_text_preview=summary_text[:80] if summary_text else None,
            recent_turn_count=len(recent_turns),
            trimmed_sections=tuple(trimmed_sections),
        )
        return BuiltContext(messages=messages, trace=trace)

    def _compose_system_prompt(self, persona_prompt: str, language_note: str) -> str:
        if language_note:
            return f"{persona_prompt} {language_note}"
        return persona_prompt

    def _apply_profile_preferences(self, system_prompt: str, profile: UserProfile) -> str:
        preferences: list[str] = []
        if profile.nickname:
            preferences.append(f"The user's name is {profile.nickname}.")
        if profile.conversation_style:
            preferences.append(f"Preferred conversation style: {profile.conversation_style}.")
        preferences.append(f"Use {profile.units} units.")
        if not preferences:
            return system_prompt
        return f"{system_prompt} {' '.join(preferences)}"

    def _retrieve_memories(
        self, user_text: str
    ) -> tuple[list[MemorySearchResult], list[RetrievedMemoryTrace]]:
        if (
            self._retriever is None
            or self._embedding_provider is None
            or not self._settings.memory.embedding_enabled
        ):
            return [], []
        start = time.perf_counter()
        query_vector = self._embedding_provider.embed(user_text).tobytes()
        # Searches every conversation, not just the active one — semantic
        # memory is meant to recall *past* conversations (Part 3: "related
        # conversations", "similar memories"), not just the current session's
        # recent-turn window (which `recent_turns` below already covers).
        results = self._retriever.retrieve(
            query_vector,
            top_k=self._settings.memory.retrieval_top_k,
            conversation_id=None,
        )
        self._last_retrieval_ms = int((time.perf_counter() - start) * 1000)
        self._last_retrieval_top_score = results[0].score if results else None
        trace = [
            RetrievedMemoryTrace(
                turn_id=r.turn.id, score=r.score, text_preview=r.turn.text[:80]
            )
            for r in results
            if r.turn.id is not None
        ]
        return results, trace

    def _format_memory_block(self, results: list[MemorySearchResult]) -> tuple[str, bool]:
        if not results:
            return "", False
        lines = [f"- {r.turn.text}" for r in results]
        block = "Potentially relevant earlier context:\n" + "\n".join(lines)
        budget = self._settings.memory.max_memory_chars
        if len(block) > budget:
            return block[:budget], True
        return block, False
