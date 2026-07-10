"""Context Builder (ADR-021): deterministic prompt composition.

A single, fixed-order composition class — not a registry (see ADR-021's
rationale: swappable composition policy would undermine the determinism and
inspectability this milestone explicitly asks for). It is built *from*
registry-resolved parts (persona, language, a `MemoryStore`, optionally a
`MemoryRetriever` + `EmbeddingProvider` + `UserProfileStore`), which
individually stay swappable.

Composition order, always: ONE system message (identity + persona + language
+ profile preferences + technical backend facts + retrieved relevant
memories + latest conversation summary, all merged into one string) →
recent-turn window (strictly alternating user/assistant) → the current
utterance. Every build returns a `ContextTrace` alongside the message list
so the result is inspectable without spending a generation on it (the API's
context-preview endpoint, Part 12).

Exactly one system message, first, is a hard requirement — llama.cpp's
chat-template engine (Qwen, Llama, Mistral, ...) rejects a second system
message anywhere in the list (ADR-021 amendment). `validate_chat_messages()`
(`eva.llm.base`) enforces this on every `build()` call.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass

from eva.config.settings import Settings
from eva.conversation.language import resolve_language
from eva.conversation.personas import resolve_persona
from eva.conversation.system_info import system_facts_block
from eva.embedding.base import EmbeddingProvider
from eva.llm.base import ChatMessage, validate_chat_messages
from eva.memory.base import MemoryRetriever, MemoryStore, UserProfileStore
from eva.memory.models import MemorySearchResult, UserProfile

logger = logging.getLogger(__name__)

# System-prompt building blocks (ADR-021 Amendment 3). Hierarchy: identity
# (one sentence, name used sparingly) → how to converse → capability honesty
# → persona style → language → profile. Ordered so behavior guidance
# dominates and self-description is de-emphasized — manual testing showed a
# small LLM with a heavy identity block kept talking about itself ("I am
# not a spreadsheet") instead of helping.

_IDENTITY_PREAMBLE = (
    "You are Edge Voice Assistant, a private assistant that runs entirely "
    "on this device. Mention your own name only when the user asks who you "
    "are — never work it into ordinary replies, and don't describe what "
    "you are unless asked."
)

_CONVERSATION_GUIDANCE = (
    "This is a flowing spoken conversation, not isolated questions. Short "
    "fragments, pronouns (it, that, this, them), and incomplete sentences "
    'continue the current topic — if the user says "with rows and '
    'columns" right after discussing a table, extend that table; never '
    "treat a follow-up as an unrelated request. Focus on accomplishing "
    "what the user is trying to do rather than explaining what you are or "
    "are not. Anything expressible in text — lists, tables, structured "
    "data, step-by-step plans, calculations — you can and should produce; "
    "never refuse on the grounds of not being that kind of tool. When a "
    "request is ambiguous, make the most helpful reasonable assumption, or "
    "ask one short clarifying question. Stay concise by default; expand "
    "when detail genuinely helps."
)

_CAPABILITY_GUIDANCE = (
    "This build works with voice and text only. Some capabilities (for "
    "example image understanding, internet access, or reading local files) "
    "are planned for the platform but not enabled in this build — when "
    "asked about one, say it is not enabled in the current build rather "
    "than claiming it is impossible. The user also controls permissions in "
    "Settings: if a local fact (like the time or hardware details) is not "
    "listed in your system information below, the user has not granted "
    "that permission — say so, rather than saying you can never know it."
)


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

        # Every chat-template-based engine (Qwen, Llama, Mistral, ...) rejects
        # more than one system message, or one appearing anywhere but first
        # (ADR-021 Amendment 2) — so everything folds into ONE system
        # message. Section order is a deliberate hierarchy (Amendment 3):
        # identity/behavior/persona first (dominates tone), then this
        # conversation's summary (continuity), then cross-conversation
        # memories (background knowledge), then technical facts last (rarely
        # relevant — low salience keeps the model from volunteering them).
        system_sections = [system_prompt]
        if summary_text:
            system_sections.append(
                f"Summary of the earlier part of this conversation: {summary_text}"
            )
        if memory_block:
            system_sections.append(memory_block)
        # Permission-gated local facts (M5.3, ADR-025) — fresh every turn so
        # the date/time is current; omitted entirely when nothing is allowed.
        facts = system_facts_block(self._settings.permissions)
        if facts:
            system_sections.append(facts)
        system_sections.append(self._technical_facts_block())
        combined_system_prompt = "\n\n".join(system_sections)

        turn_pairs = [(turn.speaker, turn.text) for turn in recent_turns]
        turn_pairs.append(("user", user_text))
        messages: list[ChatMessage] = [
            ChatMessage(role="system", content=combined_system_prompt),
            *(
                ChatMessage(role=role, content=content)
                for role, content in self._normalize_alternation(turn_pairs)
            ),
        ]
        validate_chat_messages(messages)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Composed chat messages: %s",
                [(m.role, len(m.content)) for m in messages],
            )

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

    def _normalize_alternation(
        self, turn_pairs: Sequence[tuple[str, str]]
    ) -> list[tuple[str, str]]:
        """Merge consecutive same-role turns into one message.

        `MemoryStore.recent_turns()` almost always alternates user/assistant
        (the orchestrator only ever writes a matched pair together), but
        nothing in the store enforces that — an imported snapshot, a future
        plugin, or a dangling unanswered turn could leave two turns from the
        same speaker adjacent. Rather than let that reach
        `validate_chat_messages()` and hard-fail the turn, merge same-role
        neighbors by joining their text — the chat-template contract (one
        system message, then strict user/assistant alternation) still holds,
        and no conversation content is dropped."""
        merged: list[tuple[str, str]] = []
        for role, content in turn_pairs:
            if merged and merged[-1][0] == role:
                merged[-1] = (role, f"{merged[-1][1]}\n{content}")
            else:
                merged.append((role, content))
        return merged

    def _compose_system_prompt(self, persona_prompt: str, language_note: str) -> str:
        """Identity → conversational behavior → capability honesty → persona
        style → language. Behavior before persona so continuity/helpfulness
        rules hold for every persona; persona after so its voice is the last
        (most salient) style instruction (ADR-021 Amendment 3)."""
        parts = [
            _IDENTITY_PREAMBLE,
            _CONVERSATION_GUIDANCE,
            _CAPABILITY_GUIDANCE,
            persona_prompt,
        ]
        if language_note:
            parts.append(language_note)
        return " ".join(parts)

    def _technical_facts_block(self) -> str:
        """Backend details the model may cite only when explicitly asked a
        technical question (see `_IDENTITY_PREAMBLE`) — a separate system
        message so identity/persona text never has to name a concrete
        model."""
        s = self._settings
        return (
            "Technical backend details (share only if explicitly asked): "
            f"LLM model = {s.llm.model}; ASR model = {s.asr.model}; "
            f"TTS model = {s.tts.model}; VAD engine = {s.vad.engine}."
        )

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
        start = time.perf_counter()
        if (
            self._retriever is None
            or self._embedding_provider is None
            or not self._settings.memory.embedding_enabled
        ):
            # No embedding model installed (or disabled): degrade to keyword
            # recall instead of NO recall (M5.4 — previously the memory block
            # was simply empty here, so long-term memory silently did nothing
            # on machines without the embedding model).
            results = self._keyword_fallback(user_text)
        else:
            query_vector = self._embedding_provider.embed(user_text).tobytes()
            # Searches every conversation, not just the active one — semantic
            # memory is meant to recall *past* conversations (Part 3: "related
            # conversations", "similar memories"), not just the current
            # session's recent-turn window (which `recent_turns` covers).
            results = self._retriever.retrieve(
                query_vector,
                top_k=self._settings.memory.retrieval_top_k,
                conversation_id=None,
            )
        self._last_retrieval_ms = int((time.perf_counter() - start) * 1000)
        self._last_retrieval_top_score = results[0].score if results else None
        trace = [
            RetrievedMemoryTrace(turn_id=r.turn.id, score=r.score, text_preview=r.turn.text[:80])
            for r in results
            if r.turn.id is not None
        ]
        return results, trace

    def _keyword_fallback(self, user_text: str) -> list[MemorySearchResult]:
        """Recall by keyword when semantic search isn't available: run the
        store's text search once per salient word (a whole-utterance phrase
        match would almost never hit) and merge, preserving first-seen order.
        Failures degrade to no results — recall must never break a turn."""
        words = [w.strip(".,!?;:'\"()") for w in user_text.split()]
        salient = [w for w in words if len(w) >= 4][:4]
        merged: dict[int, MemorySearchResult] = {}
        top_k = self._settings.memory.retrieval_top_k
        for word in salient:
            try:
                for result in self._memory.search_text(word, limit=top_k):
                    if result.turn.id is not None and result.turn.id not in merged:
                        merged[result.turn.id] = result
            except Exception:
                logger.debug("Keyword fallback search failed for %r", word, exc_info=True)
        return list(merged.values())[:top_k]

    def _format_memory_block(self, results: list[MemorySearchResult]) -> tuple[str, bool]:
        """Retrieved memories, highest-scored first (the retriever's order is
        preserved). Framed as things the assistant *remembers* — to be woven
        in naturally when relevant — not as a document to recite (ADR-021
        Amendment 3; manual testing found mechanical 'according to earlier
        context' phrasing)."""
        if not results:
            return "", False
        lines = [f"- {r.turn.text}" for r in results]
        block = (
            "You remember these things from earlier conversations. Use them "
            "naturally when relevant — don't announce that you are recalling "
            "them, and ignore any that don't apply:\n" + "\n".join(lines)
        )
        budget = self._settings.memory.max_memory_chars
        if len(block) > budget:
            return block[:budget], True
        return block, False
