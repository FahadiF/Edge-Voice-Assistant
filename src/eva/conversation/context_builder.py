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
from collections.abc import Callable, Mapping, Sequence
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
    "This is a flowing conversation, not isolated questions. Short "
    "fragments, pronouns (it, that, this, them), and incomplete sentences "
    'continue the current topic — if the user says "with rows and '
    'columns" right after discussing a table, extend that table; never '
    "treat a follow-up as an unrelated request. Focus on accomplishing "
    "what the user is trying to do rather than explaining what you are or "
    "are not. Never refuse on the grounds of not being that kind of tool: "
    "anything expressible in words, you can produce. When a request is "
    "ambiguous, make the most helpful reasonable assumption, or ask one "
    "short clarifying question."
)

# Delivery-specific style. Exactly one of these is included per turn, chosen
# by `build(spoken=...)`. Splitting them is what lets the same assistant be
# terse and markdown-free out loud while still producing tables and fenced
# code on screen — one prompt cannot honestly ask for both.
_SPOKEN_STYLE = (
    "Your reply will be read aloud. Give the direct answer in one to three "
    "sentences and then stop — no preamble, no recap, no summary of what "
    "you just said. Go longer only when the user asks for more. Use plain "
    "flowing prose: no markdown, no headings, no bullet or numbered lists, "
    "and no tables unless the user explicitly asks for one. Never refer to "
    "anything on screen or tell the user to paste, click, type or look at "
    "something. Ask a follow-up question only when you genuinely need the "
    "answer to help — otherwise simply stop talking."
)

_TEXT_STYLE = (
    "Your reply is read on screen. Markdown is welcome where it genuinely "
    "helps — lists, tables and fenced code blocks are all fine. Stay "
    "concise by default; expand when detail genuinely helps."
)

_CAPABILITY_GUIDANCE = (
    "This build works with voice and text only. Some capabilities (for "
    "example image understanding, internet access, or reading local files) "
    "are planned for the platform but not enabled in this build — when "
    "asked about one, say it is not enabled in the current build rather "
    "than claiming it is impossible. The user also controls permissions in "
    "Settings: if a local fact (like the time or hardware details) is not "
    "listed in your system information below, the user has not granted "
    "that permission — say so, rather than saying you can never know it. "
    "Never state a personal detail about the user — their name above all — "
    "unless it is given to you here or earlier in this conversation; if you "
    "do not have it, say so plainly and ask, rather than guessing or "
    "inventing one. Do not contradict something you already established this "
    "conversation. The same rule covers the world, not just the user: you "
    "have no live data, so anything that changes over time — weather, news, "
    "prices, scores, standings, schedules, who currently holds a position — "
    "is something you do not know. Say that in one short sentence and stop. "
    "Do not estimate it, do not infer it from the date or location, do not "
    'hedge it with "probably" or "usually", and do not ask the user to '
    "supply it."
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
        runtime_devices: Callable[[], Mapping[str, str]] | None = None,
    ) -> None:
        self._settings = settings
        self._memory = memory
        self._retriever = retriever
        self._embedding_provider = embedding_provider
        self._profile_store = profile_store
        # Called once per build() rather than captured at construction: a
        # component reports "unloaded" until it has actually loaded (and
        # `tts.lazy_load` defers that past the first turns), so a snapshot
        # taken here would be wrong for the whole session.
        self._runtime_devices = runtime_devices
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

    def build(
        self,
        conversation_id: str,
        user_text: str,
        *,
        session_name: str | None = None,
        spoken: bool = False,
    ) -> BuiltContext:
        """Compose the turn's messages.

        `spoken` selects the delivery-style block: True for a turn that came
        in over the microphone and will be read aloud, False for typed input
        rendered on screen. It defaults to False so every existing caller
        (context preview, benchmarks) keeps the on-screen behaviour it had.
        """
        language = resolve_language(self._settings)
        persona = resolve_persona(self._settings)
        profile = self._profile_store.active() if self._profile_store is not None else None
        trimmed_sections: list[str] = []

        system_prompt = self._compose_system_prompt(
            persona.system_prompt, language.prompt_note, spoken=spoken
        )
        if profile is not None:
            system_prompt = self._apply_profile_preferences(system_prompt, profile)
        # The user's name is a durable fact, so it goes in the always-present
        # system prompt rather than being left to the recent-turn window or
        # query-dependent retrieval (which is what made the assistant know the
        # name for some questions but not others within one session). A name
        # the user stated this session takes precedence over a stored nickname
        # — it is the most current thing they told us.
        known_name = session_name or (profile.nickname if profile and profile.nickname else None)
        if known_name:
            system_prompt = f"{system_prompt} The user's name is {known_name}."

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

    def _compose_system_prompt(
        self, persona_prompt: str, language_note: str, *, spoken: bool = False
    ) -> str:
        """Identity → conversational behavior → capability honesty → persona
        style → delivery style → language. Behavior before persona so
        continuity/helpfulness rules hold for every persona; persona after so
        its voice is the last (most salient) style instruction (ADR-021
        Amendment 3). Delivery style comes after the persona deliberately: a
        persona may set tone, but it must not talk the model out of the
        length and formatting limits the output channel imposes."""
        parts = [
            _IDENTITY_PREAMBLE,
            _CONVERSATION_GUIDANCE,
            _CAPABILITY_GUIDANCE,
            persona_prompt,
            _SPOKEN_STYLE if spoken else _TEXT_STYLE,
        ]
        if language_note:
            parts.append(language_note)
        return " ".join(parts)

    def _technical_facts_block(self) -> str:
        """Backend details the model may cite only when explicitly asked a
        technical question (see `_IDENTITY_PREAMBLE`) — a separate system
        message so identity/persona text never has to name a concrete
        model.

        Each component's *live* execution device is included when a provider
        was supplied. Without it the model had only model names and the
        hardware the machine happens to contain, so "are you using my GPU?"
        was answered by guessing — measured at three wrong answers out of
        five phrasings, including "your RTX 3060 is free" while both the LLM
        and ASR were resident on CUDA.
        """
        s = self._settings
        devices = self._runtime_devices() if self._runtime_devices is not None else {}

        def component(label: str, model: str, key: str) -> str:
            device = devices.get(key)
            if not device:
                return f"{label} = {model}"
            # Spelled out in the words the user will ask in. Reporting the raw
            # "cuda" left the model to connect it to the GPU named in the
            # system-information block, and a 4B model does not reliably make
            # that leap: asked "is my RTX 3060 being used?" it answered "no,
            # I'm running on your CPU" with "running on CUDA" in its context.
            # "unloaded" is a real state (lazy TTS, pre-preload) and is said
            # plainly — an omitted device gets guessed at instead.
            human = {
                "cuda": "the GPU (CUDA)",
                "rocm": "the GPU (ROCm)",
                "cpu": "the CPU",
                "unloaded": "not loaded yet",
            }.get(device, device)
            return f"{label} = {model}, running on {human}"

        parts = [
            component("LLM model", s.llm.model, "llm"),
            component("ASR model", s.asr.model, "asr"),
            component("TTS model", s.tts.model, "tts"),
            f"VAD engine = {s.vad.engine}",
        ]
        return (
            "Technical backend details (share only if explicitly asked, and "
            "quote them exactly rather than guessing): "
            + "; ".join(parts)
            + ". Every one of these runs on this machine; no request is sent "
            "to any remote service."
        )

    def _apply_profile_preferences(self, system_prompt: str, profile: UserProfile) -> str:
        # The name is handled centrally in build() (a session-stated name takes
        # precedence over the stored nickname), so it is intentionally not added
        # here — only the non-identity preferences are.
        preferences: list[str] = []
        if profile.conversation_style:
            preferences.append(f"Preferred conversation style: {profile.conversation_style}.")
        preferences.append(f"Use {profile.units} units.")
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
