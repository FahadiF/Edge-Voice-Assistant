# ADR-021: Context Builder

Status: Accepted · Date: 2026-07-05

## Context

Before M4, prompt composition is `ConversationHistory.messages()`: system
prompt + a sliding window of recent turns + the new utterance. M4 adds
several more inputs that must combine into the LLM's message list: retrieved
relevant memories, conversation summaries, the active persona, and user
profile preferences. The milestone brief asks for this composition to be
deterministic and inspectable through diagnostics.

## Decision

1. **A single composition class, `eva.conversation.context_builder.
   ContextBuilder`, not a registry.** Every other new M4 capability
   (`MemoryStore`, `EmbeddingProvider`, `MemoryRetriever`, `Summarizer`) is a
   port with adapters because this product plausibly has more than one
   implementation of each. Prompt-composition *policy* is different: making
   it swappable would directly undermine the "deterministic" and
   "inspectable" requirements the brief itself sets — a caller inspecting a
   `ContextTrace` needs the composition order to be one fixed, well-tested
   thing, not whichever strategy happens to be registered. `ContextBuilder`
   is built *from* registry-resolved parts (a `MemoryStore`, a
   `MemoryRetriever`, the persona registry, the user profile store), which
   individually stay swappable — only the assembly policy itself is fixed.

2. **Fixed composition order** (documented here as the contract, enforced by
   `tests/test_context_builder.py`):

   ```
   1. System prompt = persona.system_prompt
                     + language.prompt_note (existing, ADR-016)
                     + user profile preferences, templated in
                       (nickname, units, timezone — only if a profile is active)
   2. Long-term memory: top-K semantically relevant turns for the current
      utterance (via MemoryRetriever), formatted as a compact block
   3. Latest conversation summary, if the active conversation has one
      (ADR-020 §9's Summarizer output) — keeps old context compact instead
      of re-sending everything it summarized
   4. Recent-turn window: last N raw turns, unchanged from the
      pre-M4 ConversationHistory windowing behavior
   5. The current user utterance
   ```

   Each stage has a settings-driven character budget (`top_k`,
   `max_memory_chars`, `max_summary_chars`, window size — all in the new
   `MemorySettings`/existing `ConversationSettings`, never hardcoded per
   Part 15). If a stage would exceed its budget, lowest-scored/oldest items
   are trimmed first, and the trim is recorded in the trace, not silent.

3. **Every build returns a `ContextTrace` alongside the message list**:
   which memories were retrieved and their scores, whether a summary was
   included, which persona/profile were resolved, and what (if anything)
   was trimmed for budget. This is what satisfies "inspectable through
   diagnostics" (Part 4) and backs the `GET /api/v1/memory/context-preview`
   endpoint (Part 12) — a caller can see exactly what the LLM would receive
   without spending a generation on it.

## Rationale

Determinism and inspectability are explicit product requirements for this
specific piece — they take priority over the "everything is a registry"
default that applies to actual capabilities. The distinction that matters:
`MemoryStore`/`Retriever`/`Summarizer` are things a plugin or a future
milestone plausibly reimplements; the *order in which a personal assistant
should assemble its context* is a product decision this ADR is making once,
not a pluggable strategy.

## Consequences

- The orchestrator's turn pipeline (`_pipeline()` in `orchestrator.py`)
  calls `self._context_builder.build(user_text)` where it previously called
  `self._history.messages(user_text)`; the LLM sees no change in message
  shape (still a `list[ChatMessage]`).
- `ContextTrace` objects are cheap to construct and are not published as
  full event-bus events (some retrieved-memory text could be long) — the
  API exposes them on request via context-preview, and `RuntimeSnapshot`
  exposes only summary numbers (retrieval latency, top score) per ADR-019's
  diagnostics fields, not full trace contents.
