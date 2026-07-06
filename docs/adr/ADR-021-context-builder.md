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

## Amendment 1 (M4 integration pass, 2026-07-05): identity is a fixed
concern, not a persona field

Manual testing surfaced that the assistant would introduce itself using the
underlying LLM's own trained identity (e.g. "I'm Qwen...") — none of the six
built-in personas, nor the composition logic, ever stated the product's
actual identity. This is a gap in step 1 of the composition order above, not
a new stage:

1. **`_IDENTITY_PREAMBLE`, a fixed constant, is prepended to the system
   prompt** ahead of the persona's own `system_prompt` text: `identity +
   persona.system_prompt + language.prompt_note`. It is not a per-persona
   field — every persona changes tone/verbosity, but identity ("you are Edge
   Voice Assistant, never volunteer your backend") must hold regardless of
   which persona is active, so it lives in `ContextBuilder` itself rather
   than being duplicated into all six (and every future custom) persona
   definitions.
2. **`_technical_facts_block()`** carries the actual backend details
   (`llm.model`, `asr.model`, `tts.model`, `vad.engine`) with an explicit
   instruction to share them only if the user asks a technical question
   ("which model are you?", "what LLM powers you?"). Keeping this text
   separate from the identity/persona text means the model still has the
   real facts available to answer honestly when asked, without those facts
   leaking into ordinary conversation.

At the time of Amendment 1, this technical-facts text was emitted as its
*own* `system`-role message. Amendment 2 (below) corrects that.

## Amendment 2 (M4 integration pass, 2026-07-05 — real-hardware fix): every
chat-template engine requires exactly one system message, first

**The bug.** Amendment 1's implementation, plus the pre-existing memory
block and summary block, meant `ContextBuilder.build()` emitted **up to
four separate `system`-role messages** in a row: identity+persona+language,
technical facts, retrieved memories, and the conversation summary. This
passed every unit test (pure message-list assertions, no real inference)
but failed on real hardware the first time a memory block or summary was
present: llama.cpp's Jinja chat template for Qwen (loaded from the GGUF's
embedded `chat_template` metadata) raises `ValueError: System message must
be at the beginning.` the moment it sees a second `system`-role message
anywhere in the list. This is not Qwen-specific — Llama's and Mistral's
official chat templates enforce the identical rule (exactly one system
message, always first), because none of these formats have a native
representation for "system" appearing anywhere else.

**The fix — collapse to one system message, always.** Every per-turn
system-level input — identity, persona, language note, user-profile
preferences, technical backend facts, retrieved memories, and the
conversation summary — is now joined (with blank-line separators) into a
**single** `system`-role message, which is always `messages[0]`. Nothing
past that point may ever be `system`. The fixed composition order from
section 2 above is unchanged in *content* and *sequence*; only the message
*framing* changed — what used to be up to four `ChatMessage(role="system",
...)` calls is now one string built from up to four sections.

**Enforcement, not just discipline.** `eva.llm.base.validate_chat_messages()`
is a new, model-agnostic guard — no Qwen/Llama/Mistral-specific logic —
checking exactly the two rules every template-based chat format shares:
(1) the first message must be `system`, and no other message may be; (2)
every message after it must strictly alternate `user`/`assistant`, starting
with `user`. `ContextBuilder.build()` calls this on every composed message
list before returning it, so a future regression (e.g. someone adding a new
system-level input as its own message again) fails immediately and loudly
in a unit test, instead of silently passing every mocked test and only
surfacing on real hardware against a real model.

**Defensive normalization, not just validation.** `MemoryStore.
recent_turns()` almost always alternates user/assistant (the orchestrator
only ever writes a matched pair together — `orchestrator.py`'s `add_turn`
calls are conditioned on `if reply:`), but nothing in the store schema
enforces that: a malformed `POST /memory/import` payload, a future plugin,
or a dangling turn left unanswered by a cancelled generation could produce
two adjacent same-speaker turns. Rather than let that possibility turn into
a hard `InvalidChatSequenceError` crash for an otherwise-normal user,
`ContextBuilder._normalize_alternation()` merges adjacent same-role turns
(joining their text) before the message list is built — `validate_chat_messages()`
is a safety-net assertion that should never actually fire in production, not
the primary defense.

**Why this generalizes without model-specific hacks (Part 5 of the M4
integration-pass brief).** The single-system-message-first-then-alternating
shape is not a Qwen workaround — it is the lowest common denominator every
mainstream open-weight chat template shares (Qwen, Llama 2/3, Mistral/Mixtral
Instruct, Gemma). `ContextBuilder` and `validate_chat_messages()` produce
and enforce exactly that shape unconditionally, so no `if model_id ==
"qwen..."` branch exists or is needed anywhere in this codebase. If a future
model's template genuinely needs a different shape, that adapter's contract
lives with `LLMEngine`/its factory, not with `ContextBuilder` — which stays
one deterministic policy for every engine, per this ADR's original
rationale.

**Consequences**
- `docs/MANUAL_TESTING.md` step 1's "ask who are you? then ask which LLM
  powers you?" now also implicitly exercises the memory-block and summary
  paths, since those are exactly the previously-untested code paths that
  triggered the real failure.
- `tests/test_llm_chat_validation.py` (new) unit-tests
  `validate_chat_messages()` directly against every violation named in this
  amendment. `tests/test_context_builder.py` gained
  `TestAlternationNormalization` and a "only one system message ever
  emitted" assertion covering memory+summary+history all present at once —
  the exact combination real hardware hit and the unit tests previously
  didn't.

## Amendment 3 (M5.2, 2026-07-06): prompt hierarchy and conversational guidance

Real-conversation testing (M5.2 brief) surfaced behavioral failures that were
prompt-engineering problems, not context-selection problems — the history
window (20 turns) already contained everything the model needed:

- a fragment follow-up ("with rows and columns.") was treated as a brand-new
  request;
- the assistant said "I cannot read or process images directly" (a permanent
  claim the roadmap contradicts) and "I am not a spreadsheet" (identity
  defense instead of helping);
- the default persona sounded generic, and the assistant repeated its own
  name.

Root cause: the system prompt was one dense identity block plus a one-line
persona. A 4B model gives disproportionate weight to whatever the prompt
emphasizes — and the prompt emphasized *what the assistant is* over *how to
converse*.

**The system message now has an explicit hierarchy** (still one message,
Amendment 2 unchanged):

1. **Identity** — one sentence; the name is stated once with an instruction
   to use it only when asked ("natural identity").
2. **Conversation guidance** (new, shared by every persona) — fragments/
   pronouns/ellipsis continue the current topic; prioritize the user's goal
   over self-description; anything expressible in text can be produced
   (never "I am not that kind of tool"); ambiguity → helpful assumption or
   one short clarifying question; concise by default.
3. **Capability honesty** (new) — capabilities not enabled in this build
   (e.g. image understanding) are described as "not enabled in the current
   build", never as permanently impossible.
4. **Persona style** — voice only; rewritten from one-liners into
   substantial, mutually distinct style instructions (plus a new `teacher`
   built-in). Placed after the shared guidance so style never overrides
   behavior.
5. **Language note, profile preferences** — unchanged.
6. **Conversation summary** — moved BEFORE retrieved memories: the current
   conversation's own continuity outranks cross-conversation background.
7. **Retrieved memories** — reframed from "Potentially relevant earlier
   context:" (recital-inducing) to "You remember these things … use them
   naturally, don't announce that you are recalling them".
8. **Technical backend facts** — moved to the very LAST section: position
   is salience for small models, and these are the least-wanted tokens in
   an ordinary reply.

`tests/test_conversation_quality.py` pins all of this: section order,
name-appears-once, guidance presence, fragment/pronoun antecedents in the
message list, persona pairwise distinctness, and memory-block ordering.
Behavioral validation against the real LLM is in MANUAL_TESTING §15.
