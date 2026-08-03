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
from typing import Literal, Protocol, TypeGuard, runtime_checkable

from pydantic import BaseModel, ConfigDict

from eva.core.errors import InvalidChatSequenceError
from eva.core.events import FinishReason as FinishReason
from eva.core.tools import ToolCall, ToolDefinition


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
    call_id: str | None = None
    """On a `tool` message, the `ToolCall.id` this one answers.

    Correlation lives here rather than on `ToolResult` because the ambiguity
    it resolves is one of message *sequence*, and this is the type that models
    the sequence. One generation can request several tools at once, and their
    answers need not come back in the order they were issued; position alone
    then attributes an answer to the wrong request, silently.

    Not a copy of a provider wire field — adapters that need an id read it,
    and adapters whose format is purely ordered (Qwen's `<tool_response>`)
    ignore it. Optional so a single-call exchange stays as simple as it reads;
    `validate_chat_messages` requires it exactly when position stops being
    sufficient.
    """


def _reject_stray_call_id(message: ChatMessage) -> None:
    """`call_id` names the tool call a message answers, so only a `tool`
    message can carry one meaningfully. Checked for the leading system
    message too, which the main loop never visits."""
    if message.call_id is not None:
        raise InvalidChatSequenceError(
            f"Only a 'tool' message may set call_id; got one on '{message.role}'"
        )


def _check_tool_correlation(
    message: ChatMessage, governing: ChatMessage | None, answered: list[str]
) -> None:
    """Verify that a `tool` message names the call it answers.

    Required only once the governing turn issued more than one call: with a
    single outstanding request there is nothing to confuse it with, and
    demanding an id there would make the common exchange noisier for no gain.
    With two or more, arrival order is not request order — a slow first tool
    and a fast second one swap places — so an uncorrelated answer would be
    attributed by position to the wrong request, and the model would be told
    a plausible, wrong thing. `answered` accumulates the ids already used
    under `governing`, so the same call cannot be answered twice.
    """
    if governing is None:  # unreachable via validate_chat_messages; defensive
        return
    issued = [call.id for call in governing.tool_calls]
    if message.call_id is None:
        if len(issued) > 1:
            raise InvalidChatSequenceError(
                f"This turn issued {len(issued)} tool calls, so each 'tool' message "
                f"must set call_id (one of: {', '.join(issued)})"
            )
        return
    if message.call_id not in issued:
        raise InvalidChatSequenceError(
            f"call_id '{message.call_id}' answers no call issued by the preceding "
            f"assistant message (issued: {', '.join(issued) or '<none>'})"
        )
    if message.call_id in answered:
        raise InvalidChatSequenceError(f"call_id '{message.call_id}' is answered more than once")
    answered.append(message.call_id)


def validate_chat_messages(messages: list[ChatMessage]) -> None:
    """Enforce the chat-format contract every template-based chat engine
    needs — Qwen, Llama, and Mistral's GGUF-embedded Jinja templates all
    reject a message list that isn't: exactly one system message, first,
    then alternating user/assistant turns, where an assistant turn that
    issued tool calls may be answered by one or more `tool` messages before
    the next assistant turn. This is a generic contract (no model-specific
    logic), so one validator call at message composition time
    (`ContextBuilder.build()`) protects every current and future
    `LLMEngine` adapter without each adapter needing its own check.

    It also enforces call/answer correlation, which no chat template checks
    but every multi-call exchange depends on: see `_check_tool_correlation`.
    """
    if not messages:
        raise InvalidChatSequenceError("Message list must not be empty")
    if messages[0].role != "system":
        raise InvalidChatSequenceError(
            f"The first message must have role 'system', got '{messages[0].role}'"
        )
    _reject_stray_call_id(messages[0])
    expected: tuple[Literal["user", "assistant", "tool"], ...] = ("user",)
    previous: ChatMessage | None = None
    # The assistant turn whose calls the current run of `tool` messages is
    # answering; reset when an ordinary turn ends the run.
    governing: ChatMessage | None = None
    answered: list[str] = []
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
            if issuing:
                governing, answered = previous, []
            _check_tool_correlation(message, governing, answered)
        elif message.role != "tool":
            _reject_stray_call_id(message)
            governing, answered = None, []

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
    """Transport-neutral LLM port (Batch 8 / C1): generation only.

    Deliberately carries no `load`/`unload`/`device` — those describe a *local
    weights* lifecycle a remote/API-backed provider does not have. An adapter
    that manages on-disk weights (llama.cpp today) additionally implements
    `LocalWeights`; callers that need to know branch on `is_local()`, never on
    `isinstance(engine, LLMEngine)` or a bare `hasattr` guess.
    """

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
        *,
        tools: tuple[ToolDefinition, ...] = (),
    ) -> Generator[str, None, GenerationOutcome]:
        """Yield response text incrementally; honor `should_abort` per token.

        RETURNS (via `StopIteration.value`, i.e. `return` in the generator
        body) the `GenerationOutcome`. Callers that only iterate keep working;
        a generator with a bare `return` yields `None`, which callers treat as
        an ordinary `stop` with no tool calls.

        `tools` are the capabilities the model may be offered this pass —
        descriptions only, never invocable objects, so implementing this port
        never confers the ability to run one. Deciding *which* tools to offer
        (permissions, context) happens before the call. Keyword-only with an
        empty default: omitting it must generate exactly as it always has.
        An adapter that cannot yet offer tools should say so rather than
        ignore a non-empty tuple, which would silently answer without them.
        """


@runtime_checkable
class LocalWeights(Protocol):
    """On-disk model lifecycle (Batch 8 / C1): implemented only by adapters
    that own local model weights (llama.cpp today), never by a remote/
    API-backed adapter (the OpenAI-compatible one).

    `device` must be present — and readable — before `load()` is ever called:
    `is_local()` is checked to decide *whether* to call `load`/`unload` at
    all, so an implementing class needs a class-level default (e.g.
    `device: str = "unloaded"`), not just an attribute set inside `load()`.
    """

    device: str

    def load(self) -> None:
        """Load model weights. Idempotent."""

    def unload(self) -> None:
        """Release model resources (hot-swap support)."""


def is_local(engine: LLMEngine) -> TypeGuard[LocalWeights]:
    """True when `engine` manages its own local model weights.

    The one sanctioned way to decide whether `load()`/`unload()` apply to an
    `LLMEngine` — e.g. before `Assistant.preload()`/`unload_models()` call
    them. A remote provider simply is not `LocalWeights`; nothing else marks
    it. Typed as a `TypeGuard` so a caller's `if is_local(engine):` block lets
    mypy see `engine.load()`/`.unload()`/`.device` as valid, not just at
    runtime.
    """
    return isinstance(engine, LocalWeights)


def engine_device(engine: LLMEngine) -> str:
    """Device string for diagnostics and the runtime-awareness prompt.

    A local adapter reports what it actually loaded onto ("cuda"/"cpu"/
    "unloaded"); a remote/API-backed adapter has no device concept and
    reports "remote" explicitly, rather than the caller raising
    `AttributeError` or silently guessing.
    """
    if isinstance(engine, LocalWeights):
        return engine.device
    return "remote"
