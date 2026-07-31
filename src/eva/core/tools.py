"""Provider-neutral tool vocabulary.

Lives in `core` for the same reason `FinishReason` does: the LLM port, the
event bus, the tool registry and the orchestrator all name these types, and
`core` may not import a subsystem to do it (ADR-010). Keeping them here is
what stops `eva.core.events` from depending on `eva.tools` once tool events
exist.

Deliberately free of any provider's wire format. Qwen emits an XML-like
markup and expresses a tool result as a plain `<tool_response>` block with no
call identifier; other providers use JSON with correlation ids. Translating
between those shapes is an adapter's job, so nothing here mirrors one of them.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class Source(BaseModel):
    """Where a piece of tool output came from, for attribution in the UI."""

    model_config = ConfigDict(frozen=True)

    url: str
    title: str
    """A bare URL is not readable in a citation; tools supply a human label."""


class ToolDefinition(BaseModel):
    """What the model is told a tool can do — never the tool itself.

    The LLM port is offered these rather than `Tool` objects so an adapter can
    describe a capability to the model without gaining the ability to run one:
    a definition has no `execute`. It also keeps `eva.llm` from importing
    `eva.tools`, which would be a sibling dependency (ADR-010).

    Derived from a `Tool` by `Tool.definition()`. Whoever derives it decides
    what to include, which is where permission filtering belongs — a tool the
    user has not permitted simply never becomes a definition.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    """Matches the tool's registry id, and the name the model calls."""
    description: str
    """The model's only basis for choosing this tool."""
    parameters: dict[str, Any]
    """JSON Schema for the arguments, passed to the provider as-is."""


class ToolCall(BaseModel):
    """A model's request to invoke one tool."""

    model_config = ConfigDict(frozen=True)

    id: str
    """Identifies this call so its answer can name it.

    The answer carries the correlation, not the result value: a tool result
    becomes a `tool`-role `ChatMessage`, and that message's `call_id` points
    back here. See `eva.llm.base.validate_chat_messages`, which enforces it
    once a turn issues more than one call and position stops being enough.
    """
    name: str
    arguments: dict[str, Any]
    """Already parsed; validated against the tool's schema before execution."""


class ToolResult(BaseModel):
    """The outcome of one tool invocation.

    `ok` is separate from `content` because failure is not just unusual text:
    a denied permission or an unreachable network is a state the orchestrator
    and the UI both need to branch on, and the model needs told about.

    Deliberately carries no call id. Which call a result answers is a property
    of the message sequence, not of the outcome value, and it is recorded once
    — on the `tool`-role `ChatMessage` built from this result. Putting it here
    too would duplicate the same fact into a layer that never reads it, and
    would oblige every `Tool.execute` to thread an id it is not given.
    """

    model_config = ConfigDict(frozen=True)

    ok: bool
    content: str
    """What the model sees. Rendering for the model is the adapter's job."""
    sources: tuple[Source, ...] = ()
