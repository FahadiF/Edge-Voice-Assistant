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


class ToolCall(BaseModel):
    """A model's request to invoke one tool."""

    model_config = ConfigDict(frozen=True)

    id: str
    """Correlates the call with its result across messages and events."""
    name: str
    arguments: dict[str, Any]
    """Already parsed; validated against the tool's schema before execution."""


class ToolResult(BaseModel):
    """The outcome of one tool invocation.

    `ok` is separate from `content` because failure is not just unusual text:
    a denied permission or an unreachable network is a state the orchestrator
    and the UI both need to branch on, and the model needs told about.
    """

    model_config = ConfigDict(frozen=True)

    ok: bool
    content: str
    """What the model sees. Rendering for the model is the adapter's job."""
    sources: tuple[Source, ...] = ()
