"""Tool port: a capability the model may invoke during a turn.

A tool declares what it does, what arguments it takes, and which permission
gates it; the orchestrator owns when it runs. Adapters translate a `ToolCall`
into whatever wire format their provider speaks — no tool ever sees that.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from eva.core.tools import ToolResult


class Tool(ABC):
    """One invocable capability.

    Implementations are registered by id in `eva.tools.registry`. Subclasses
    set the four class attributes and implement `execute`.
    """

    id: str
    """Registry key, and the name the model calls."""

    description: str
    """Shown to the model; it is the only basis for choosing this tool."""

    parameters: dict[str, Any]
    """JSON Schema for `execute`'s arguments, passed to the provider as-is."""

    required_permission: str
    """Dotted path into `Settings.permissions`, e.g. `general.internet`.

    Checked before the tool is offered to the model and again before it runs,
    so a permission revoked mid-turn still takes effect.
    """

    @abstractmethod
    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Run the tool.

        Must be genuinely async and must not block the event loop: barge-in
        cancels the turn task, and only an awaiting coroutine observes that
        cancellation. A tool that blocks makes a turn uninterruptible.
        """
