"""Tool port and registry (ADR-010).

The registry is the shared `Registry` primitive, so these tests pin the
behaviour tools inherit from it — raising on an unknown id rather than
returning None, refusing silent duplicate registration, and supporting the
unregister and replace that plugin enable/disable will need.
"""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

import pytest

from eva.core.errors import RegistryError
from eva.core.registry import Registry
from eva.core.tools import Source, ToolCall, ToolResult
from eva.tools import Tool, tool_registry


class _EchoTool(Tool):
    id = "echo"
    description = "Return the text it is given."
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }
    required_permission = "general.internet"

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        return ToolResult(ok=True, content=str(arguments.get("text", "")))


class _OtherTool(_EchoTool):
    id = "other"


@pytest.fixture
def registry() -> Registry[Tool]:
    return Registry("tool")


class TestRegistrySemantics:
    def test_registered_tool_is_returned_by_id(self, registry: Registry[Tool]) -> None:
        tool = _EchoTool()
        registry.register(tool.id, tool)
        assert registry.get("echo") is tool

    def test_unknown_id_raises_rather_than_returning_none(self, registry: Registry[Tool]) -> None:
        """Every registry in the platform reports a miss the same way. One that
        returned None would need its own error handling at every call site."""
        with pytest.raises(RegistryError, match="unknown id"):
            registry.get("nope")

    def test_duplicate_registration_is_refused(self, registry: Registry[Tool]) -> None:
        """Silent replacement would hide a collision between two plugins."""
        tool = _EchoTool()
        registry.register(tool.id, tool)
        with pytest.raises(RegistryError, match="already registered"):
            registry.register(tool.id, _EchoTool())

    def test_replace_is_explicit(self, registry: Registry[Tool]) -> None:
        first, second = _EchoTool(), _EchoTool()
        registry.register(first.id, first)
        registry.register(second.id, second, replace=True)
        assert registry.get("echo") is second

    def test_unregister_removes_the_tool(self, registry: Registry[Tool]) -> None:
        """Disabling a plugin has to withdraw what it contributed."""
        tool = _EchoTool()
        registry.register(tool.id, tool)
        registry.unregister(tool.id)
        with pytest.raises(RegistryError):
            registry.get("echo")

    def test_snapshot_lists_every_registration(self, registry: Registry[Tool]) -> None:
        echo, other = _EchoTool(), _OtherTool()
        registry.register(echo.id, echo)
        registry.register(other.id, other)
        assert set(registry.snapshot()) == {"echo", "other"}


class TestToolContract:
    def test_execute_returns_a_tool_result(self) -> None:
        result = asyncio.run(_EchoTool().execute({"text": "hi"}))
        assert isinstance(result, ToolResult)
        assert result.ok is True
        assert result.content == "hi"

    def test_a_tool_declares_the_permission_that_gates_it(self) -> None:
        assert _EchoTool().required_permission == "general.internet"

    def test_execute_must_be_implemented(self) -> None:
        class _Incomplete(Tool):
            id = "incomplete"
            description = ""
            parameters: ClassVar[dict[str, Any]] = {}
            required_permission = "general.internet"

        with pytest.raises(TypeError):
            _Incomplete()  # type: ignore[abstract]


class TestNeutralTypes:
    def test_results_carry_sources_for_attribution(self) -> None:
        result = ToolResult(
            ok=True,
            content="Helsinki is the capital of Finland.",
            sources=(Source(url="https://example.org/fi", title="Finland overview"),),
        )
        assert result.sources[0].title == "Finland overview"

    def test_a_result_defaults_to_no_sources(self) -> None:
        assert ToolResult(ok=False, content="permission denied").sources == ()

    def test_neutral_types_carry_no_provider_wire_fields(self) -> None:
        """A correlation id and JSON-string arguments are one provider's shape;
        translating into them belongs in an adapter."""
        assert "tool_call_id" not in ToolCall.model_fields
        assert ToolCall.model_fields["arguments"].annotation is not str


class TestGlobalRegistry:
    def test_the_shared_registry_is_the_core_primitive(self) -> None:
        assert isinstance(tool_registry, Registry)
        assert tool_registry.kind == "tool"
