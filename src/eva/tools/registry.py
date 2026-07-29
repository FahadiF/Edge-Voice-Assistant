"""Tool registry (ADR-010): id → `Tool`, resolved at runtime."""

from __future__ import annotations

from eva.core.registry import Registry
from eva.tools.base import Tool

tool_registry: Registry[Tool] = Registry("tool")
