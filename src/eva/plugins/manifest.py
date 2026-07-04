"""Plugin manifest — declarative metadata a plugin package exposes (ADR-011).

The manifest is data: the UI and marketplace can display a plugin fully
without importing it. A plugin package registers an entry point in the
`eva.plugins` group whose value is a zero-argument callable returning a
`PluginManifest`.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PluginManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    version: str
    description: str = ""
    author: str = ""
    license: str = ""
    min_engine_version: str = "0.0.0"
    contributes: tuple[str, ...] = Field(
        default=(),
        description="Kinds of registry entries this plugin adds "
        "(e.g. 'tool', 'llm-engine', 'persona')",
    )
    permissions: tuple[str, ...] = Field(
        default=(), description="Declared capabilities the plugin requests (e.g. 'filesystem')"
    )
