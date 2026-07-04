"""Plugin lifecycle manager (ADR-011, backend contract only).

Discovery uses standard Python entry points (group ``eva.plugins``) — no
custom loader, versioned via pip like the rest of the ecosystem. Enable/disable
state is in-memory per process (persisted state and hot registry
unregistration arrive with real plugin loading in M5+); this milestone gives
the platform API a genuine, testable surface even though the discovered list
is normally empty until a plugin package is installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib.metadata import EntryPoint, entry_points

from eva.core.errors import PluginError
from eva.plugins.manifest import PluginManifest

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "eva.plugins"


@dataclass
class PluginState:
    manifest: PluginManifest
    enabled: bool
    healthy: bool
    entry_point: str
    error: str | None = None


class PluginManager:
    def __init__(self) -> None:
        self._plugins: dict[str, PluginState] = {}
        self._discovered = False

    def discover(self, *, force: bool = False) -> list[PluginState]:
        """Find installed plugins via entry points. Never raises: a plugin
        that fails to load is recorded as unhealthy, not fatal to discovery."""
        if self._discovered and not force:
            return self.list()
        self._plugins.clear()
        for ep in entry_points(group=ENTRY_POINT_GROUP):
            self._load_one(ep)
        self._discovered = True
        return self.list()

    def _load_one(self, ep: EntryPoint) -> None:
        try:
            factory = ep.load()
            manifest = factory()
            if not isinstance(manifest, PluginManifest):
                raise PluginError(f"entry point '{ep.name}' did not return a PluginManifest")
            self._plugins[manifest.id] = PluginState(
                manifest=manifest, enabled=True, healthy=True, entry_point=ep.value
            )
        except Exception as exc:  # a broken plugin must not break discovery
            logger.warning("Plugin entry point '%s' failed to load: %s", ep.name, exc)
            placeholder = PluginManifest(id=ep.name, name=ep.name, version="unknown")
            self._plugins[ep.name] = PluginState(
                manifest=placeholder,
                enabled=False,
                healthy=False,
                entry_point=ep.value,
                error=str(exc),
            )

    def list(self) -> list[PluginState]:
        return sorted(self._plugins.values(), key=lambda p: p.manifest.id)

    def get(self, plugin_id: str) -> PluginState:
        try:
            return self._plugins[plugin_id]
        except KeyError:
            known = ", ".join(sorted(self._plugins)) or "<none installed>"
            raise PluginError(f"unknown plugin '{plugin_id}' (installed: {known})") from None

    def enable(self, plugin_id: str) -> PluginState:
        state = self.get(plugin_id)
        if not state.healthy:
            raise PluginError(f"cannot enable unhealthy plugin '{plugin_id}': {state.error}")
        state.enabled = True
        return state

    def disable(self, plugin_id: str) -> PluginState:
        state = self.get(plugin_id)
        state.enabled = False
        return state

    def reload(self, plugin_id: str) -> PluginState:
        """Re-run discovery for a single plugin's entry point."""
        self.get(plugin_id)  # 404s early if the id was never discovered
        matches = [ep for ep in entry_points(group=ENTRY_POINT_GROUP) if ep.name == plugin_id]
        if not matches:
            raise PluginError(f"plugin '{plugin_id}' is no longer installed")
        self._load_one(matches[0])
        return self.get(plugin_id)
