"""Plugin lifecycle manager (ADR-011).

Discovery uses standard Python entry points (group ``eva.plugins``) — no
custom loader, versioned via pip like the rest of the ecosystem.

Two kinds of state are kept deliberately separate:

- `PluginState` is persistent CATALOG state: what plugins exist, are they
  healthy, what they declare, are they enabled. It is what the API serves,
  and it describes a plugin whether or not it is currently running.
- Activation state (the private `_Hooks`/`_Activation` records below) exists
  only while a plugin is loaded/enabled: the callables a loaded plugin
  exposed, the `PluginContext` handed to it, and its `teardown` hook. It owns
  resources and requires cleanup.

New contribution kinds (tools, engines, ...) extend activation state, never
`PluginState` — that separation is what will let ADR-011 §3's phase-2
subprocess isolation change *how* a plugin is activated (in-process today,
possibly out-of-process later) without touching the catalog model at all.

Enable state is persisted in `Settings.plugins.enabled`; a newly discovered
plugin defaults to disabled until a caller explicitly enables it — a plugin
package must never gain a live capability merely by being installed.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import EntryPoint, entry_points

from eva.config.settings import Settings
from eva.core.errors import PluginError
from eva.plugins.context import PluginContext
from eva.plugins.manifest import PluginManifest

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "eva.plugins"

_SetupHook = Callable[[PluginContext], None]
_TeardownHook = Callable[[PluginContext], None]


@dataclass
class PluginState:
    manifest: PluginManifest
    enabled: bool
    healthy: bool
    entry_point: str
    error: str | None = None


@dataclass
class _Hooks:
    """What a loaded plugin object exposed, captured once at load time.

    Both `None` for a plugin whose entry point returns a bare
    `PluginManifest` — that plugin is metadata-only and never activates.
    """

    setup: _SetupHook | None
    teardown: _TeardownHook | None


@dataclass
class _Activation:
    """Internal implementation detail: resources owned by an ENABLED plugin.

    Deliberately kept out of `PluginState` — see the module docstring.
    """

    context: PluginContext
    teardown: _TeardownHook | None


class PluginManager:
    def __init__(self) -> None:
        self._plugins: dict[str, PluginState] = {}
        self._hooks: dict[str, _Hooks] = {}
        self._activations: dict[str, _Activation] = {}
        self._discovered = False

    def discover(self, settings: Settings, *, force: bool = False) -> list[PluginState]:
        """Find installed plugins via entry points. Never raises: a plugin
        that fails to load, or fails to activate, is recorded as unhealthy,
        not fatal to discovery.

        Re-applies `settings.plugins.enabled` on every full discovery (not
        just the first) — mirrors `register_custom_personas`'s idempotent
        pattern — so a plugin enabled in a previous process is active again
        after a restart with no explicit `enable()` call required.
        """
        if self._discovered and not force:
            return self.list()
        for plugin_id in list(self._activations):
            self._deactivate(plugin_id)
        self._plugins.clear()
        self._hooks.clear()
        for ep in entry_points(group=ENTRY_POINT_GROUP):
            self._load_one(ep)
        for plugin_id, state in self._plugins.items():
            if state.healthy and plugin_id in settings.plugins.enabled:
                self._try_activate(plugin_id, state)
        self._discovered = True
        return self.list()

    def _load_one(self, ep: EntryPoint) -> None:
        try:
            factory = ep.load()
            loaded = factory()
            setup: _SetupHook | None = None
            teardown: _TeardownHook | None = None
            if isinstance(loaded, PluginManifest):
                manifest: PluginManifest = loaded
            else:
                candidate = getattr(loaded, "manifest", None)
                if not isinstance(candidate, PluginManifest):
                    raise PluginError(
                        f"entry point '{ep.name}' returned neither a PluginManifest "
                        "nor an object exposing one as `.manifest`"
                    )
                manifest = candidate
                setup = getattr(loaded, "setup", None)
                teardown = getattr(loaded, "teardown", None)
            self._plugins[manifest.id] = PluginState(
                manifest=manifest, enabled=False, healthy=True, entry_point=ep.value
            )
            self._hooks[manifest.id] = _Hooks(setup=setup, teardown=teardown)
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

    def _try_activate(self, plugin_id: str, state: PluginState) -> None:
        """Activate during discovery: a failure here is recorded on the
        plugin, not raised — discovery as a whole must never fail."""
        try:
            self._activate(plugin_id)
            state.enabled = True
        except Exception as exc:
            logger.warning("Plugin '%s' failed to activate: %s", plugin_id, exc)
            state.healthy = False
            state.error = str(exc)
            state.enabled = False

    def _activate(self, plugin_id: str) -> None:
        """Run `setup(ctx)` and record the activation. Raises on failure
        (e.g. a `RegistryError` from a colliding persona id) — callers
        decide whether that should abort loudly (`enable`) or be recorded
        as unhealthy (`_try_activate`, during discovery)."""
        if plugin_id in self._activations:
            return  # already active; enable() is idempotent
        hooks = self._hooks.get(plugin_id)
        context = PluginContext(plugin_id)
        try:
            if hooks is not None and hooks.setup is not None:
                hooks.setup(context)
        except Exception:
            # A setup that registers two things and fails on the second must
            # not leave the first orphaned in the registry forever.
            context.release()
            raise
        self._activations[plugin_id] = _Activation(
            context=context, teardown=hooks.teardown if hooks is not None else None
        )

    def _deactivate(self, plugin_id: str) -> None:
        activation = self._activations.pop(plugin_id, None)
        if activation is None:
            return
        # Unregister everything this plugin owns before notifying it via
        # teardown() — correctness must not depend on the plugin remembering
        # to release its own registrations.
        activation.context.release()
        if activation.teardown is not None:
            try:
                activation.teardown(activation.context)
            except Exception:
                logger.warning("Plugin '%s' teardown raised; ignoring", plugin_id, exc_info=True)

    def list(self) -> list[PluginState]:
        return sorted(self._plugins.values(), key=lambda p: p.manifest.id)

    def get(self, plugin_id: str) -> PluginState:
        try:
            return self._plugins[plugin_id]
        except KeyError:
            known = ", ".join(sorted(self._plugins)) or "<none installed>"
            raise PluginError(f"unknown plugin '{plugin_id}' (installed: {known})") from None

    def enable(self, plugin_id: str, settings: Settings) -> PluginState:
        """Enable and activate a plugin. `settings` is mutated in place
        (the id is added to `plugins.enabled`); the caller is responsible
        for persisting it — this mirrors `register_custom_personas`, which
        also never calls `save_settings` itself."""
        state = self.get(plugin_id)
        if not state.healthy:
            raise PluginError(f"cannot enable unhealthy plugin '{plugin_id}': {state.error}")
        self._activate(plugin_id)  # propagates (e.g. RegistryError) on a real failure
        state.enabled = True
        if plugin_id not in settings.plugins.enabled:
            settings.plugins.enabled = [*settings.plugins.enabled, plugin_id]
        return state

    def disable(self, plugin_id: str, settings: Settings) -> PluginState:
        state = self.get(plugin_id)
        self._deactivate(plugin_id)
        state.enabled = False
        if plugin_id in settings.plugins.enabled:
            settings.plugins.enabled = [pid for pid in settings.plugins.enabled if pid != plugin_id]
        return state

    def reload(self, plugin_id: str) -> PluginState:
        """Re-run discovery for a single plugin's entry point.

        Refreshes catalog state (manifest, health) only. If the plugin was
        active, its activation is left untouched — reloading a live
        plugin's hooks safely (tear down the old activation, re-run setup
        against the freshly loaded object) is deferred; `ep.load()`
        typically returns the same already-imported module, so the hooks
        captured before and after a reload are normally identical anyway.
        """
        self.get(plugin_id)  # 404s early if the id was never discovered
        matches = [ep for ep in entry_points(group=ENTRY_POINT_GROUP) if ep.name == plugin_id]
        if not matches:
            raise PluginError(f"plugin '{plugin_id}' is no longer installed")
        was_enabled = self._plugins[plugin_id].enabled
        self._load_one(matches[0])
        self._plugins[plugin_id].enabled = was_enabled
        return self.get(plugin_id)
