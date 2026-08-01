"""Plugin activation context (ADR-011) — internal, not a published surface.

`PluginContext` is what a plugin's `setup(ctx)`/`teardown(ctx)` hooks receive.
It is deliberately narrow: for this batch it exposes exactly one contribution
kind (personas), scoped so a plugin can register and unregister its own
entries without ever importing `eva.conversation.personas` or touching
`persona_registry` directly.

Deliberately internal (`eva.plugins.context`, not `eva.sdk`): ADR-011 commits
`eva.sdk` to semver discipline from its first release, and a facade shaped by
a single contribution kind is not ready for that promise. Promote this module
to `eva.sdk` once a second, structurally different contribution kind (tools)
is wired — at that point the shape can be validated against two consumers
before it is frozen. Until then this stays an implementation detail plugins
receive but never import.
"""

from __future__ import annotations

import contextlib
import logging

from eva.conversation.personas import PersonaProfile, persona_registry
from eva.core.errors import RegistryError


class PersonaHandle:
    """A plugin's narrow view of `persona_registry`.

    Ids are namespaced to the owning plugin (`f"{plugin_id}:{local_id}"`) so
    two plugins may use the same local id without colliding, and so a plugin
    can never overwrite a built-in or another plugin's persona — `replace` is
    never offered here. Every id registered through this handle is tracked,
    so the manager can unregister exactly what this plugin owns on disable
    without depending on the plugin's own `teardown()` to remember.
    """

    def __init__(self, plugin_id: str) -> None:
        self._plugin_id = plugin_id
        self._owned: list[str] = []

    def register(self, local_id: str, persona: PersonaProfile) -> str:
        """Register `persona` under a namespaced id derived from `local_id`.

        The persona's own `.id` is set to the namespaced id (mirroring how
        built-in and custom personas both work — the registry key always
        equals `PersonaProfile.id`), so callers never need to compute it.
        Returns the namespaced id actually registered under.
        """
        namespaced = f"{self._plugin_id}:{local_id}"
        qualified = persona.model_copy(update={"id": namespaced})
        persona_registry.register(namespaced, qualified)
        self._owned.append(namespaced)
        return namespaced

    def release(self) -> None:
        """Unregister everything this handle registered. Idempotent —
        safe to call more than once (the manager calls it defensively on
        both a failed setup and a normal disable)."""
        for namespaced_id in self._owned:
            # already gone is fine; release() must not fail on re-entry
            with contextlib.suppress(RegistryError):
                persona_registry.unregister(namespaced_id)
        self._owned.clear()


class PluginContext:
    """Narrow facade passed to a plugin's `setup(ctx)`/`teardown(ctx)`.

    Grows one attribute per contribution kind as future batches wire tools,
    engines, and other registries through this same pattern.
    """

    def __init__(self, plugin_id: str) -> None:
        self.personas = PersonaHandle(plugin_id)
        self.logger = logging.getLogger(f"eva.plugins.{plugin_id}")

    def release(self) -> None:
        """Unregister everything this plugin registered, across every
        contribution kind. Called by the manager before `teardown()`, and
        again (harmlessly, since `release()` is idempotent) if `setup()`
        itself fails partway through, so a partial registration never
        outlives the failed activation that produced it."""
        self.personas.release()
