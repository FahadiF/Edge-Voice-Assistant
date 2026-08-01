"""Plugin subsystem: manifest schema, lifecycle manager, and capability
wiring (ADR-011).

Discovery imports each entry point and obtains its `PluginManifest`.
Enabling a plugin activates it: an internal `PluginContext`
(`eva.plugins.context` — not yet the published `eva.sdk`) is passed to its
`setup(ctx)`, which registers into the appropriate subsystem registry —
today, personas only. Disabling unregisters everything the plugin owns and
calls `teardown(ctx)`. A newly discovered plugin defaults to disabled;
enable state persists in `Settings.plugins.enabled`.

Deferred: contribution kinds beyond personas (tools, engines, ...), a
published `eva.sdk` facade, install/uninstall, and phase-2 subprocess
isolation — see ADR-011's implementation-status note for what's shipped
versus planned.
"""

from eva.plugins.manager import PluginManager, PluginState
from eva.plugins.manifest import PluginManifest

__all__ = ["PluginManager", "PluginManifest", "PluginState"]
