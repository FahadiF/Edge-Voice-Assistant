"""Plugin subsystem: manifest schema + lifecycle manager (ADR-011).

M2.6 scope is the backend contract only — discovery, enable/disable/reload,
and health reporting — so the platform API has a real (if currently empty)
surface to expose. Loading third-party code and the `eva.sdk` facade land
with the marketplace work in M5+.
"""

from eva.plugins.manager import PluginManager, PluginState
from eva.plugins.manifest import PluginManifest

__all__ = ["PluginManager", "PluginManifest", "PluginState"]
