# ADR-011: Plugin SDK — manifest, entry points, marketplace-ready lifecycle

Status: Accepted · Date: 2026-07-03

## Context
The platform must grow (vision, OCR, RAG, filesystem, calendar, IoT, robotics,
home automation, external APIs) without core redesign, and a future in-UI plugin
marketplace must be able to install/update/enable/disable/remove plugins.
Reference points: Home Assistant integrations (manifest + discovery),
VS Code extensions (declarative contribution points + activation events),
Open WebUI functions (user-installable units).

## Decision
1. **A plugin is a Python package** exposing an entry point in the
   `eva.plugins` group and shipping a `plugin.json` manifest:
   id, name, version, description, license, author, minimum engine version,
   declared **contributions** (tools, engines, personas, prompt templates,
   settings sections, background services), and declared **permissions**
   (filesystem paths, network, devices). Manifest is data — the UI can display
   a plugin fully without importing it.
2. **Lifecycle contract** (`eva/plugins/`): discovered → loaded (entry point
   import) → `setup(ctx)` → contributions registered through the ADR-010
   registries → `teardown(ctx)` on disable/unload. `ctx` is a narrow SDK facade
   (registries, settings access scoped to the plugin, event bus, logger) — plugins
   never import engine internals, only `eva.sdk`.
3. **Isolation policy, staged:** phase 1 (in-process, permission manifest is
   informational and user-visible); phase 2 (marketplace) adds install-time
   consent UI and optional subprocess isolation for untrusted plugins. The SDK
   surface is designed now so isolation can change without breaking plugins.
4. **Installation = pip into a managed plugins environment** under the user data
   dir, driven by the plugin manager (no manual file copying). Uninstall removes
   the package and its settings.

## Rationale
Entry points + manifest is the proven Python pattern (mature tooling, versioning
via pip, no custom loader). A narrow `eva.sdk` facade keeps a five-year
compatibility contract small; declarative contributions let the marketplace and
settings UI reason about plugins without executing them.

## Consequences
- The SDK facade and event bus get stable, versioned APIs (semver discipline
  from first release).
- Registries (ADR-010) must support unregister for clean plugin disable.
- Full implementation is scheduled M5+ (manager UI) and post-1.0 (marketplace);
  the contracts above constrain all earlier design.
