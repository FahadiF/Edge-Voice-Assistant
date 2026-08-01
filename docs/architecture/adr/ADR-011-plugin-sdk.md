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

## Implementation status (M7.3)

**Partially implemented.** The lifecycle contract of §2 is real for one
contribution kind; the rest of this ADR still describes intent.

**Shipped.** Discovery, manifests, and enable/disable, plus the
`setup(ctx)` → registration → `teardown(ctx)` lifecycle for **personas**: an
enabled plugin registers into `persona_registry` through a narrow context, and
disabling unregisters it. Plugin registrations are namespaced
(`{plugin_id}:{local_id}`) and never use `replace`, so a plugin can neither
collide with a built-in nor overwrite another plugin. Enable state persists in
`Settings.plugins.enabled`, and a newly discovered plugin defaults to
**disabled** — installing a package must never, by itself, grant a live
capability.

**Not yet shipped.** Every other contribution kind named by `contributes`
(`tool`, `llm-engine`, …) remains declarative. Marketplace install/uninstall
(§4) and phase-2 isolation (§3) are unchanged from the plan above.

### State separation (the invariant that makes §3 possible)

- `PluginState` represents persistent **catalog** state: what exists, health,
  declarations, enabled flag. It is what the platform API serves, and it
  describes a plugin whether or not it is running.
- **Activation** state remains an internal implementation detail, with
  resource ownership and cleanup: the context handed to a plugin, what that
  plugin registered, and its teardown hook.
- New contribution kinds extend **activation** state, never `PluginState`.
- This separation is what enables §3's phase-2 isolation without changing the
  catalog model: activation can move out of process while the catalog — and
  therefore the API and the UI — stays exactly as it is.

### Deviation: `eva.sdk` is deferred, not abandoned

§2 requires plugins to see only `eva.sdk`. The context object exists, but as an
**internal** module rather than a published `eva.sdk`, because the Consequences
above bind `eva.sdk` to semver discipline *from first release* — and a facade
shaped by a single contribution kind is not ready for a five-year compatibility
promise. The boundary §2 actually requires holds regardless: the manager
constructs the context and passes it to the plugin, so a plugin imports nothing
in order to use it.

**Promotion trigger:** publish as `eva.sdk` when the second, structurally
different contribution kind (tools) is wired and the shape has two consumers to
validate against. The internal activation-state grouping is revisited then too.

### Deviation: manifests are obtained by import, not read as data

§1 says the manifest is data the UI can display "without importing" the plugin.
In practice the entry point is imported and a factory called to obtain a
`PluginManifest`, so displaying a plugin *does* execute its module. This
predates the capability wiring and is unchanged by it. Recorded in BACKLOG for
a decision: either move to static `plugin.json` delivery, or amend §1 to
describe entry-point delivery as the accepted mechanism.
