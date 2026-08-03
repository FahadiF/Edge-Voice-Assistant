# Security Policy

## Supported Versions

Edge Voice Assistant (EVA) is currently in **Alpha** status.

We provide security updates for the current minor release line. Please ensure you are testing against the latest version on the `main` branch or the latest tagged release.

| Version | Supported |
|---------|-----------|
| 0.7.x-alpha | :white_check_mark: |
| < 0.7.x | :x: |

## Scope

EVA is designed as a local-first, offline voice assistant. The policy broadly covers security vulnerabilities affecting EVA, including:
- Application and runtime privilege escalation or code execution
- Local API and frontend boundaries
- Filesystem and data handling (including semantic memory storage)
- Model and runtime integration
- Audio processing integrity
- Configuration and settings management
- Dependencies affecting the runtime
- Plugin and tool execution surfaces (when those features actually ship)

**Note:** Runtime tool execution, online web search, and multimodal capabilities are **NOT**
currently implemented. A local OpenAI-compatible provider adapter (Ollama/LM Studio/vLLM,
loopback-only — see "Provider and secret handling" below) shipped in Batch 8/M7.4; a genuine
**remote** provider integration remains a future addition (M7.5, Online Mode). Please do not
report hypothetical vulnerabilities for unreleased capabilities until they exist in the
repository.

## Plugin trust model

Plugins are ordinary Python packages installed with pip, discovered through the
`eva.plugins` entry-point group (ADR-011). They run **in-process, with the same
privileges as EVA itself** — this is ADR-011 phase 1; optional subprocess
isolation for untrusted plugins is phase 2 and is not implemented.

Two properties are worth stating precisely, because the words are easy to
misread:

- **Discovery imports the plugin.** Obtaining a plugin's manifest executes its
  module. A plugin that is installed but has never been enabled has still had
  its code run.
- **Enabling gates *registration*, not *execution*.** Enabling is what lets a
  plugin contribute a capability (today: a persona) into EVA's registries;
  disabling withdraws it. "Disabled" therefore means *contributes nothing*, not
  *does not run*.

Consequently, **installing a plugin is the trust decision** — treat it exactly
as you would `pip install` of any package. Newly discovered plugins default to
disabled so that installation alone never grants a live capability, and
registrations are namespaced per plugin so one plugin cannot overwrite a
built-in or another plugin's contribution. Neither of those is a sandbox.

Only the persona contribution kind is wired today; a plugin declaring any other
kind is listed and toggleable, but that kind registers nothing.

## Provider and secret handling

M7.4 (ADR-029) splits the LLM port into a transport-neutral interface plus an
optional local-weights lifecycle, and adds an OpenAI-compatible adapter
(`eva.llm.openai_compat`) covering Ollama, LM Studio, and vLLM.

- **Local providers only, this milestone.** The OpenAI-compatible adapter's
  constructor rejects any `base_url` that does not resolve to loopback
  (`127.0.0.1`/`::1`/`localhost`) — a genuinely remote endpoint is out of
  scope until M7.5 (Online Mode) defines its own connection-mode and consent
  model. This is enforced in code, not only documented: `eva.core.net`
  (Batch 6/H2) is the sole module permitted to perform real egress, and an
  import-direction test asserts nothing else — including this adapter —
  opens a non-loopback connection.
- **Secrets are references, never values.** A provider's `api_key_ref`
  setting is an opaque string (e.g. `"eva/openai"`); the actual credential is
  resolved only inside the adapter's request path, immediately before an
  authenticated call, via `eva.core.secrets`. Settings — and therefore
  `eva diagnose` output and any settings export — hold only the reference,
  never the resolved value, so there is nothing for those surfaces to leak.
  The base install resolves references from `EVA_SECRET_<REF>` environment
  variables (`EnvSecretStore`); OS-keychain storage (Windows Credential
  Manager / macOS Keychain / Secret Service) is available via the optional
  `pip install -e ".[secrets]"` extra.

## Reporting a Vulnerability

We take the security of this project seriously. Please **DO NOT** disclose vulnerabilities publicly until we have had a chance to coordinate a fix.

If **GitHub Private Vulnerability Reporting** is enabled for this repository (under the **Security > Advisories** tab), please use it to privately notify the maintainers.

If Private Vulnerability Reporting is unavailable, please contact the repository owner privately via direct message or another private channel before opening a public issue.

### What to include in your report
To help us resolve the issue quickly, please include the following information:
- A detailed description of the vulnerability and its potential impact.
- The version of EVA and your operating system.
- Step-by-step instructions to reproduce the issue.
- Expected versus actual behavior.
- Any relevant logs, crash dumps, or error messages (please remove any personal or sensitive information before submitting).

### What to expect
- You should receive an acknowledgement of your report within 72 hours.
- We will investigate the issue and communicate our assessment.
- Since this is an open-source project maintained voluntarily, we cannot guarantee an immediate SLA for a fix, but we will prioritize critical vulnerabilities affecting the offline integrity of the assistant.
- We ask for your patience and cooperation in keeping the details private until a patch is released.
