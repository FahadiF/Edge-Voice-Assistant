# ADR-010: Subsystem packages with per-subsystem registries

Status: Accepted · Date: 2026-07-03

## Context
The initial layout (ARCHITECTURE.md, first revision) grouped all interfaces in
`ports/` and all implementations in `adapters/`. Review against long-lived
plugin-oriented platforms (Home Assistant's integration registry, VS Code's
contribution points) showed two weaknesses: (1) everything about one capability —
its interface, its registry, its implementations — was spread across three
top-level packages, and (2) there was no single, uniform mechanism for
"register a new X without touching core code", which the product requires for
models, personas, prompt templates, profiles, tools, and plugins alike.

## Decision
1. **Package by subsystem, not by layer.** Each capability lives in one package
   that owns its port (abstract interface), its registry, and its built-in
   adapters: `eva/vad/`, `eva/asr/`, `eva/llm/`, `eva/tts/`, `eva/memory/`,
   `eva/tools/`. Cross-cutting packages: `eva/core/` (turn FSM, epochs, events,
   errors — pure domain, imports nothing else in eva), `eva/audio/`,
   `eva/conversation/`, `eva/models/` (model manager), `eva/hardware/`,
   `eva/config/`, `eva/benchmark/`, `eva/metrics/`, `eva/plugins/`, `eva/server/`.
2. **One generic registry primitive** (`eva/core/registry.py`): a typed,
   id-keyed registry with register/unregister/get/list and freeze-safe iteration.
   Every subsystem instantiates it for its own entry type. Registered kinds:
   engines (LLM/ASR/TTS/VAD), memory stores, tools, prompt templates,
   personalities, hardware profiles, plugins, and (future) embeddings.
3. **Self-registration.** Built-ins register in their subsystem's
   `register_builtins()` called during engine startup; third-party packages
   register through Python entry points (`eva.plugins` group) discovered by the
   plugin manager. No core file lists concrete implementations.
4. **Dependency direction** (enforced by review + import-linter later):
   `core` ← subsystems ← `conversation`/orchestrator ← `server` ← UIs.
   Subsystems never import each other's adapters; they may import `core` and
   `config` only.

## Rationale
Package-by-subsystem makes the tree itself communicate the architecture (the
folder list *is* the pipeline), keeps a contributor's change surface to one
directory, and the uniform registry gives the settings UI, model manager, and
benchmark suite one enumeration mechanism for everything they display.

## Consequences
- ARCHITECTURE.md §6 layout updated; `ports/`/`adapters/` packages are dropped.
- The registry primitive lands in M1 (VAD is its first consumer), model/engine
  registries in M2, persona/template/tool registries in M4, plugin registry in M5+.
