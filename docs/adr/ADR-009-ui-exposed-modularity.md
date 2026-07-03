# ADR-009: Modularity exposed through the UI — registries + schema-driven settings

Status: Accepted · Date: 2026-07-03

## Context
Product requirement: the application must not be designed around any specific
LLM/ASR/TTS/VAD. Users must be able to install, swap, benchmark, and configure
components entirely from the UI (LM Studio / Open WebUI class settings surface),
with dedicated pages for Models, Conversation, Hardware, Performance, Plugins,
Audio, Developer, and Themes — no config-file or code edits.

## Decision
1. **Registries as the single source of component truth.** Each port (LLM, ASR,
   TTS, VAD, plugins) has a runtime registry mapping string ids → adapter
   factories, and a model registry mapping model ids → metadata (files, size,
   quantization, context length, license, VRAM needs, download URLs, hashes).
   Settings reference components **only by id** (`llm.engine="llamacpp"`,
   `llm.model="qwen3-4b-instruct-q4_k_m"`). Nothing in core code names a concrete
   engine.
2. **Schema-driven settings UI.** The pydantic `Settings` model is the single
   schema: bounds, enums, defaults, and descriptions are declared once and exported
   via JSON Schema through the REST API. Settings pages render from that schema, so
   adding a setting is one field in one file — engine, API, UI, and validation all
   follow.
3. **Capability discovery.** Each adapter reports its capabilities (voices, pitch
   support, languages, streaming, quantizations) through the port interface; the UI
   populates choices from live discovery, never from hardcoded lists.
4. **Hot-swap contract.** Registries support unload/load at runtime behind the
   engine's state machine (swap allowed in IDLE; in-flight turns finish or cancel
   first).

## Rationale
The "would this hold for 1000 users over five years" test: hardcoded component
lists rot; a registry + schema approach means new engines/models are data and
adapters, not UI rewrites. It also makes the benchmark harness (M7) trivially
generic — it iterates the same registries the UI shows.

## Consequences
- M2 implements the registries and model manager with this contract from day one.
- The settings REST API (M5) exposes: JSON Schema, current values, validated
  patch endpoint; the React settings pages are generated/driven by it.
- ARCHITECTURE.md §7 default stack is now merely the *default registry contents*.
