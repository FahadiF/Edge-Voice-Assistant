# ADR-015: Deterministic runtime configuration

Status: Accepted · Date: 2026-07-04

## Context
Field testing showed the assistant behaving differently across restarts: a
different LLM loaded, and barge-in responsiveness changed. Root-cause analysis
found three compounding architectural gaps — none of them threshold problems:

1. **No configuration was ever persisted.** `settings.json` was only read,
   never written, so every start re-derived model selection from *code
   defaults* — which change between releases (Qwen3 → Qwen3.5 did exactly
   this). The user's "selected model" silently followed the code.
2. **Device placement was order-dependent and silent.** ASR loaded before the
   LLM and fell back CUDA→CPU without surfacing it; the LLM then offloaded
   "as many layers as fit" into whatever VRAM remained. Ambient VRAM usage
   changed placement, and placement changed latency — perceived as changed
   barge-in behavior.
3. **Nothing displayed what was active**, so drift was undetectable.

## Decision
1. **Persist resolved configuration.** On first run (wizard or `eva run`),
   the active preset is resolved against the detected hardware tier and
   written to `settings.json`. From then on the file — not code defaults —
   decides. `eva profiles set` and `eva models use` also persist. A regression
   test pins this: a persisted selection must survive default changes.
2. **Presets + tiers, user-overridable.** Goal presets (Balanced, Fast,
   High Accuracy, Low Memory, Developer) map each capability tier to concrete
   models (`eva/hardware/presets.py`, registry data). Manual model selection
   (`eva models use`) flips the profile to `custom`, which presets never touch.
3. **Deterministic engine load order: LLM → ASR → TTS.** The LLM owns the GPU
   (architecture §5); ASR takes what remains and its fallback is visible; TTS
   is CPU. Placement no longer depends on load races.
4. **Startup banner + device truth.** Engine ports expose `device` (set at
   load with what was *actually* used); `eva run` prints profile, tier, all
   four active models, their devices, and the language before listening starts.
   The diagnostics API exposes the same data programmatically.

## Consequences
- "Which model am I running?" is answerable three ways (banner, `eva models
  list`/`info`, diagnostics snapshot) and stable across restarts.
- Changing presets or models is an explicit, persisted act; upgrades cannot
  silently change a user's configuration. Settings-file migration (schema
  version bumps) rides on the same persistence.
- The settings file is created eagerly on first run, making the configuration
  inspectable/editable from day one.
