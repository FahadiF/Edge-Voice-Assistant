# Backlog

Ideas that are worth doing but are not scheduled. Milestone-level work lives in
[ROADMAP.md](ROADMAP.md); this file is for smaller items that would otherwise be lost —
each with a short rationale so a future reader knows *why*, not just *what*.

Items move out of here by being scheduled into a milestone, or by being deleted with a
note when they stop making sense.

---

## Models page — UX

Recorded during the M7 UX polish pass. None of these require architectural change; they
were deferred to keep that pass small.

| # | Item | Rationale |
|---|---|---|
| U1 | **Deterministic ordering** — keep `ACTIVE → INSTALLED → AVAILABLE` within every category | Implemented in M7 polish via `STATUS_ORDER`. Listed so the invariant survives the M1 `ModelState` change: new states must be given an explicit rank rather than falling back to catalog order. Consider a test pinning it. |
| U2 | **Search** — "Search models…" | The catalog is 10 entries today, so search is premature. It becomes necessary once third-party catalogs land (ADR-010 allows them). |
| U3 | **Filters** — All / Installed / Available | Same reasoning as U2. Cheap once the status vocabulary is stable after M1. |
| U4 | **Collapsible sections** | Five always-expanded groups make the page long. Most users care about one kind at a time. Worth pairing with U2/U3 rather than doing alone. |
| U5 | **Richer hardware summary** — GPU / CPU / RAM as a labelled block | The current one-line summary was added for the VRAM fit messages. A proper block belongs on the Dashboard as much as here; do both together. |
| U6 | **Tooltips for jargon** — `Q4_K_M`, `GGUF`, `ONNX`, `CT2` | These terms are opaque to non-specialists and currently appear with no explanation. Needs a shared tooltip component, so it is a small design task, not a one-liner. |
| U7 | **Recommendation icons** — ⭐ Recommended, ⚡ Fastest, 🎯 Best accuracy, 💾 Low memory, 🌍 Multilingual | Icons scan faster than text. Requires an icon vocabulary in catalog data (an enum beside `recommendation`), not UI string-matching — otherwise it re-introduces the UI conditionals that `recommendation` was created to remove. |
| U8 | **Model comparison** — compare two models side by side | Genuinely useful for choosing an ASR model, but only once there is something to compare on beyond catalog metadata. Depends on U9. |
| U9 | **Per-model benchmark button** — startup, TTFA, VRAM, latency | The most valuable of these, and the largest. Belongs with the M8 benchmark harness, which already has to measure exactly these. Would let users make evidence-based choices instead of reading our recommendation text. |
| U10 | **Hardware-aware recommendations** — "Recommended for YOUR PC" | The data already exists (`compatible`, `vram_mb`, `HardwareSummary`); M7 polish uses it for fit warnings. Turning it into per-machine recommendations needs a policy ("what is best for 6 GB?"), which is really the preset system (`hardware/presets.py`) surfaced in the UI. Design it as *presets made visible*, not as a second recommendation engine. |
| U11 | **`ModelState` promotion** — badge tones for `READY` / `CORRUPTED` / `INVALID` | The CSS and tone vocabulary already exist and are unused. Purely an M1 follow-through: emit the states, delete the "not yet emitted" comment. |
| U12 | **Model metadata dialog** — click a model for repository, revision, license, author, languages, quantization, download/installed size, verification state, last verified, engine, provider, used-by | The card deliberately shows only what fits at a glance; the rest currently has nowhere to live. Worth building *after* M1, because verification state and last-verified are half the value and neither exists yet. |
| U13 | **Model version history** — installed revision, current upstream revision, update-available date | `ModelInfo.version` and `update_available` exist but only express "newer catalog entry", not upstream revisions. Real revision tracking depends on M1 pinning `hf_revision`, so this follows M1/M2 rather than preceding them. |

## CLI — UX

| # | Item | Rationale |
|---|---|---|
| C1 | **`eva open <page>`** — `eva open models`, `eva open voices`, `eva open settings`, `eva open diagnostics`, later `plugins` / `logs` / `downloads` | Natural extension now that `eva desktop` exists: open the desktop app directly on a page, starting it first if it is not running. The `eva open` prefix is deliberate — `eva models` and `eva voices` already are (and should remain) CLI subcommand *groups* for model and voice management, so a bare `eva models` would break scripted usage. `open` also scales to new pages without ever colliding with a management verb. |

## Engineering

| # | Item | Rationale |
|---|---|---|
| E1 | **Generate `web/src/api/types.ts` from OpenAPI** | The mirror is hand-maintained and has drifted twice. `tests/test_web_types_sync.py` now fails CI on drift, which removes the *risk*; generation would remove the *duplication*. A build-pipeline change (openapi-typescript + a check-in step), so it needs its own scoping. |
| E2 | **`web/tsconfig.*.tsbuildinfo` are tracked in git** | Every `npm run build` dirties the working tree. They are build artifacts and should be ignored. |
| E3 | **pre-commit strips Markdown hard line breaks** | `trailing-whitespace` removed an intentional two-space break in README during the documentation milestone. Fix: `args: [--markdown-linebreak-ext=md]`. |
| E4 | **README test-count badge is hardcoded** | `837 passing` will rot. Either wire it to CI or drop it. |
| E5 | **`docs/adr/README.md` is hand-maintained** | 28 entries with titles and dates, updated by hand per ADR. Fine now; generate it if the count grows. |
| E6 | **Add the install-detection symptom to the ADR gap list** | `docs/adr/README.md` describes the engine-managed lifecycle gap abstractly. The concrete symptom — "Speech Recognition — 0/3 installed" while a model is demonstrably running — makes it land. Do this when M1 closes the gap, as part of removing the entry. |
| E7 | **Subsystem dependency direction is unenforced** | ADR-010's inward-only rule is stated in docs and reviewed by eye. An import-direction test would make it structural. M1 plans one for `acquire`/`verify`; consider generalising it. |

## Deferred to M1 (not backlog — scheduled)

Listed here only so this file is not read as the complete picture:

- `ModelState` / `ModelStatus` lifecycle representation
- Install detection for engine-managed models (`is_installed()` currently always `False` for ASR)
- Post-download integrity verification and corruption detection
- Prefetch with progress reporting for engine-managed models
- Removal support for engine-managed models
- `eva models status` diagnostic command
- CUDA runtime registration decoupled from the LLM adapter

See [ROADMAP.md](ROADMAP.md) § M7.3 and the M1 design.
