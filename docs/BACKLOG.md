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

## ASR model selection

Recorded during the M7.2 catalog audit. `large-v3-turbo` is catalogued and
selectable; **no tier default changed**, because the evidence does not yet support
one. See ADR-003 § Model Selection History for the measurements taken so far.

| # | Item | Rationale |
|---|---|---|
| A1 | **Benchmark the `gpu-12gb` ASR default** — compare `large-v3`, `large-v3-turbo`, and any newer production-ready multilingual model available at that time | The current default (`small` in balanced) was never chosen on evidence, and no 12 GB card has been tested. `distil-large-v3` is no longer presented as the preferred high-accuracy option because it is English-only, but replacing it with turbo on a 6 GB measurement would substitute one assumption for another. Decide from data. |
| A2 | **Benchmark the `gpu-6gb` ASR default** — `small` vs `large-v3-turbo` on a real fixture set | Turbo recovered 3 of 4 fricative failures, but n=11, one speaker, one room, one session. Needs a wider sample before it becomes a default. Blocked on the capture probe's fixed-window truncation bug (4 of 11 samples unusable). |
| A3 | **Measure `distil-large-v3`'s real footprint** | Its `vram_mb=1600` / `ram_mb=2000` are original estimates, never verified. Left unchanged rather than replaced with a guess. |
| A4 | **Evaluate `distil-large-v3.5`** | Present in faster-whisper 1.2.1's resolver map, not catalogued. English-only, so it inherits distil's disqualification as a general default — but may be worth offering alongside `distil-large-v3`. |
| A6 | **Revisit ASR quality thresholds now that temperature fallback is off** | `temperature=0.0` removed the multi-second stalls, but it also removed Whisper's only recovery path for a bad decode. The retries were measured to hallucinate rather than recover, so this is the right default — but a bounded ladder (`[0.0, 0.2]`) or a confidence-gated single retry may recover genuine near-misses without the tail. Needs the A2 fixture set to evaluate. |
| A7 | **`refs/main` missing after a pinned-revision prefetch** | `ModelManager._download_from_hub` passes `revision=hf_revision`, which writes `snapshots/<sha>/` but no `refs/main`. The engine loads at revision `main` with `local_files_only=True`, so it reports "not in local cache", goes to the network, and would fail outright on an offline machine. Self-heals after one online start. Tracked separately from the latency work; fix before release. |
| A9 | **Exact mid-generation continuation** | After a `length` truncation EVA now says truthfully that the reply was cut off, but "continue where you left off" makes it regenerate the artifact from the top rather than resume mid-stream. Defensible (a re-generated file is at least self-consistent) but not literal continuation, and it wastes the tokens already spent. Real resumption needs the truncated text fed back as a prefix to continue from, plus a rule for stitching the two halves — enough design to deserve its own item. Deferred from the M7.3 Tier 1 correctness batch. |
| A8 | **First-chunk length dominates time-to-first-audio** | Measured on a real session: a 5.41 s TTFA turn was 442 ms ASR + 1234 ms LLM + **3736 ms TTS synthesis (69%)** of a first sentence that played for 7.14 s. Kokoro's RTF was a normal 0.52 — the synthesizer was simply handed a seven-second sentence. `conversation.first_sentence_min_chars` sets a floor (6) but nothing sets a *ceiling* on the first chunk, and the M5.6 clause split only triggers on a comma/semicolon/colon. A first-clause/max-length experiment is the lever here; GPU TTS is not. Deliberately kept out of the M7.3 correctness batch. |
| A5 | **Activate repository pinning** | `hf_repo`/`hf_revision` are recorded but **inert**: the adapter still passes the model alias, which the engine resolves through its own map. M1b makes downloads repository-aware and turns these authoritative. Until then the pin is documentation, not a guarantee. |

## Engineering

| # | Item | Rationale |
|---|---|---|
| E3 | **pre-commit strips Markdown hard line breaks** | `trailing-whitespace` removed an intentional two-space break in README during the documentation milestone. Fix: `args: [--markdown-linebreak-ext=md]`. |
| E4 | **README test-count badge is hardcoded** | `837 passing` will rot. Either wire it to CI or drop it. |
| E5 | **`docs/adr/README.md` is hand-maintained** | 28 entries with titles and dates, updated by hand per ADR. Fine now; generate it if the count grows. |
| E7 | **Subsystem dependency direction is unenforced** | ADR-010's inward-only rule is stated in docs and reviewed by eye. An import-direction test would make it structural. M1 plans one for `acquire`/`verify`; consider generalising it. |
| E9 | **ADR-011 §1 says a manifest is displayable "without importing"; it is not** | Discovery calls `ep.load()` and invokes a factory to obtain the `PluginManifest`, so listing a plugin executes its module. This predates the M7.3 capability wiring and is unchanged by it, but it means "disabled" never meant "has not run" — the trust boundary is `pip install`, not the enable toggle (now stated in SECURITY.md). Decide one of two: move to static `plugin.json` delivery so metadata really is inert data, or amend ADR-011 §1 to describe entry-point delivery as the accepted mechanism. The former matters most for a future marketplace, where browsing untrusted plugins must not execute them. |
| E8 | **Five LLM test fakes have drifted from the `LLMEngine.stream` port** | `tests/server_fakes.py`, `tests/test_orchestrator.py` (two), `tests/test_summarizer.py`, and `tests/test_benchmark.py` declare `Iterator[str]` where the port returns `Generator[str, None, GenerationOutcome]`, and none accepts the `tools` keyword added for the capability contract. They pass today only because no caller supplies `tools` and the return value is discarded — and CI cannot see it, because `mypy` is scoped to `packages = ["eva"]`. The drift stays invisible until something passes `tools`, at which point all five break at once. Realign them together rather than one at a time. |

## Deferred to M1 (not backlog — scheduled)

Listed here only so this file is not read as the complete picture:

- `ModelState` / `ModelStatus` lifecycle representation
- Post-download integrity verification and corruption detection
- `eva models status` diagnostic command
- CUDA runtime registration decoupled from the LLM adapter

Install detection, prefetch, and removal for engine-managed models shipped early on
2026-07-27 — the Models page could not install or even see a Whisper model, which
blocked the release. See CHANGELOG and `docs/M1_READINESS.md`.

See [ROADMAP.md](ROADMAP.md) § M7.3 and the M1 design.
