# M1 Readiness Checklist

Baseline verification before M1 (engine-managed model lifecycle) begins.
**Transient — delete when M1 closes.** Design lives in ROADMAP § M7.3 and the M1 design.

## Baseline

| | Status |
|---|---|
| Working tree clean | ✅ `d45daf0`, no uncommitted changes |
| Tests passing | ✅ 857 (`pytest -m "not integration"`) + 78 vitest |
| Lint / format / types | ✅ ruff, ruff format, mypy strict (118 files) |
| Web build | ✅ eslint + `tsc -b` + vite |
| Documentation merged | ✅ `16ff46f` (refresh) + `d45daf0` (backlog) |
| UX milestone complete | ✅ `a76d89c` CLI · `3466ab6` Models page |
| **M0 result** | ⛔ **NOT REPORTED — blocks M1a** |

## Known technical debt entering M1

Everything M1 is scheduled to fix, plus what it deliberately is not.

Three rows shipped early: the Models page could not install, detect, or remove
an engine-managed model at all, which blocked the release. That fix is the
minimum plumbing only — `ModelState`, integrity verification, repair, and
`eva models status` are untouched and remain M1's substance.

| Debt | Fixed by |
|---|---|
| ~~`is_installed()` always `False` for engine-managed models~~ | **shipped 2026-07-27** (release blocker, ahead of M1) |
| ~~No prefetch — 1.6 GB downloads silently inside `load()`~~ | **shipped 2026-07-27** (release blocker, ahead of M1) |
| ~~No removal for engine-managed models~~ | **shipped 2026-07-27** (release blocker, ahead of M1) |
| Engine-managed downloads can finalize corrupt, undetected until engine start | M1b/M1c |
| CUDA registration is a side effect of constructing the LLM | M1a |
| `load()` reports success, fails later at `encode()` | M1a |
| Auto-summarization implemented, never wired | *not M1* — ROADMAP backlog |
| CPU-bound TTS (~1.63 s of 3.5 s TTFA) | *not M1* — M8 |
| Prompt prefix defeats KV cache (~1.65 s) | *not M1* — M8 |

## Deferred backlog

`docs/BACKLOG.md` — 13 Models-page UX items (U1–U13), `eva open <page>` (C1), 7 engineering
items (E1–E7). None blocks M1. E7 (import-direction enforcement) overlaps M1's
`acquire`/`verify` decoupling test; generalise it there if cheap.

## Outstanding risks

| Risk | Note |
|---|---|
| **M0 unreported** | M1a/M1b/M1c are independent of the turbo decision, but the gate is yours. |
| **Corrupt-download root cause unknown** | Reproduced once, under a killed process. Whether an uninterrupted download is safe is unproven — raises M1c's integrity work from nice-to-have to necessary. |
| Turbo cold load ~56 s | If per-session rather than one-time, M1c should surface *load* progress, not just download progress. Watch during M0. |
| Local settings modified | `asr.model` = turbo; revert via `settings.json.pre-m0-backup`. |
| Repaired cache is hand-patched | Turbo's snapshot links were replaced with hardlinks manually. Not a clean-download state; re-verify after M1c prefetch exists. |

## Acceptance criteria

**M1a — CUDA registration**
- [ ] `ensure_cuda_libraries()` lives in `core/`, called by both llama.cpp and faster-whisper adapters
- [ ] ASR loads **and decodes** on CUDA in a process that never constructs the LLM
- [ ] Load-time smoke decode makes a cuBLAS failure surface at `load()`, so cuda→cpu fallback works
- [ ] `capture_probe.py` private-import workaround removed
- [ ] No change to engine startup behaviour; full gate green

**M1b — Install detection and state**
- [ ] `ModelState` enum (`ABSENT`/`DOWNLOADING`/`INSTALLED`/`READY`/`CORRUPTED`/`INVALID`/`FAILED`) + `ModelStatus`
- [ ] `hf_repo` / `hf_revision` on `ModelInfo`, populated for all ASR entries, **revision pinned**
- [ ] Tier-0 check = every expected file *opens*; **fixture reproducing the dangling-symlink corruption** (`islink=True`, `exists=False`)
- [ ] `is_installed()` truthful — Models page no longer shows "0/3 installed" for a running model
- [ ] `eva models status` reports state, detail, remedy
- [ ] Corrupt cache produces an actionable error, not a raw CTranslate2 message
- [ ] Independently releasable: on its own it stops the UI lying

**M1c — Acquisition, verification, repair**
- [ ] `download()` works for engine-managed models, emitting `ModelDownloadProgress`
- [ ] Resumable; idempotent when already present; engine start with weights present makes **no network access**
- [ ] Tier-1 deep verification (construct + one tiny inference) at download completion and repair; result recorded so startup need not repeat it
- [ ] Tier-1 registered **per engine id** — no ASR-specific assumptions
- [ ] `remove()` works for engine-managed models
- [ ] `eva models verify` / `eva models repair`
- [ ] `verify.py` does not import `acquire.py` and vice versa — **enforced by test**
- [ ] Docs: ADR-034, CHANGELOG, TROUBLESHOOTING (obsolete entries replaced by the repair workflow), ARCHITECTURE §10 loses two gaps

**All three:** small, independently reviewable, full gate green, no M2 work.
