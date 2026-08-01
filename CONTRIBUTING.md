# Contributing to Edge Voice Assistant

Thanks for considering a contribution. EVA is a long-term open-source platform for
local conversational AI, and it is built to be extended — most useful contributions are
new adapters and registry entries rather than changes to core code.

This document covers **process**. For setting up your environment, running the quality
gate, and understanding the codebase, see **[docs/DEVELOPMENT.md](docs/development/DEVELOPMENT.md)**.

---

## Ways to contribute

| | |
|---|---|
| **Bug reports** | Include `eva diagnose` output, what you expected, and what happened. Logs are in the directory `eva diagnose` prints. |
| **New engine adapters** | ASR, TTS, VAD, LLM runtimes, memory stores. The highest-value contributions — see [Adding an adapter](#adding-an-adapter). |
| **Language support** | A new entry in the language registry (`src/eva/conversation/language.py`) plus a voice mapping. Small and self-contained. |
| **Platform support** | macOS packaging, AMD/ROCm validation, low-latency audio host APIs. |
| **Documentation** | Treated as production work here. Corrections and clarifications are always welcome. |
| **Performance work** | Bring a measurement. See [Performance changes](#performance-changes). |

---

## Before you start

**For anything beyond a bug fix or documentation edit, open an issue first.** EVA has a
deliberate architecture recorded in [28 ADRs](docs/architecture/adr/README.md), and a design
discussion before implementation saves rework for both of us.

Please check [ROADMAP.md](docs/project/ROADMAP.md) and [BACKLOG.md](docs/project/BACKLOG.md) — the change
you have in mind may already be planned, or deliberately deferred for a reason worth
knowing.

---

## The five architecture rules

Every change is reviewed against these. They are what keep EVA modular.

1. **Core code never names a concrete implementation.** No `if engine == "kokoro"`
   anywhere outside an adapter. Register it, resolve it by id.
2. **Dependencies point inward.** Subsystems (`audio`, `asr`, `llm`, `tts`, `vad`,
   `memory`, `embedding`) may import `core` and `config`, never each other's adapters and
   never `conversation` or `server`. One documented exception exists (`memory` imports
   `embedding`'s port — ADR-010 amendment).
3. **Every significant design decision gets an ADR.** If your change alters a contract,
   adds a dependency, or picks between real alternatives, write one. Copy the format of
   an existing ADR: Context → Decision → Rationale → Consequences → Alternatives rejected.
4. **New behavior needs tests.** Tests that require audio hardware or model weights are
   marked `@pytest.mark.integration` and excluded from CI.
5. **Documentation ships with the change**, not after it. Update `CHANGELOG.md`, the
   roadmap if status changed, and any affected document.

---

## Adding an adapter

The common case, and deliberately small. To add a speech-synthesis engine:

1. Implement the port (`src/eva/tts/base.py` — `TTSEngine`).
2. Register a factory in the subsystem registry (`src/eva/tts/registry.py`).
3. Add a catalog entry if it ships model weights (`src/eva/models/catalog.py`).
4. Add tests with a fake — the existing suite shows the pattern.

You should not need to touch the orchestrator, the server, or any UI. **If you do, that
is a design smell worth raising in the issue** — it usually means the port is missing
something, and fixing the port benefits every adapter.

---

## Performance changes

EVA's performance work is measurement-driven, and the project has been bitten by
plausible-sounding changes that measured as noise. So:

- **Bring a before/after measurement** on real hardware, with the method described.
- Synthetic benchmarks are not sufficient on their own for audio-path changes —
  the project has a documented history of synthetic tests hiding real-world failures.
- State the hardware, OS, and model configuration you measured on.

---

## Pull requests

- **One logical change per PR.** Unrelated fixes bundled together are hard to review and
  harder to revert.
- **The quality gate must pass** (see [DEVELOPMENT.md](docs/development/DEVELOPMENT.md)). CI runs it
  on Windows and Linux.
- **Write a description that explains *why*.** The diff shows what changed; the
  description should say what problem it solves and what alternatives you rejected.
- **Note any behavior change explicitly**, especially anything affecting audio timing,
  cancellation, or privacy.

Commit messages: a short imperative subject line, and a body explaining the reasoning
when the change is not self-evident.

---

## Things that will be pushed back on

Stated up front so nothing is a surprise in review:

- **Anything that makes a network call outside the model downloader.** EVA is offline by
  construction. Online capability is planned as an explicitly opt-in subsystem with a
  single controlled egress point — until that exists, new network code will be declined.
- **Vendor-specific branching in core code.** See rule 1.
- **Telemetry, analytics, or usage reporting.** No exceptions.
- **Dependencies that require a C++ toolchain to install.** EVA installs from wheels on a
  clean machine; that is a release gate (ADR-013).
- **Behavior changes without an ADR** where the ADR rules say one is required.
- **Silent capability claims.** If a feature is partial, it must report that it is
  partial rather than appearing to work.

---

## Code of conduct

Be respectful and constructive. Assume good faith. Technical disagreement is welcome and
expected; personal hostility is not. Maintainers may remove comments or contributions
that do not meet this standard. Please review our full **[Code of Conduct](CODE_OF_CONDUCT.md)** for detailed guidelines and enforcement policies.

---

## Security

If you discover a potential security issue, please do not report it in the public issue tracker. Instead, review our **[Security Policy](SECURITY.md)** for instructions on how to privately report vulnerabilities.

---

## Licensing

EVA is Apache-2.0. By contributing you agree your contribution is licensed under the
same terms.

If you add a dependency, check its license is compatible and note it in your PR.
Copyleft dependencies need explicit discussion — the one existing case (`pystray`,
LGPL-3.0) is dynamically imported and replaceable by design, per
[ADR-027](docs/architecture/adr/ADR-027-native-desktop-shell.md).
