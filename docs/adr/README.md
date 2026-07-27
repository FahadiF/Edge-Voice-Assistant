# Architecture Decision Records

Every significant design decision in EVA is recorded here: what was decided, why, what it
costs, and which alternatives were rejected. ADRs are **immutable once accepted** — a
decision that changes gets a new ADR or a dated amendment, never a silent edit.

Writing one is required for any change that alters a contract, adds a dependency, or
picks between real alternatives. Copy the structure of any existing record:
**Context → Decision → Rationale → Consequences → Alternatives rejected**.

---

## By subsystem

### Audio and speech pipeline
| ADR | Decision |
|---|---|
| [005](ADR-005-full-duplex-webrtc-apm.md) | Full-duplex audio with WebRTC APM echo cancellation |
| [003](ADR-003-asr-faster-whisper.md) | faster-whisper as the default speech recognizer |
| [004](ADR-004-tts-kokoro.md) | Kokoro-82M as the default speech synthesizer |
| [012](ADR-012-onnx-speech-stack-and-streaming.md) | ONNX-first speech stack; streaming as a pipeline property |
| [018](ADR-018-tts-streaming-synthesis.md) | Sub-sentence streaming synthesis |
| [028](ADR-028-speech-synchronized-text-display.md) | Speech-synchronized text display (playback-clock markers) |

### Conversation engine
| ADR | Decision |
|---|---|
| [006](ADR-006-concurrency-turn-epochs.md) | asyncio orchestration, worker threads, turn-epoch cancellation |
| [002](ADR-002-llamacpp-llm-runtime.md) | llama.cpp (GGUF) as the LLM runtime |
| [021](ADR-021-context-builder.md) | Deterministic prompt composition |
| [024](ADR-024-markdown-presentation-layer.md) | Markdown canonical everywhere except two presentation boundaries |
| [016](ADR-016-multilingual-architecture.md) | Language profiles as registry data |
| [026](ADR-026-engine-lifecycle-and-supervision.md) | Engine lifecycle: startup, shutdown, cancellation, supervision |

### Memory and personalization
| ADR | Decision |
|---|---|
| [019](ADR-019-memory-subsystem-and-sqlite-storage.md) | Memory ports and SQLite storage |
| [020](ADR-020-semantic-memory-and-retrieval.md) | Embedding model and retrieval strategy |
| [022](ADR-022-personas-user-profiles-voices.md) | Personas, user profiles, and voices |

### Platform and extensibility
| ADR | Decision |
|---|---|
| [010](ADR-010-subsystem-packages-and-registries.md) | Subsystem packages with per-subsystem registries |
| [009](ADR-009-ui-exposed-modularity.md) | Modularity exposed through the UI — registries + schema-driven settings |
| [011](ADR-011-plugin-sdk.md) | Plugin SDK: manifest, entry points, lifecycle |
| [017](ADR-017-platform-api.md) | Platform API — FastAPI backend for every client |
| [025](ADR-025-permissions-and-system-information.md) | Permissions and local system information |

### Clients
| ADR | Decision |
|---|---|
| [007](ADR-007-ui-strategy.md) | One engine, web UI + desktop shell sharing a frontend |
| [023](ADR-023-web-ui-architecture-and-hosting.md) | Web UI architecture and hosting |
| [027](ADR-027-native-desktop-shell.md) | Native desktop shell — supervision, window state, client boundary |
| [014](ADR-014-onboarding-wizard.md) | Guided onboarding wizard on first run |

### Configuration, models, packaging
| ADR | Decision |
|---|---|
| [015](ADR-015-deterministic-runtime-configuration.md) | Deterministic runtime configuration |
| [013](ADR-013-llm-runtime-installation.md) | LLM runtime installed via `eva setup`, not as a base dependency |
| [008](ADR-008-packaging-and-models.md) | Packaging and model distribution |
| [001](ADR-001-new-standalone-repository.md) | New standalone repository; thesis repo frozen |

---

## Chronological

| ADR | Title | Status | Date |
|---|---|---|---|
| [001](ADR-001-new-standalone-repository.md) | New standalone repository, thesis repo frozen | Accepted | 2026-07-03 |
| [002](ADR-002-llamacpp-llm-runtime.md) | llama.cpp (GGUF) as the LLM runtime, Qwen3.5-4B default | Accepted · amended 2026-07-04 | 2026-07-03 |
| [003](ADR-003-asr-faster-whisper.md) | faster-whisper as default ASR | Accepted | 2026-07-03 |
| [004](ADR-004-tts-kokoro.md) | Kokoro-82M as default TTS | Accepted | 2026-07-03 |
| [005](ADR-005-full-duplex-webrtc-apm.md) | Full-duplex audio with WebRTC APM echo cancellation | Accepted | 2026-07-03 |
| [006](ADR-006-concurrency-turn-epochs.md) | asyncio orchestration + worker threads + turn-epoch cancellation | Accepted | 2026-07-03 |
| [007](ADR-007-ui-strategy.md) | One engine, web UI + desktop shell sharing the same frontend | Accepted | 2026-07-03 |
| [008](ADR-008-packaging-and-models.md) | Packaging and model distribution | Accepted | 2026-07-03 |
| [009](ADR-009-ui-exposed-modularity.md) | Modularity exposed through the UI | Accepted | 2026-07-03 |
| [010](ADR-010-subsystem-packages-and-registries.md) | Subsystem packages with per-subsystem registries | Accepted | 2026-07-03 |
| [011](ADR-011-plugin-sdk.md) | Plugin SDK — manifest, entry points, lifecycle | Accepted | 2026-07-03 |
| [012](ADR-012-onnx-speech-stack-and-streaming.md) | ONNX-first speech stack; pipeline-level streaming | Accepted | 2026-07-04 |
| [013](ADR-013-llm-runtime-installation.md) | LLM runtime installed via `eva setup` | Accepted | 2026-07-04 |
| [014](ADR-014-onboarding-wizard.md) | Guided onboarding wizard on first run | Accepted | 2026-07-04 |
| [015](ADR-015-deterministic-runtime-configuration.md) | Deterministic runtime configuration | Accepted | 2026-07-04 |
| [016](ADR-016-multilingual-architecture.md) | Multilingual architecture — language profiles as registry data | Accepted | 2026-07-04 |
| [017](ADR-017-platform-api.md) | Platform API — FastAPI backend for every client | Accepted | 2026-07-04 |
| [018](ADR-018-tts-streaming-synthesis.md) | TTS streaming synthesis | Accepted | 2026-07-04 |
| [019](ADR-019-memory-subsystem-and-sqlite-storage.md) | Memory subsystem — ports and SQLite storage | Accepted | 2026-07-05 |
| [020](ADR-020-semantic-memory-and-retrieval.md) | Semantic memory — embedding model and retrieval | Accepted | 2026-07-05 |
| [021](ADR-021-context-builder.md) | Context Builder | Accepted | 2026-07-05 |
| [022](ADR-022-personas-user-profiles-voices.md) | Personas, user profiles, and voices | Accepted | 2026-07-05 |
| [023](ADR-023-web-ui-architecture-and-hosting.md) | Web UI architecture and hosting | Accepted | 2026-07-05 |
| [024](ADR-024-markdown-presentation-layer.md) | Markdown presentation layer | Accepted | 2026-07-05 |
| [025](ADR-025-permissions-and-system-information.md) | Permissions and local system information | Accepted | 2026-07-06 |
| [026](ADR-026-engine-lifecycle-and-supervision.md) | Engine lifecycle — startup, shutdown, cancellation, supervision | Accepted | 2026-07-06 |
| [027](ADR-027-native-desktop-shell.md) | Native desktop shell | Accepted | 2026-07-12 |
| [028](ADR-028-speech-synchronized-text-display.md) | Speech-synchronized text display | Accepted | 2026-07-26 |

---

## Known gaps between ADRs and implementation

Recorded here rather than left for a contributor to discover:

- **[ADR-011](ADR-011-plugin-sdk.md) is partially implemented.** Plugin discovery,
  manifests, and enable/disable work. The mechanism by which a plugin's declared
  `contributes` actually registers entries into the subsystem registries **does not exist
  yet** — plugins can be listed and toggled, but cannot currently add capabilities.
  Scheduled for the Architecture Stabilization milestone.
- **[ADR-025](ADR-025-permissions-and-system-information.md) permissions are partly
  aspirational by design.** `general.*`, `devices.microphone`, and
  `privacy.remember_conversations` are enforced. The `files`, `tools`, and
  `internet` toggles are the contract future capabilities must respect; no capability
  currently reads them because none exist. This is stated in the ADR and in the settings
  descriptions.

---

## Planned ADRs

Designed but not yet written, pending the milestones that need them:

| Proposed | Subject |
|---|---|
| ADR-029 | Provider abstraction — transport-neutral LLM port, local vs remote lifecycle |
| ADR-030 | Capability and tool port — the abstraction behind the `tools` permissions |
| ADR-031 | Online Mode — connection modes, consent, egress boundary, citations |
| ADR-032 | Secret storage — OS keychain, never settings files |
| ADR-033 | Plugin capability wiring — completing ADR-011 |
| ADR-034 | Engine-managed model lifecycle — `ModelState`, integrity verification, repair (install state, prefetch, and removal shipped 2026-07-27 as a release fix) |
