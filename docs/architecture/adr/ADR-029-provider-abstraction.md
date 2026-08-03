# ADR-029: Provider abstraction — transport-neutral LLM port, local vs remote lifecycle

Status: Accepted · Date: 2026-08-03

## Context

`LLMEngine` (`eva.llm.base`) carried `load()`, `unload()`, and `device` as
required abstract members from the start (M2), because the only adapter that
ever existed was `LlamaCppLLM` — a local, on-disk GGUF model. `LLMFactory`
(`eva.llm.registry`) mirrored that assumption: `Callable[[Settings, Path],
LLMEngine]`, the caller (`build_assistant`) resolving a model file path and
handing it to the factory, unlike `eva.asr.registry`'s `Callable[[Settings,
AppPaths], ASREngine]`, which already lets the adapter resolve its own path.

M7.4 ("Provider Abstraction") requires the orchestrator stop assuming models
are local files: a provider chain with fallback, an OpenAI-compatible adapter
(covering Ollama, LM Studio, vLLM, and most cloud vendors' chat-completions
API at once), and OS-keychain secret storage. Fixing the port's shape *after*
a second adapter and a nested settings structure exist would multiply the
change across every adapter instead of doing it once — this is finding **C1**
from the Principal Architecture Review, landing together with **H7** (nested
provider settings) and **H6** (secret storage) as one coordinated change,
since all three touch the same files and the same construction path.

A full read-only implementation review was performed against the shipped
state of Batches 4A/6/7/9/11 before any code was written (recorded in the
Final Execution Roadmap's Batch 8 section, frozen 2026-08-03). It found the
review's own file-list estimate incomplete (17 `settings.llm.*` read/write
sites existed, not the ~5 named) and one factual defect in the settings
migration machinery that bumping the schema version would have exposed (see
Decision 3 below). Six design decisions were required to resolve real
conflicts the review surfaced, approved and frozen before implementation
began; they are Decisions 1–6 here.

## Decision

### 1. `LLMEngine` carries generation only; `LocalWeights` carries the local lifecycle

```python
class LLMEngine(ABC):
    def count_tokens(self, text: str) -> int: ...
    @abstractmethod
    def stream(self, messages, params, should_abort, *, tools=()) -> Generator[str, None, GenerationOutcome]: ...

@runtime_checkable
class LocalWeights(Protocol):
    device: str
    def load(self) -> None: ...
    def unload(self) -> None: ...
```

`LocalWeights` is a `@runtime_checkable` `Protocol`, not a subclass
relationship: an adapter satisfies it structurally (present `device`,
`load`, `unload`), so a remote/API-backed provider simply does not implement
it — no flag, no opt-out, structural absence. Two module-level helpers are
the *only* sanctioned way to branch on this:

```python
def is_local(engine: LLMEngine) -> TypeGuard[LocalWeights]: ...
def engine_device(engine: LLMEngine) -> str:  # "cuda"/"cpu"/"unloaded", or "remote"
```

`is_local` is typed as a `TypeGuard` so a caller's `if is_local(engine):`
block lets mypy see `.load()`/`.unload()`/`.device` as valid, not merely at
runtime. `LlamaCppLLM` declares `device: str = "unloaded"` as a **class**
attribute (previously inherited from `LLMEngine`) — `is_local()` is checked
*before* `load()` ever runs (to decide whether to call it at all), so the
attribute must exist ahead of that call, not only be set inside it.

Every caller of `.device`/`.load()`/`.unload()` on an `LLMEngine` was found
and updated to go through these two functions instead
(`Assistant.preload()`/`.unload_models()`, `ContextBuilder`'s
`runtime_devices` callback, `RuntimeSnapshot`'s device map, the CLI banner,
benchmark provenance) — **4 direct `.device` reads and 2 direct
`.load()`/`.unload()` call sites**, all in `src/eva/engine.py` and its
consumers.

### 2. `LLMFactory` converges on the ASR pattern

```python
LLMFactory = Callable[[Settings, AppPaths], LLMEngine]
```

`_make_llamacpp` resolves its own path via `ModelManager(paths).files_for(...)`
internally, exactly as `_make_faster_whisper` already does for ASR.
`build_assistant` becomes the only call site touched at the wiring level, per
C1's own migration strategy — its LLM construction is now
`create_llm(settings, paths)`, no `llm_path` variable in between.

### 3. Provider settings are a fixed-key model, not a dict — and each schema-migration transform is now version-gated individually

```python
class ProvidersSettings(_Section):
    local: LocalProviderSettings = Field(default_factory=LocalProviderSettings)
    openai_compatible: OpenAICompatibleProviderSettings = Field(
        default_factory=OpenAICompatibleProviderSettings
    )

class LLMSettings(_Section):
    engine: str = Field("llamacpp", ...)   # UNCHANGED position
    model: str = Field(...)                # UNCHANGED position
    providers: ProvidersSettings = Field(default_factory=ProvidersSettings)
    chain: list[str] = Field(default_factory=lambda: ["local"], ...)
```

**Why not `dict[str, ProviderSettings]`.** The Settings page renders nested
settings only by resolving a `$ref` in the JSON Schema
(`web/src/components/SchemaForm.tsx`'s `resolveRef`); an
`additionalProperties` dict has no such schema for the UI to resolve and
falls through to a bare text input showing `[object Object]`. Verified
directly against both `Settings.model_json_schema()` and the generated
OpenAPI mirror: `llm.providers` is `$ref: "#/$defs/ProvidersSettings"`,
`providers.local`/`.openai_compatible` are each their own named `$defs`
entry, and `chain` is a plain `array` of `string` — the Settings page renders
all of it with **zero frontend code change**.

**Why `engine`/`model` stay top-level.** Only the local-runtime knobs
(`context_length`, `gpu_layers`, `threads`, `batch_size`) move under
`providers.local`. `engine`/`model` remain exactly where 11 existing
read/write sites expect them — including two *writers*,
`hardware/presets.py` (preset application) and
`server/routers/models.py` (the models API's "activate" endpoint) — so this
batch's blast radius on existing call sites is 3 files (both `context_builder.py`
read sites, one `presets.py` write site), not the ~17 a flat rename would
have touched.

**Why `chain` is schema-only this batch.** `chain: list[str]` records the
fallback order a *future* addition will walk; `settings.llm.engine` remains
the sole active-provider selector this batch. Wiring automatic chain-walking
into `build_assistant` was considered and deliberately deferred: doing it
correctly requires moving the local provider's `.load()` call earlier (to
detect a load failure and fall back), which would change *when* GPU memory
is claimed relative to `Assistant.preload()`'s existing GPU-ownership
ordering (ADR-015 §5) and its `ComponentLoadStarted`/`Finished` event
bracketing — a real behavior change to a pinned contract that no approved
decision asked for. Building it un-reviewed would be scope expansion past
what was frozen; the schema exists so a future batch has somewhere correct
to land it.

**Migration (`schema_version` 5 → 6).** `_migrate_raw`'s v1-v5 transforms
were gated on one check: `schema_version >= SETTINGS_SCHEMA_VERSION`. That
meant bumping the version made every already-migrated document re-run every
prior transform, keyed only on the *old default value* — a v5 user who
deliberately reset `sentence_max_chars` to 350 (the pre-A8 default) would
have had it silently rewritten the moment v6 shipped, and the existing test
`test_already_v5_not_touched` would have broken. Each transform is now
gated on the version it migrates *from* individually
(`if from_version < N:`), not on the single top-level check; a v5 document
now skips every v1-v5 transform outright, regardless of content, and only
the new v5→v6 transform (moving the four local-runtime fields, adding
`chain`) ever touches it. Migrations remain one-directional, consistent with
the existing v1-v5 chain; a reverted binary that meets a `schema_version`
newer than its own fails loudly with a `ConfigError` naming both versions,
rather than guessing at an unfamiliar shape.

### 4. The OpenAI-compatible adapter is loopback IPC, not egress

Ollama, LM Studio, and vLLM all run on `127.0.0.1` in the deployments this
milestone targets. Batch 6 made `eva.core.net` the sole egress point and
pinned `urllib.request` to four named files via an import-direction test
(`tests/test_offline_invariant.py`). `eva.llm.openai_compat` is added as a
**fifth** allowlisted file — loopback IPC, exactly like the desktop client
and the service supervisor — rather than routed through `core.net`, which
exists specifically to gate traffic *leaving* the machine. The adapter's
constructor rejects a non-loopback `base_url` outright, with an error naming
M7.5 (Online Mode) as where a real remote endpoint belongs; this keeps the
offline invariant's socket-level test meaningful for this adapter without
inventing a second classification mechanism.

### 5. Secrets: references in settings, values only in `eva.core.secrets`

```python
class SecretStore(ABC):
    def get(self, ref: str) -> str: ...

class EnvSecretStore(SecretStore): ...      # base install: EVA_SECRET_<REF>
class KeyringSecretStore(SecretStore): ...  # optional `[secrets]` extra
```

A provider config field is named `api_key_ref: str | None` — an opaque
reference, never a raw key. `EnvSecretStore` ships in the base install
(no new dependency, identical behavior on every platform);
`KeyringSecretStore` imports `keyring` lazily inside its own `get()`, never
at module import time, so a bare install never pulls it in. `keyring`
becomes a new optional extra, `pip install -e ".[secrets]"`, following the
same shape as the existing `[cpu]`/`[cuda]`/`[desktop]` extras. Because the
settings document only ever holds the reference, there is nothing for a
diagnostics dump or settings export to redact — the secret value was never
there to redact.

### 6. Provider fallback is construction/load-time only, never mid-stream

A provider that fails to construct or load may be retried against the next
entry in `chain` (once that walk is implemented — see Decision 3). Once a
provider has begun streaming a turn, that turn either completes or is
cancelled by the existing turn-epoch mechanism (ADR-006); it never silently
switches providers mid-generation. A mid-stream swap would restart
generation inside a live epoch, which the cancellation contract does not
model, and was never a stated M7.4 requirement — "chain with fallback" was
ambiguous between the two, and this closes the ambiguity rather than leaving
it implicit.

## Consequences

- `LlamaCppLLM` is unaffected in every observable way: same construction
  arguments (now sourced from `providers.local` instead of flat `llm.*`
  fields), same `load`/`unload`/`device` behavior, same generation contract.
- A remote/API-backed adapter can be added without ever touching
  `load`/`unload`/`device` — it simply does not implement `LocalWeights`, and
  every consumer already branches on `is_local()`/`engine_device()` rather
  than assuming the attribute exists.
- `required_models()` (readiness/preflight) excludes the LLM model when
  `llm.engine` is not in `eva.llm.registry.LOCAL_ENGINE_IDS` — otherwise a
  correctly-configured Ollama setup would fail `eva doctor`/`eva bench`
  demanding a local file that was never meant to exist.
- `ContextBuilder`'s history-token budget still reads
  `providers.local.context_length` even when a remote provider is active —
  a remote provider has no context-length field of its own this batch, so
  the budget calculation currently borrows the local provider's configured
  value. Acceptable for M7.4 (local providers only); a follow-up should give
  each provider its own context-length knob if remote generation ships.
- No settings-schema field was removed; `extra="forbid"` means an already-
  migrated v6 document with the old flat `llm.context_length` etc. would be
  rejected outright by an older binary — expected and desired given the
  one-directional migration policy (Decision 3).

## Alternatives rejected

- **`providers: dict[str, ProviderSettings]`** — most flexible in the
  abstract, but verified to break the Settings page's generic `$ref`-based
  renderer; would have required a frontend change explicitly out of scope.
- **Moving `engine`/`model` under `providers` too** — roadmap-literal, but
  would have touched all 17 `settings.llm.*` sites (including two writers)
  for no behavioral gain, against the smaller, UI-safe alternative.
- **Routing the OpenAI-compatible adapter through `eva.core.net`** — no test
  change required, but misclassifies loopback IPC as egress, which is
  precisely the distinction Batch 6 exists to keep correct.
- **`keyring` as a base dependency** — matches "OS-keychain" most literally,
  but grows the install footprint of every user for a milestone that ships
  local-only providers and needs no key yet.
- **Wiring real chain-walking fallback now** — the stated M7.4 deliverable,
  but doing it correctly requires changing `Assistant.preload()`'s pinned
  GPU-ownership load order and event bracketing (ADR-015 §5, §4), which no
  approved decision authorized changing.

## Related

- ADR-002 (llama.cpp runtime) — amended: `LlamaCppLLM` now also implements
  `LocalWeights`, documented as an additive fact, not a redesign.
- ADR-013 (LLM runtime installation) — amended: the installation story is
  unchanged; the factory that constructs the installed engine now takes
  `AppPaths` instead of a resolved `Path`.
- ADR-015 (deterministic runtime configuration) — amended: §4's "engine
  ports expose `device`" is narrowed to "`LocalWeights`-implementing ports
  expose `device`; a non-local port reports `engine_device() == "remote"`
  instead."
- Supersedes the "Secret storage" entry previously reserved as a standalone
  ADR-032 in the ADR index — folded into this ADR because H6 landed as part
  of the same coordinated change, not a separate one.
