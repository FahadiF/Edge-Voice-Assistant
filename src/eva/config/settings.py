"""Application settings schema and persistence.

Design rules:

- Components (LLM/ASR/TTS/VAD engines, models, voices) are referenced by string
  **ids resolved through registries at runtime** — never by class or module name.
  This is what lets the UI swap any component without code or config-file edits.
- Every field has a sensible default; a missing or partial settings file always
  yields a valid ``Settings``. Unknown keys are rejected (typo protection).
- The file on disk is plain JSON so the settings UI, the REST API, and a human
  with a text editor all edit the same document.
- Numeric fields carry validation bounds; the API/UI derives widget ranges from
  this schema (single source of truth via ``Settings.model_json_schema()``).

Defaults marked "(thesis-tuned)" carry over experimentally validated values from
the thesis prototype.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from eva.core.errors import ConfigError

SETTINGS_SCHEMA_VERSION = 6


class _Section(BaseModel):
    """Base for all settings sections: strict keys, validate on assignment."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# ──────────────────────── Audio ────────────────────────


class AudioSettings(_Section):
    input_device: str | None = Field(None, description="Input device name; None = system default")
    output_device: str | None = Field(None, description="Output device name; None = system default")
    sample_rate: Literal[16000] = Field(16000, description="Pipeline sample rate (Hz)")
    mic_gain: Annotated[float, Field(ge=0.0, le=4.0)] = Field(
        1.0, description="Microphone gain multiplier"
    )
    speaker_volume: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        1.0, description="Playback volume"
    )
    echo_cancellation: bool = Field(True, description="WebRTC acoustic echo cancellation")
    noise_suppression: bool = Field(True, description="WebRTC noise suppression")
    auto_gain_control: bool = Field(True, description="WebRTC automatic gain control")
    fade_out_ms: Annotated[int, Field(ge=10, le=500)] = Field(
        40, description="Playback fade-out on interruption (click-free stop)"
    )
    duplex_mode: Literal["full-duplex", "half-duplex", "push-to-talk"] = Field(
        "full-duplex", description="Fallback ladder position (see ADR-005)"
    )


class VADSettings(_Section):
    engine: str = Field("silero", description="VAD engine id (registry key)")
    threshold: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        0.5, description="Speech probability threshold (thesis-tuned)"
    )
    silence_timeout_ms: Annotated[int, Field(ge=100, le=5000)] = Field(
        800, description="Base end-of-utterance silence window"
    )
    min_speech_ms: Annotated[int, Field(ge=50, le=2000)] = Field(
        380, description="Noise gate: shorter bursts are discarded (thesis-tuned)"
    )
    max_utterance_s: Annotated[int, Field(ge=5, le=120)] = Field(
        30, description="Hard recording timeout"
    )
    barge_in_enabled: bool = Field(True, description="Allow interrupting the assistant by voice")
    barge_in_confirm_ms: Annotated[int, Field(ge=50, le=2000)] = Field(
        200, description="Confirmed speech required to trigger barge-in during playback"
    )


# ──────────────────────── Models ────────────────────────


class ASRSettings(_Section):
    engine: str = Field("faster-whisper", description="ASR engine id (registry key)")
    model: str = Field("faster-whisper/small", description="Installed ASR model id")
    language: str | None = Field(
        None, description="ISO 639-1 hint; None = follow conversation language"
    )
    device: Literal["auto", "cuda", "cpu"] = Field(
        "auto", description="Inference device; auto tries CUDA, then CPU"
    )
    compute_type: Literal["auto", "int8", "float16", "float32"] = Field(
        "auto", description="Numeric precision; auto = int8"
    )
    partial_transcripts: bool = Field(
        True, description="Transcribe in-progress utterances for live feedback"
    )
    partial_interval_ms: Annotated[int, Field(ge=300, le=5000)] = Field(
        1200, description="How often to refresh the partial transcript"
    )


class LocalProviderSettings(_Section):
    """Runtime knobs for the local llama.cpp provider (Batch 8 / H7)."""

    context_length: Annotated[int, Field(ge=512, le=131072)] = Field(
        8192, description="Context window in tokens (VRAM grows with this)"
    )
    gpu_layers: Annotated[int, Field(ge=-1, le=200)] = Field(
        -1, description="-1 = offload as many layers as fit (auto)"
    )
    threads: Annotated[int, Field(ge=0, le=128)] = Field(0, description="0 = auto (physical cores)")
    batch_size: Annotated[int, Field(ge=1, le=4096)] = Field(
        512, description="Prompt-processing batch size"
    )


class OpenAICompatibleProviderSettings(_Section):
    """Config for the OpenAI-compatible adapter (Ollama/LM Studio/vLLM/...).

    Local-only this milestone (Batch 8 decision 8.3): `base_url` must resolve
    to loopback — the adapter's constructor enforces this, not just the UI.
    """

    base_url: str = Field(
        "http://127.0.0.1:11434/v1",
        description="OpenAI-compatible API base URL (loopback only this milestone)",
    )
    model: str = Field("", description="Model name as the endpoint expects it (e.g. an Ollama tag)")
    api_key_ref: str | None = Field(
        None,
        description="Opaque secret reference (never a raw key), resolved via eva.core.secrets",
    )


class ProvidersSettings(_Section):
    """Per-provider configuration, one fixed field per provider (Batch 8
    decision 8.2) — not a `dict[str, ...]`. The Settings UI renders nested
    settings only by resolving a `$ref` to a named schema; a dict of objects
    has no such schema for the UI to resolve, so a fixed-key model is what
    keeps this batch's schema change UI-safe with zero frontend edits."""

    local: LocalProviderSettings = Field(default_factory=LocalProviderSettings)
    openai_compatible: OpenAICompatibleProviderSettings = Field(
        default_factory=OpenAICompatibleProviderSettings
    )


class LLMSettings(_Section):
    engine: str = Field("llamacpp", description="LLM engine id (registry key)")
    model: str = Field("qwen3.5-4b-instruct-q4_k_m", description="Installed LLM model id")
    providers: ProvidersSettings = Field(default_factory=ProvidersSettings)
    chain: list[str] = Field(
        default_factory=lambda: ["local"],
        description=(
            "Provider fallback order, tried on construction/load failure only "
            "(never mid-stream — ADR-006 turn epochs do not model a mid-turn "
            "provider swap). Schema-only this batch: `engine` above remains "
            "the sole active-provider selector; automatic chain-walking is a "
            "future addition, not wired into build_assistant yet."
        ),
    )


class TTSSettings(_Section):
    engine: str = Field("kokoro", description="TTS engine id (registry key)")
    model: str = Field("kokoro-82m-v1.0", description="Installed TTS model id")
    voice: str | None = Field(
        None,
        description="Voice id within the active engine; None = follow the conversation language",
    )
    speed: Annotated[float, Field(ge=0.5, le=2.0)] = Field(
        1.0, description="Speech rate multiplier"
    )
    pitch: Annotated[float, Field(ge=0.5, le=2.0)] = Field(
        1.0, description="Pitch multiplier; engines without pitch control ignore this"
    )
    streaming: bool = Field(True, description="Sentence-chunked synthesis during generation")
    lazy_load: bool = Field(
        False,
        description=(
            "Skip loading TTS at engine start; load it on the first spoken reply instead. "
            "Faster startup for typed-chat-heavy use, at the cost of a one-time delay on "
            "the first reply that speaks"
        ),
    )


# ──────────────────────── Conversation ────────────────────────


class PersonaSettingsEntry(_Section):
    """A user-created persona, persisted in settings (ADR-022) and converted
    to a `PersonaProfile` by `eva.conversation.personas` at startup — kept
    here, not imported from the conversation subsystem, so `eva.config`
    stays a dependency leaf (subsystems depend on config, not vice versa)."""

    id: str = Field(description="Unique persona id (must not collide with a built-in id)")
    display_name: str = Field(description="Shown in the UI/CLI persona picker")
    system_prompt: str = Field(description="Base instruction defining this persona's behavior")
    verbosity: Literal["minimal", "concise", "normal", "detailed"] = Field(
        "normal", description="How much detail replies should include"
    )
    tone: str = Field("neutral", description="Free-text tone descriptor (e.g. 'warm', 'blunt')")
    reasoning_style: str = Field(
        "direct", description="Free-text reasoning-style descriptor (e.g. 'step-by-step')"
    )
    temperature_override: Annotated[float, Field(ge=0.0, le=2.0)] | None = Field(
        None, description="Override the conversation-level sampling temperature; None = inherit"
    )


class ConversationSettings(_Section):
    system_prompt: str = Field(
        "You are a helpful voice assistant. Answer conversationally and concisely — "
        "one to three short sentences unless the user asks for detail.",
        description="Base system prompt (personas and language notes layer on top)",
    )
    persona: str = Field("default", description="Active persona id")
    language: str = Field(
        "en", description="Conversation language (BCP-47 code from the language registry)"
    )
    max_history_turns: Annotated[int, Field(ge=1, le=200)] = Field(
        20, description="Turns kept verbatim before summarization kicks in"
    )
    temperature: Annotated[float, Field(ge=0.0, le=2.0)] = Field(
        0.4, description="Sampling temperature (higher = more varied replies)"
    )
    top_p: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        0.9, description="Nucleus-sampling probability mass"
    )
    max_tokens: Annotated[int, Field(ge=16, le=8192)] = Field(
        2048,
        description="Maximum tokens per reply (a ceiling, not a reservation — "
        "spoken answers stop far earlier)",
    )
    stop_sequences: list[str] = Field(
        default_factory=list, description="Extra sequences that end generation"
    )
    sentence_min_chars: Annotated[int, Field(ge=1, le=200)] = Field(
        12, description="Minimum segment length before speech starts (avoids fragment replies)"
    )
    sentence_max_chars: Annotated[int, Field(ge=20, le=2000)] = Field(
        50, description="Force a speakable split if generation runs on without punctuation"
    )
    first_sentence_min_chars: Annotated[int, Field(ge=1, le=200)] = Field(
        4,
        description=(
            "Minimum length for the first spoken segment of a turn only (M3). "
            "Largely superseded by low sentence_max_chars in M8 (A8), but retained "
            "for backwards compatibility."
        ),
    )
    active_profile_id: str | None = Field(
        None,
        description=(
            "Active user profile id (M4: nickname/preferences stored in the memory "
            "database, ADR-022) — None if no profile has been created yet"
        ),
    )
    custom_personas: list[PersonaSettingsEntry] = Field(
        default_factory=list,
        description=(
            "User-created personas, registered alongside the built-ins at startup "
            "(ADR-022) — persisted here because a persona is configuration, not "
            "conversation data"
        ),
    )


# ──────────────────────── Memory (M4, ADR-019/ADR-020) ────────────────────────


class MemorySettings(_Section):
    engine: str = Field("sqlite", description="Memory store engine id (registry key)")
    embedding_enabled: bool = Field(
        True,
        description=(
            "Compute and store embeddings for semantic search; if disabled or the "
            "embedding model is not installed, search falls back to keyword-only"
        ),
    )
    embedding_engine: str = Field(
        "onnx-embedding", description="Embedding provider engine id (registry key)"
    )
    embedding_model: str = Field(
        "all-minilm-l6-v2-onnx", description="Installed embedding model id"
    )
    retention_days: Annotated[int, Field(ge=1, le=36500)] | None = Field(
        None, description="Delete turns older than this many days; None = keep forever"
    )
    max_turns_per_conversation: Annotated[int, Field(ge=10, le=1_000_000)] | None = Field(
        None, description="Cap turns retained per conversation; None = unlimited"
    )
    auto_cleanup_enabled: bool = Field(
        False, description="Apply the retention policy automatically on engine start"
    )
    encrypt_at_rest: bool = Field(
        False,
        description=(
            "Reserved for a future encrypted-database adapter (ADR-019) — not yet "
            "implemented; enabling this has no effect today"
        ),
    )
    retrieval_top_k: Annotated[int, Field(ge=1, le=50)] = Field(
        5, description="Number of semantically relevant memories the Context Builder retrieves"
    )
    retrieval_min_similarity: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        0.30,
        description=(
            "Minimum raw cosine similarity for a memory to be recalled. "
            "Applied before recency weighting, which would otherwise make an "
            "old relevant memory score below a fresh irrelevant one"
        ),
    )
    retrieval_scan_limit: Annotated[int, Field(ge=100, le=100_000)] = Field(
        2000,
        description=(
            "Maximum embedded turns scored per semantic search, most-recent-first — "
            "bounds retrieval latency independent of total accumulated history"
        ),
    )
    max_memory_chars: Annotated[int, Field(ge=100, le=20_000)] = Field(
        2000, description="Character budget for retrieved-memory context (ADR-021)"
    )
    max_summary_chars: Annotated[int, Field(ge=100, le=20_000)] = Field(
        1000, description="Character budget for the included conversation summary (ADR-021)"
    )
    recency_half_life_days: Annotated[float, Field(gt=0.0, le=3650.0)] = Field(
        14.0, description="Retrieval recency-decay half-life: older memories score lower"
    )
    pinned_boost: Annotated[float, Field(ge=0.0, le=10.0)] = Field(
        0.3, description="Retrieval score bonus for pinned turns"
    )
    favorite_boost: Annotated[float, Field(ge=0.0, le=10.0)] = Field(
        0.15, description="Retrieval score bonus for favorited turns"
    )
    summarize_after_turns: Annotated[int, Field(ge=5, le=10_000)] = Field(
        40, description="Summarize a conversation once it exceeds this many turns"
    )


# ──────────────────────── Permissions ────────────────────────


class GeneralPermissions(_Section):
    internet: bool = Field(
        False, description="Allow internet access (coming soon — not available in this build)"
    )
    date_time: bool = Field(True, description="Share the current local date, time, and timezone")
    system_information: bool = Field(
        True, description="Share OS, CPU, GPU, RAM, and locale details"
    )


class FilesPermissions(_Section):
    read_files: bool = Field(
        False, description="Allow reading local files (coming soon — not available in this build)"
    )
    write_files: bool = Field(
        False, description="Allow writing local files (coming soon — not available in this build)"
    )


class DevicesPermissions(_Section):
    camera: bool = Field(
        False, description="Allow camera access (coming soon — not available in this build)"
    )
    microphone: bool = Field(
        True,
        description="Allow microphone capture — off makes this a typed-chat-only assistant",
    )


class ToolsPermissions(_Section):
    browser: bool = Field(
        False, description="Allow controlling a browser (coming soon — not available in this build)"
    )
    python: bool = Field(
        False, description="Allow executing Python code (coming soon — not available in this build)"
    )
    shell: bool = Field(
        False,
        description="Allow running shell commands (coming soon — not available in this build)",
    )
    plugins: bool = Field(True, description="Allow enabled plugins to contribute capabilities")


class PrivacyPermissions(_Section):
    remember_conversations: bool = Field(
        True,
        title="Save conversations to memory",
        description=(
            "When off, new conversation turns are not saved. Memories already "
            "stored stay available until you delete them"
        ),
    )
    learn_preferences: bool = Field(
        True,
        description="Reserved for automatic preference learning (not used in this build)",
    )


class PermissionsSettings(_Section):
    """What the assistant is allowed to know about / do on this machine
    (ADR-025, regrouped in M5.4). Read-only local facts default ON (they
    power questions like "what time is it?"); anything that *acts* on the
    system or leaves the device defaults OFF and, in this build, is also
    not implemented — the toggle is the contract future capability
    providers must respect. Enforced today: `general.*` (system-info
    prompt), `devices.microphone` (audio capture), and
    `privacy.remember_conversations` (turn storage)."""

    general: GeneralPermissions = Field(default_factory=GeneralPermissions)
    files: FilesPermissions = Field(default_factory=FilesPermissions)
    devices: DevicesPermissions = Field(default_factory=DevicesPermissions)
    tools: ToolsPermissions = Field(default_factory=ToolsPermissions)
    privacy: PrivacyPermissions = Field(default_factory=PrivacyPermissions)


# ──────────────────────── Server / UI / Developer ────────────────────────


class ServerSettings(_Section):
    host: str = Field("127.0.0.1", description="Bind address; localhost-only by default")
    port: Annotated[int, Field(ge=1024, le=65535)] = Field(8765, description="API server port")


class UISettings(_Section):
    theme: Literal["dark", "light", "system"] = Field("system", description="Color theme")
    scale: Annotated[float, Field(ge=0.75, le=2.0)] = Field(1.0, description="UI scale factor")
    reduced_motion: bool = Field(False, description="Disable non-essential animations")
    sync_text_to_speech: bool = Field(
        True,
        description=(
            "Reveal each sentence of a reply as it starts being spoken, so the text on "
            "screen stays in step with the voice (M7.1). Turn off to show text as soon as "
            "the model generates it — faster to read, but it runs ahead of the speech"
        ),
    )


class DeveloperSettings(_Section):
    debug: bool = Field(False, description="Verbose diagnostics for development")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        "INFO", description="Minimum level written to the log"
    )
    log_json: bool = Field(False, description="Emit structured JSON logs to file")
    metrics_enabled: bool = Field(True, description="Collect per-turn latency metrics")


class DesktopSettings(_Section):
    """Native desktop-shell behavior (M6, ADR-027). These only take effect
    when running through `eva-desktop`; the CLI/server ignore them. The
    window theme is not here — it reuses `ui.theme` (no duplication)."""

    close_to_tray: bool = Field(
        True, description="Closing the window hides it to the tray instead of quitting"
    )
    minimize_to_tray: bool = Field(False, description="Minimizing the window hides it to the tray")
    start_minimized: bool = Field(
        False, description="Launch hidden to the tray instead of showing the window"
    )
    start_with_os: bool = Field(
        False,
        description="Launch EVA automatically when you sign in (registers an OS start-up entry)",
    )
    auto_start_engine: bool = Field(
        False,
        description="Start the engine automatically once the desktop app launches",
    )
    notifications_enabled: bool = Field(
        True, description="Show desktop notifications for downloads, engine state, and errors"
    )
    hotkey_enabled: bool = Field(True, description="Enable the global push-to-talk / mute hotkey")
    hotkey_binding: str = Field(
        "ctrl+space",
        description="Global hotkey combination (e.g. 'ctrl+space', 'alt+space')",
    )
    hotkey_mode: Literal["push-to-talk", "push-to-mute", "toggle"] = Field(
        "push-to-talk",
        description=(
            "push-to-talk: listen only while held; push-to-mute: mute only while held; "
            "toggle: each press flips listening"
        ),
    )


class PluginsSettings(_Section):
    enabled: list[str] = Field(
        default_factory=list,
        description="Ids of plugins the user has explicitly enabled; a newly "
        "discovered plugin defaults to disabled until added here (ADR-011)",
    )


# ──────────────────────── Root ────────────────────────


class Settings(_Section):
    """Root settings document. Persisted as JSON; edited by UI, API, or hand."""

    schema_version: int = Field(
        SETTINGS_SCHEMA_VERSION, description="Settings document version (for migration)"
    )
    profile: str = Field(
        "balanced",
        description="Model preset id (balanced/fast/high-accuracy/low-memory/developer), "
        "or 'custom' when models are chosen manually",
    )
    audio: AudioSettings = Field(default_factory=AudioSettings)
    vad: VADSettings = Field(default_factory=VADSettings)
    asr: ASRSettings = Field(default_factory=ASRSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    tts: TTSSettings = Field(default_factory=TTSSettings)
    conversation: ConversationSettings = Field(default_factory=ConversationSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    permissions: PermissionsSettings = Field(default_factory=PermissionsSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    ui: UISettings = Field(default_factory=UISettings)
    developer: DeveloperSettings = Field(default_factory=DeveloperSettings)
    desktop: DesktopSettings = Field(default_factory=DesktopSettings)
    plugins: PluginsSettings = Field(default_factory=PluginsSettings)


def _migrate_raw(raw: Any) -> Any:
    """Upgrade an older settings document, dict-level, before validation —
    mirrors the memory database's numbered-migration pattern (ADR-019).

    Each transform below is gated on `from_version` individually (Batch 8
    decision 8.1), not on a single top-level "below current version" check.
    The single-gate design used through v5 meant bumping
    `SETTINGS_SCHEMA_VERSION` made every already-migrated document re-run
    every prior transform, keyed only on the old default *value* — a v5 user
    who deliberately re-set a field to what used to be the old default (e.g.
    `sentence_max_chars: 350`) would have had it silently rewritten the
    moment v6 shipped, with no version boundary to stop it. Gating each
    transform on the version it actually migrates *from* closes that gap:
    a v5 document skips every v1-v5 transform outright, regardless of its
    field values, and only the v5→v6 transform below ever touches it.

    v1 → v2 (M5.4): flat `permissions` keys become grouped sections
    (ADR-025 regroup); the dead `conversation.memory_enabled` flag moves to
    `permissions.privacy.remember_conversations` (now actually enforced).

    v2 → v3 (M7.3): `tts.voice` gains a `None` sentinel meaning "follow the
    conversation language". The old default `"af_heart"` was indistinguishable
    from a deliberate choice, so `effective_voice()` could not honour user
    selections. Documents still carrying the old default are rewritten to
    `None` — behaviour-preserving, because `"af_heart"` is exactly what the
    English profile resolves to, and non-English profiles were overriding it
    anyway. Any other value was necessarily set by hand and is left alone.

    v3 → v4 (M7.3): `conversation.max_tokens` rises from 512 to 2048. The cap
    is a ceiling rather than a reservation, so a short reply is unaffected,
    but 512 truncated generated code mid-file. Because `save_settings` writes
    every field, an existing install carries the old default explicitly and
    would keep truncating unless it is rewritten; only that value is migrated,
    and a custom one is preserved.

    v4 → v5 (M8/A8): `conversation.sentence_max_chars` drops from 350 to 50.
    The lower ceiling enables punctuation-aware bounded chunking that halves
    worst-case TTFA and cuts playback starvation by ~76%.  Only the old
    default (350) is migrated; any other value was deliberately set by the
    user and is preserved.

    v5 → v6 (Batch 8, M7.4): flat `llm.{context_length,gpu_layers,threads,
    batch_size}` fields nest under `llm.providers.local`; `llm.chain` is
    added, defaulting to `["local"]`. `llm.engine`/`llm.model` are left at
    the top level untouched (decision 8.2) — nothing reads them differently
    after this migration than before it.
    """
    if not isinstance(raw, dict):
        return raw
    from_version = raw.get("schema_version", 1)
    if from_version >= SETTINGS_SCHEMA_VERSION:
        return raw

    if from_version < 2:
        perms = raw.get("permissions")
        if isinstance(perms, dict) and "general" not in perms:
            raw["permissions"] = {
                "general": {
                    "internet": perms.get("internet", False),
                    "date_time": perms.get("date_time", True) or perms.get("timezone", True),
                    "system_information": any(
                        perms.get(key, True) for key in ("cpu", "gpu", "ram", "os", "locale")
                    ),
                },
                "files": {"read_files": perms.get("local_files", False), "write_files": False},
                "devices": {"camera": perms.get("camera", False), "microphone": True},
                "tools": {
                    "browser": perms.get("browser", False),
                    "python": perms.get("python", False),
                    "shell": perms.get("shell", False),
                    "plugins": perms.get("plugins", True),
                },
                "privacy": {"remember_conversations": True, "learn_preferences": True},
            }
        conversation = raw.get("conversation")
        if isinstance(conversation, dict) and "memory_enabled" in conversation:
            remember = conversation.pop("memory_enabled")
            raw.setdefault("permissions", {}).setdefault("privacy", {})[
                "remember_conversations"
            ] = remember

    if from_version < 3:
        tts = raw.get("tts")
        if isinstance(tts, dict) and tts.get("voice") == "af_heart":
            tts["voice"] = None

    if from_version < 4:
        conversation = raw.get("conversation")
        if isinstance(conversation, dict) and conversation.get("max_tokens") == 512:
            conversation["max_tokens"] = 2048

    if from_version < 5:
        conversation = raw.get("conversation")
        if isinstance(conversation, dict) and conversation.get("sentence_max_chars") == 350:
            conversation["sentence_max_chars"] = 50

    if from_version < 6:
        llm = raw.get("llm")
        if isinstance(llm, dict):
            local = {
                key: llm.pop(key)
                for key in ("context_length", "gpu_layers", "threads", "batch_size")
                if key in llm
            }
            providers = llm.setdefault("providers", {})
            if isinstance(providers, dict):
                providers.setdefault("local", {}).update(local)
            llm.setdefault("chain", ["local"])

    raw["schema_version"] = SETTINGS_SCHEMA_VERSION
    return raw


def load_settings(path: Path) -> Settings:
    """Load settings from JSON, falling back to defaults if the file is absent.

    A malformed file raises :class:`ConfigError` rather than silently resetting —
    losing a user's configuration is worse than failing loudly. Older schema
    versions are migrated in memory (persisted on the next save).
    """
    if not path.exists():
        return Settings()
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"Cannot read settings file {path}: {exc}") from exc
    try:
        return Settings.model_validate(_migrate_raw(raw))
    except ValueError as exc:
        raise ConfigError(f"Invalid settings in {path}: {exc}") from exc


def save_settings(settings: Settings, path: Path) -> None:
    """Atomically persist settings as pretty-printed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    try:
        tmp.write_text(
            settings.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        tmp.replace(path)
    except OSError as exc:
        raise ConfigError(f"Cannot write settings file {path}: {exc}") from exc
