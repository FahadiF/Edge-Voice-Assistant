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

SETTINGS_SCHEMA_VERSION = 1


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


class LLMSettings(_Section):
    engine: str = Field("llamacpp", description="LLM engine id (registry key)")
    model: str = Field("qwen3.5-4b-instruct-q4_k_m", description="Installed LLM model id")
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


class TTSSettings(_Section):
    engine: str = Field("kokoro", description="TTS engine id (registry key)")
    model: str = Field("kokoro-82m-v1.0", description="Installed TTS model id")
    voice: str = Field(
        "af_heart", description="Voice id within the active engine (language may override)"
    )
    speed: Annotated[float, Field(ge=0.5, le=2.0)] = Field(
        1.0, description="Speech rate multiplier"
    )
    pitch: Annotated[float, Field(ge=0.5, le=2.0)] = Field(
        1.0, description="Pitch multiplier; engines without pitch control ignore this"
    )
    streaming: bool = Field(True, description="Sentence-chunked synthesis during generation")


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
    memory_enabled: bool = Field(True, description="Keep conversation history across turns")
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
        512, description="Maximum tokens per reply"
    )
    stop_sequences: list[str] = Field(
        default_factory=list, description="Extra sequences that end generation"
    )
    sentence_min_chars: Annotated[int, Field(ge=1, le=200)] = Field(
        12, description="Minimum segment length before speech starts (avoids fragment replies)"
    )
    sentence_max_chars: Annotated[int, Field(ge=50, le=2000)] = Field(
        350, description="Force a speakable split if generation runs on without punctuation"
    )
    first_sentence_min_chars: Annotated[int, Field(ge=1, le=200)] = Field(
        6,
        description=(
            "Minimum length for the first spoken segment of a turn only (M3: lower than "
            "sentence_min_chars to start audio sooner; later segments use sentence_min_chars)"
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


class PermissionsSettings(_Section):
    """What the assistant is allowed to know about / do on this machine
    (M5.3, ADR-025). Read-only local facts default ON (they power questions
    like "what time is it?"); anything that *acts* on the system or leaves
    the device defaults OFF and, in this build, is also not implemented —
    the toggle is the contract future capability providers must respect."""

    date_time: bool = Field(True, description="Share the current local date and time")
    timezone: bool = Field(True, description="Share the system timezone")
    locale: bool = Field(True, description="Share the system language/locale")
    cpu: bool = Field(True, description="Share CPU model and core count")
    gpu: bool = Field(True, description="Share GPU model and VRAM")
    ram: bool = Field(True, description="Share installed memory size")
    os: bool = Field(True, description="Share operating system name and version")
    internet: bool = Field(False, description="Allow internet access (not available in this build)")
    local_files: bool = Field(
        False, description="Allow reading local files (not available in this build)"
    )
    camera: bool = Field(False, description="Allow camera access (not available in this build)")
    clipboard: bool = Field(
        False, description="Allow clipboard access (not available in this build)"
    )
    browser: bool = Field(
        False, description="Allow controlling a browser (not available in this build)"
    )
    shell: bool = Field(
        False, description="Allow running shell commands (not available in this build)"
    )
    python: bool = Field(
        False, description="Allow executing Python code (not available in this build)"
    )
    plugins: bool = Field(True, description="Allow enabled plugins to contribute capabilities")


# ──────────────────────── Server / UI / Developer ────────────────────────


class ServerSettings(_Section):
    host: str = Field("127.0.0.1", description="Bind address; localhost-only by default")
    port: Annotated[int, Field(ge=1024, le=65535)] = Field(8765, description="API server port")


class UISettings(_Section):
    theme: Literal["dark", "light", "system"] = Field("system", description="Color theme")
    scale: Annotated[float, Field(ge=0.75, le=2.0)] = Field(1.0, description="UI scale factor")
    reduced_motion: bool = Field(False, description="Disable non-essential animations")


class DeveloperSettings(_Section):
    debug: bool = Field(False, description="Verbose diagnostics for development")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        "INFO", description="Minimum level written to the log"
    )
    log_json: bool = Field(False, description="Emit structured JSON logs to file")
    metrics_enabled: bool = Field(True, description="Collect per-turn latency metrics")


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


def load_settings(path: Path) -> Settings:
    """Load settings from JSON, falling back to defaults if the file is absent.

    A malformed file raises :class:`ConfigError` rather than silently resetting —
    losing a user's configuration is worse than failing loudly.
    """
    if not path.exists():
        return Settings()
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"Cannot read settings file {path}: {exc}") from exc
    try:
        return Settings.model_validate(raw)
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
