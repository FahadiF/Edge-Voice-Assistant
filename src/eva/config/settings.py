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
    mic_gain: Annotated[float, Field(ge=0.0, le=4.0)] = 1.0
    speaker_volume: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0
    echo_cancellation: bool = True
    noise_suppression: bool = True
    auto_gain_control: bool = True
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
    barge_in_enabled: bool = True
    barge_in_confirm_ms: Annotated[int, Field(ge=50, le=2000)] = Field(
        200, description="Confirmed speech required to trigger barge-in during playback"
    )


# ──────────────────────── Models ────────────────────────


class ASRSettings(_Section):
    engine: str = Field("faster-whisper", description="ASR engine id (registry key)")
    model: str = Field("faster-whisper/small", description="Installed ASR model id")
    language: str | None = Field(None, description="ISO 639-1 hint; None = auto-detect")
    device: Literal["auto", "cuda", "cpu"] = "auto"
    compute_type: Literal["auto", "int8", "float16", "float32"] = "auto"
    partial_transcripts: bool = Field(
        True, description="Transcribe in-progress utterances for live feedback"
    )
    partial_interval_ms: Annotated[int, Field(ge=300, le=5000)] = Field(
        1200, description="How often to refresh the partial transcript"
    )


class LLMSettings(_Section):
    engine: str = Field("llamacpp", description="LLM engine id (registry key)")
    model: str = Field("qwen3.5-4b-instruct-q4_k_m", description="Installed LLM model id")
    context_length: Annotated[int, Field(ge=512, le=131072)] = 8192
    gpu_layers: Annotated[int, Field(ge=-1, le=200)] = Field(
        -1, description="-1 = offload as many layers as fit (auto)"
    )
    threads: Annotated[int, Field(ge=0, le=128)] = Field(0, description="0 = auto (physical cores)")
    batch_size: Annotated[int, Field(ge=1, le=4096)] = 512


class TTSSettings(_Section):
    engine: str = Field("kokoro", description="TTS engine id (registry key)")
    voice: str = Field("af_heart", description="Voice id within the active engine")
    speed: Annotated[float, Field(ge=0.5, le=2.0)] = 1.0
    pitch: Annotated[float, Field(ge=0.5, le=2.0)] = Field(
        1.0, description="Engines without pitch control ignore this"
    )
    streaming: bool = Field(True, description="Sentence-chunked synthesis during generation")


# ──────────────────────── Conversation ────────────────────────


class ConversationSettings(_Section):
    system_prompt: str = Field(
        "You are a helpful voice assistant. Answer conversationally and concisely — "
        "one to three short sentences unless the user asks for detail.",
        description="Base system prompt (personas layer on top of this)",
    )
    persona: str = Field("default", description="Active persona id")
    memory_enabled: bool = True
    max_history_turns: Annotated[int, Field(ge=1, le=200)] = Field(
        20, description="Turns kept verbatim before summarization kicks in"
    )
    temperature: Annotated[float, Field(ge=0.0, le=2.0)] = 0.4
    top_p: Annotated[float, Field(ge=0.0, le=1.0)] = 0.9
    max_tokens: Annotated[int, Field(ge=16, le=8192)] = 512
    stop_sequences: list[str] = Field(default_factory=list)


# ──────────────────────── Server / UI / Developer ────────────────────────


class ServerSettings(_Section):
    host: str = Field("127.0.0.1", description="Bind address; localhost-only by default")
    port: Annotated[int, Field(ge=1024, le=65535)] = 8765


class UISettings(_Section):
    theme: Literal["dark", "light", "system"] = "system"
    scale: Annotated[float, Field(ge=0.75, le=2.0)] = 1.0
    reduced_motion: bool = False


class DeveloperSettings(_Section):
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = Field(False, description="Emit structured JSON logs to file")
    metrics_enabled: bool = True


# ──────────────────────── Root ────────────────────────


class Settings(_Section):
    """Root settings document. Persisted as JSON; edited by UI, API, or hand."""

    schema_version: int = SETTINGS_SCHEMA_VERSION
    profile: str = Field("auto", description="Hardware profile id, or 'auto' to follow detection")
    audio: AudioSettings = Field(default_factory=AudioSettings)
    vad: VADSettings = Field(default_factory=VADSettings)
    asr: ASRSettings = Field(default_factory=ASRSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    tts: TTSSettings = Field(default_factory=TTSSettings)
    conversation: ConversationSettings = Field(default_factory=ConversationSettings)
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
