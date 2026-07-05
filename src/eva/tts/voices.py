"""Voice registry (ADR-022, Part 7).

Not a new subsystem: a "voice" is metadata about a capability an existing
`TTSEngine` already exposes (`voices()`), not an independently pluggable
capability like ASR/LLM/TTS/VAD engines themselves. This module enriches
each engine's raw voice ids with display metadata — best-effort, never a
hard requirement for a voice to be usable.

Kokoro's voice ids follow a documented `{lang}{gender}_{name}` convention
(e.g. `af_heart` = American-English Female "Heart", `ef_dora` = Spanish
Female "Dora" — matching the id already used in `eva.conversation.language`).
Parsing it is enrichment, not correctness-critical: an unrecognized id or a
future engine's differently-shaped ids fall back to the bare id as the
display name, the same graceful-degradation shape as ADR-016's TTS-voice
fallback.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from eva.audio.frames import Frame
from eva.core.registry import Registry
from eva.tts.base import TTSEngine

_KOKORO_LANGUAGE_CODES: dict[str, str] = {
    "a": "en",  # American English
    "b": "en",  # British English
    "e": "es",
    "f": "fr",
    "h": "hi",
    "i": "it",
    "j": "ja",
    "p": "pt",
    "z": "zh",
}
_GENDER_CODES: dict[str, str] = {"f": "female", "m": "male"}


class VoiceInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    engine: str
    display_name: str
    language: str = "unknown"
    style_tag: str = ""


voice_registry: Registry[VoiceInfo] = Registry("voice")


def _parse_kokoro_voice_id(voice_id: str) -> tuple[str, str, str]:
    """Best-effort parse of `{lang}{gender}_{name}`; unknown shapes fall
    back to the bare id as the display name rather than raising."""
    prefix, _, name = voice_id.partition("_")
    if len(prefix) != 2 or not name:
        return "unknown", "", voice_id
    language = _KOKORO_LANGUAGE_CODES.get(prefix[0], "unknown")
    gender = _GENDER_CODES.get(prefix[1], "")
    return language, gender, name.replace("_", " ").title()


def _describe_voice(engine_id: str, voice_id: str) -> VoiceInfo:
    if engine_id == "kokoro":
        language, gender, display_name = _parse_kokoro_voice_id(voice_id)
        return VoiceInfo(
            id=voice_id,
            engine=engine_id,
            display_name=display_name,
            language=language,
            style_tag=gender,
        )
    return VoiceInfo(id=voice_id, engine=engine_id, display_name=voice_id)


def _registry_key(engine_id: str, voice_id: str) -> str:
    return f"{engine_id}:{voice_id}"


def register_voices_for_engine(engine_id: str, engine: TTSEngine) -> list[VoiceInfo]:
    """Populate the registry from `engine.voices()` (capability discovery).
    Safe to call repeatedly — re-registers with `replace=True` so a
    reloaded/updated engine's voice list stays current."""
    registered = []
    for voice_id in engine.voices():
        info = _describe_voice(engine_id, voice_id)
        voice_registry.register(_registry_key(engine_id, voice_id), info, replace=True)
        registered.append(info)
    return registered


def voices_for_engine(engine_id: str) -> list[VoiceInfo]:
    return [v for v in voice_registry.snapshot().values() if v.engine == engine_id]


def preview_text(
    engine: TTSEngine,
    voice_id: str,
    *,
    phrase: str = "Hello, this is a preview of my voice.",
    speed: float = 1.0,
) -> Frame:
    """Synthesize a short fixed phrase in `voice_id` — reuses the already-
    loaded engine's `synthesize()`, no new synthesis path."""
    return engine.synthesize(phrase, voice=voice_id, speed=speed)
