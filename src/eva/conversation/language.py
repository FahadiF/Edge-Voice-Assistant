"""Language registry: the multilingual foundation.

One `LanguageProfile` per supported language carries everything the pipeline
needs to speak it: the ASR language hint, a system-prompt note instructing the
LLM to respond in that language, and preferred TTS voices per engine. Adding a
language is a registry entry (ADR-010) — no pipeline changes.

Capability honesty: language support is the *intersection* of the active
models' abilities. Whisper and Qwen cover all registered languages; TTS is the
narrow end (Kokoro has no Finnish/Swedish/Bengali voice yet), so voice
resolution falls back to the configured default voice with a logged warning
rather than failing — the assistant still understands and answers in the
requested language, with an accented voice, until a native TTS model is added.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, ConfigDict

from eva.config.settings import Settings
from eva.core.registry import Registry

logger = logging.getLogger(__name__)


class LanguageProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str  # BCP-47 primary subtag ("en", "fi", …)
    display_name: str
    asr_language: str  # hint passed to the ASR engine (Whisper codes)
    prompt_note: str = ""  # appended to the system prompt for non-default languages
    tts_voices: dict[str, str] = {}  # engine id → preferred voice id

    def voice_for(self, engine_id: str, fallback: str) -> str:
        """Preferred voice for `engine_id`, or `fallback` with a warning."""
        voice = self.tts_voices.get(engine_id)
        if voice is not None:
            return voice
        if self.code != "en":
            logger.warning(
                "No %s voice registered for language '%s' — using '%s'. "
                "Speech will be understood and answered in %s, but synthesized "
                "with a non-native voice.",
                engine_id,
                self.code,
                fallback,
                self.display_name,
            )
        return fallback


language_registry: Registry[LanguageProfile] = Registry("language")


def register_builtin_languages() -> None:
    languages = (
        LanguageProfile(
            code="en",
            display_name="English",
            asr_language="en",
            tts_voices={"kokoro": "af_heart"},
        ),
        LanguageProfile(
            code="fi",
            display_name="Finnish",
            asr_language="fi",
            prompt_note="Vastaa aina suomeksi.",
        ),
        LanguageProfile(
            code="sv",
            display_name="Swedish",
            asr_language="sv",
            prompt_note="Svara alltid på svenska.",
        ),
        LanguageProfile(
            code="bn",
            display_name="Bengali",
            asr_language="bn",
            prompt_note="সবসময় বাংলায় উত্তর দিন।",
        ),
        LanguageProfile(
            code="de",
            display_name="German",
            asr_language="de",
            prompt_note="Antworte immer auf Deutsch.",
        ),
        LanguageProfile(
            code="es",
            display_name="Spanish",
            asr_language="es",
            prompt_note="Responde siempre en español.",
            tts_voices={"kokoro": "ef_dora"},
        ),
    )
    for lang in languages:
        if lang.code not in language_registry:
            language_registry.register(lang.code, lang)


def resolve_language(settings: Settings) -> LanguageProfile:
    """The active language profile (settings.conversation.language)."""
    register_builtin_languages()
    return language_registry.get(settings.conversation.language)


def effective_asr_language(settings: Settings, language: LanguageProfile) -> str:
    """Explicit ASR override wins; otherwise follow the conversation language."""
    return settings.asr.language or language.asr_language


def effective_voice(settings: Settings, language: LanguageProfile) -> str:
    return language.voice_for(settings.tts.engine, settings.tts.voice)
