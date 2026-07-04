"""Speech synthesis: port, registry, and built-in adapters."""

from eva.tts.base import TTSEngine
from eva.tts.registry import create_tts, register_builtins, tts_registry

__all__ = ["TTSEngine", "create_tts", "register_builtins", "tts_registry"]
