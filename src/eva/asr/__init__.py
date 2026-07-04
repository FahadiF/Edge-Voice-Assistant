"""Speech recognition: port, registry, and built-in adapters."""

from eva.asr.base import ASREngine, TranscriptionResult
from eva.asr.registry import asr_registry, create_asr, register_builtins

__all__ = ["ASREngine", "TranscriptionResult", "asr_registry", "create_asr", "register_builtins"]
