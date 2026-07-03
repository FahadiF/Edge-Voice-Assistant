"""VAD engine registry (ADR-010): id → zero-arg factory."""

from __future__ import annotations

from collections.abc import Callable

from eva.core.registry import Registry
from eva.vad.base import VADEngine

vad_registry: Registry[Callable[[], VADEngine]] = Registry("vad-engine")


def register_builtins() -> None:
    """Idempotent registration of the engines that ship with the platform."""
    if "silero" not in vad_registry:
        from eva.vad.silero import SileroVAD

        vad_registry.register("silero", SileroVAD)


def create_vad(engine_id: str) -> VADEngine:
    register_builtins()
    return vad_registry.get(engine_id)()
