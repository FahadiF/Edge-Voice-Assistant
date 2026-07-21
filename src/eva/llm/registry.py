"""LLM engine registry: id → factory(settings, model_path)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from eva.config.settings import Settings
from eva.core.registry import Registry
from eva.llm.base import LLMEngine

LLMFactory = Callable[[Settings, Path], LLMEngine]

llm_registry: Registry[LLMFactory] = Registry("llm-engine")


def _make_llamacpp(settings: Settings, model_path: Path) -> LLMEngine:
    from eva.llm.llamacpp import LlamaCppLLM

    return LlamaCppLLM(
        model_path,
        context_length=settings.llm.context_length,
        gpu_layers=settings.llm.gpu_layers,
        threads=settings.llm.threads,
        batch_size=settings.llm.batch_size,
        verbose=settings.developer.debug,
    )


def register_builtins() -> None:
    if "llamacpp" not in llm_registry:
        llm_registry.register("llamacpp", _make_llamacpp)


def create_llm(settings: Settings, model_path: Path) -> LLMEngine:
    register_builtins()
    return llm_registry.get(settings.llm.engine)(settings, model_path)
