"""LLM engine registry: id → factory(settings, paths).

Batch 8 (C1): the factory signature converges on the pattern
`eva.asr.registry` already uses — `Callable[[Settings, AppPaths], LLMEngine]`,
each adapter resolving its own path/config internally — instead of the caller
(`build_assistant`) resolving a model path and handing it in. This is what
lets a remote provider (no local file at all) register the same way a local
one does.
"""

from __future__ import annotations

from collections.abc import Callable

from eva.config.paths import AppPaths
from eva.config.settings import Settings
from eva.core.registry import Registry
from eva.llm.base import LLMEngine

LLMFactory = Callable[[Settings, AppPaths], LLMEngine]

llm_registry: Registry[LLMFactory] = Registry("llm-engine")

#: Registered engine ids backed by a local, on-disk model — as opposed to a
#: remote/API-backed one (`openai-compatible`), which has no file for
#: `ModelManager` to install or verify. `required_models()` (`eva.engine`)
#: reads this to decide whether the LLM model belongs in the readiness/
#: preflight check at all; nothing here instantiates an engine to find out.
LOCAL_ENGINE_IDS = frozenset({"llamacpp"})


def _make_llamacpp(settings: Settings, paths: AppPaths) -> LLMEngine:
    from eva.llm.llamacpp import LlamaCppLLM
    from eva.models.manager import ModelManager

    local = settings.llm.providers.local
    model_path = ModelManager(paths).files_for(settings.llm.model)["model"]
    return LlamaCppLLM(
        model_path,
        context_length=local.context_length,
        gpu_layers=local.gpu_layers,
        threads=local.threads,
        batch_size=local.batch_size,
        verbose=settings.developer.debug,
    )


def _make_openai_compatible(settings: Settings, paths: AppPaths) -> LLMEngine:
    from eva.llm.openai_compat import OpenAICompatibleLLM

    cfg = settings.llm.providers.openai_compatible
    return OpenAICompatibleLLM(base_url=cfg.base_url, model=cfg.model, api_key_ref=cfg.api_key_ref)


def register_builtins() -> None:
    if "llamacpp" not in llm_registry:
        llm_registry.register("llamacpp", _make_llamacpp)
    if "openai-compatible" not in llm_registry:
        llm_registry.register("openai-compatible", _make_openai_compatible)


def create_llm(settings: Settings, paths: AppPaths) -> LLMEngine:
    register_builtins()
    return llm_registry.get(settings.llm.engine)(settings, paths)
