"""LLM registry factory tests.

Constructing `LlamaCppLLM` does not import the native `llama_cpp` runtime (that
is deferred to `.load()`), so the factory wiring is exercised headless. These
guard that engine settings — and crucially `developer.debug` → llama.cpp's
`verbose` load report — reach the adapter, so GPU-offload diagnostics can be
turned on without touching the native path.
"""

from __future__ import annotations

from pathlib import Path

from eva.config.settings import Settings
from eva.llm.llamacpp import LlamaCppLLM
from eva.llm.registry import create_llm


def _build(settings: Settings) -> LlamaCppLLM:
    llm = create_llm(settings, Path("model.gguf"))
    assert isinstance(llm, LlamaCppLLM)
    return llm


def test_factory_threads_engine_settings() -> None:
    settings = Settings()
    settings.llm.context_length = 4096
    settings.llm.gpu_layers = 20
    settings.llm.threads = 6
    settings.llm.batch_size = 256

    llm = _build(settings)

    assert llm._context_length == 4096
    assert llm._gpu_layers == 20
    assert llm._threads == 6
    assert llm._batch_size == 256


def test_verbose_follows_developer_debug() -> None:
    quiet = _build(Settings())
    assert quiet._verbose is False  # quiet by default (M5.7 clean-output behavior)

    debug_settings = Settings()
    debug_settings.developer.debug = True
    assert _build(debug_settings)._verbose is True  # opt-in llama.cpp load report
