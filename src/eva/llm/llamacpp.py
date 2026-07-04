"""llama.cpp LLM adapter (ADR-002): GGUF models, streaming, per-token abort."""

from __future__ import annotations

import logging
import os
import sys
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from eva.core.errors import ModelError
from eva.llm.base import ChatMessage, GenerationParams, LLMEngine

logger = logging.getLogger(__name__)

_dll_paths_registered = False


def _register_cuda_dll_paths() -> None:
    """Make pip-installed CUDA runtime DLLs findable by llama.cpp on Windows.

    The CUDA wheels load `llama.dll` with legacy search semantics (PATH-based),
    so `os.add_dll_directory` is not sufficient — the nvidia wheel bin dirs must
    be on PATH before the first import.
    """
    global _dll_paths_registered
    if _dll_paths_registered or sys.platform != "win32":
        return
    site_packages = Path(sys.prefix) / "Lib" / "site-packages"
    nvidia_bins = [str(p) for p in (site_packages / "nvidia").glob("*/bin") if p.is_dir()]
    if nvidia_bins:
        os.environ["PATH"] = os.pathsep.join([*nvidia_bins, os.environ.get("PATH", "")])
    _dll_paths_registered = True


class LlamaCppLLM(LLMEngine):
    def __init__(
        self,
        model_path: Path,
        *,
        context_length: int = 8192,
        gpu_layers: int = -1,
        threads: int = 0,
        batch_size: int = 512,
    ) -> None:
        self._model_path = model_path
        self._context_length = context_length
        self._gpu_layers = gpu_layers
        self._threads = threads
        self._batch_size = batch_size
        self._llama: Any = None
        # llama.cpp contexts are not thread-safe; generation calls are serialized.
        self._infer_lock = threading.Lock()

    def load(self) -> None:
        if self._llama is not None:
            return
        if not self._model_path.exists():
            raise ModelError(f"LLM model file not found: {self._model_path}")
        _register_cuda_dll_paths()
        try:
            from llama_cpp import Llama
        except Exception as exc:
            raise ModelError(f"llama.cpp runtime unavailable: {exc}") from exc
        try:
            self._llama = Llama(
                model_path=str(self._model_path),
                n_ctx=self._context_length,
                n_gpu_layers=self._gpu_layers,
                n_threads=self._threads or None,
                n_batch=self._batch_size,
                verbose=False,
            )
        except Exception as exc:
            raise ModelError(f"Cannot load LLM '{self._model_path.name}': {exc}") from exc
        logger.info(
            "llama.cpp loaded %s (ctx=%d, gpu_layers=%d)",
            self._model_path.name,
            self._context_length,
            self._gpu_layers,
        )

    def unload(self) -> None:
        if self._llama is not None:
            with self._infer_lock:
                self._llama = None

    def stream(
        self,
        messages: list[ChatMessage],
        params: GenerationParams,
        should_abort: Callable[[], bool],
    ) -> Iterator[str]:
        if self._llama is None:
            self.load()
        assert self._llama is not None
        with self._infer_lock:
            completion = self._llama.create_chat_completion(
                messages=[m.model_dump() for m in messages],
                temperature=params.temperature,
                top_p=params.top_p,
                max_tokens=params.max_tokens,
                stop=list(params.stop) or None,
                stream=True,
            )
            try:
                for chunk in completion:
                    if should_abort():
                        logger.debug("LLM generation aborted")
                        break
                    delta = chunk["choices"][0]["delta"]
                    token = delta.get("content")
                    if token:
                        yield token
            finally:
                # Ensure llama.cpp's generator cleanup runs even on abort.
                close = getattr(completion, "close", None)
                if close is not None:
                    close()
