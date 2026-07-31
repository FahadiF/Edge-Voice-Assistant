"""llama.cpp LLM adapter (ADR-002): GGUF models, streaming, per-token abort."""

from __future__ import annotations

import logging
import os
import sys
import threading
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any

from eva.core.errors import ModelError
from eva.core.events import FinishReason
from eva.core.tools import ToolDefinition
from eva.llm.base import ChatMessage, GenerationOutcome, GenerationParams, LLMEngine

logger = logging.getLogger(__name__)

_dll_paths_registered = False


def _register_cuda_dll_paths() -> None:
    """Make pip-installed CUDA runtime DLLs findable by llama.cpp on Windows.

    The CUDA wheels load `llama.dll` with legacy search semantics (PATH-based),
    so `os.add_dll_directory` is not sufficient — the nvidia wheel bin dirs must
    be on PATH before the first import.
    """
    global _dll_paths_registered
    if _dll_paths_registered:
        return
    _dll_paths_registered = True
    if sys.platform == "win32":
        site_packages = Path(sys.prefix) / "Lib" / "site-packages"
        nvidia_bins = [str(p) for p in (site_packages / "nvidia").glob("*/bin") if p.is_dir()]
        if nvidia_bins:
            os.environ["PATH"] = os.pathsep.join([*nvidia_bins, os.environ.get("PATH", "")])


class LlamaCppLLM(LLMEngine):
    def __init__(
        self,
        model_path: Path,
        *,
        context_length: int = 8192,
        gpu_layers: int = -1,
        threads: int = 0,
        batch_size: int = 512,
        verbose: bool = False,
    ) -> None:
        self._model_path = model_path
        self._context_length = context_length
        self._gpu_layers = gpu_layers
        self._threads = threads
        self._batch_size = batch_size
        self._verbose = verbose
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
                # Quiet by default; when developer.debug is on, llama.cpp prints
                # its load report — including the actual "offloaded N/M layers to
                # GPU" line, the one datum that tells you if GPU offload really
                # happened (self.device is only build capability, not proof).
                verbose=self._verbose,
            )
        except Exception as exc:
            raise ModelError(f"Cannot load LLM '{self._model_path.name}': {exc}") from exc
        import llama_cpp

        gpu_active = self._gpu_layers != 0 and bool(llama_cpp.llama_supports_gpu_offload())
        self.device = "cuda" if gpu_active else "cpu"
        logger.info(
            "llama.cpp loaded %s (ctx=%d, gpu_layers=%d, device=%s; device=cuda means the "
            "build supports offload, not that layers were offloaded — set developer.debug "
            "to see the actual offload count)",
            self._model_path.name,
            self._context_length,
            self._gpu_layers,
            self.device,
        )

    def unload(self) -> None:
        if self._llama is not None:
            with self._infer_lock:
                self._llama = None

    def count_tokens(self, text: str) -> int:
        """Exact count from llama.cpp's tokenizer (falls back to the port's
        estimate before the model is loaded)."""
        if self._llama is None:
            return super().count_tokens(text)
        return len(self._llama.tokenize(text.encode("utf-8"), add_bos=False, special=False))

    def stream(
        self,
        messages: list[ChatMessage],
        params: GenerationParams,
        should_abort: Callable[[], bool],
        *,
        tools: tuple[ToolDefinition, ...] = (),
    ) -> Generator[str, None, GenerationOutcome]:
        if self._llama is None:
            self.load()
        assert self._llama is not None
        if tools:
            # Say so rather than generating a tool-less answer that looks like
            # the model declined to use them. Recognising Qwen's call markup
            # and reporting `tool_calls` is a later milestone.
            logger.warning(
                "This adapter cannot offer tools to the model yet; generating "
                "without the %d offered (%s)",
                len(tools),
                ", ".join(t.name for t in tools),
            )
        with self._infer_lock:
            completion = self._llama.create_chat_completion(
                # `exclude_none` keeps the payload exactly what it was before
                # `call_id` existed: an unset correlation id is absent from
                # the dict the chat template renders, not a null in it.
                messages=[m.model_dump(exclude_none=True) for m in messages],
                temperature=params.temperature,
                top_p=params.top_p,
                max_tokens=params.max_tokens,
                stop=list(params.stop) or None,
                stream=True,
            )
            # llama.cpp reports why it stopped on the LAST chunk only, and
            # "length" (the max_tokens ceiling) is indistinguishable from a
            # finished reply unless it is read. Default to "stop" so an
            # adapter or stub that never reports one is treated as complete.
            reason: FinishReason = "stop"
            try:
                for chunk in completion:
                    if should_abort():
                        logger.debug("LLM generation aborted")
                        return GenerationOutcome(reason="abort")
                    choice = chunk["choices"][0]
                    reported = choice.get("finish_reason")
                    if reported in ("stop", "length"):
                        reason = reported
                    token = choice["delta"].get("content")
                    if token:
                        yield token
            finally:
                # Ensure llama.cpp's generator cleanup runs even on abort.
                close = getattr(completion, "close", None)
                if close is not None:
                    close()
            if reason == "length":
                logger.info(
                    "Generation hit the %d-token ceiling; reply is truncated", params.max_tokens
                )
            # No tool calls yet: this adapter does not recognise them.
            return GenerationOutcome(reason=reason)
