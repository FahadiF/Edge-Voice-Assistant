"""ONNX embedding adapter (ADR-020): all-MiniLM-L6-v2 via onnxruntime.

No PyTorch involved (ADR-012 stays intact): tokenization uses HuggingFace's
Rust-backed `tokenizers` library, inference is a plain ONNX Runtime session,
and pooling (masked mean over token embeddings, then L2-normalize) is
straightforward numpy — the same "small model, no ML framework" pattern
Kokoro/Silero already use, applied to a new capability.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from eva.core.errors import ModelError
from eva.embedding.base import EmbeddingProvider

logger = logging.getLogger(__name__)

_MAX_SEQUENCE_LENGTH = 256
_DEFAULT_DIM = 384  # all-MiniLM-L6-v2's known hidden size


class OnnxEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_path: Path, tokenizer_path: Path) -> None:
        self._model_path = model_path
        self._tokenizer_path = tokenizer_path
        self._session: Any = None
        self._tokenizer: Any = None
        self._input_names: set[str] = set()

    def load(self) -> None:
        if self._session is not None:
            return
        for path in (self._model_path, self._tokenizer_path):
            if not path.exists():
                raise ModelError(f"Embedding model file not found: {path}")
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer

            self._tokenizer = Tokenizer.from_file(str(self._tokenizer_path))
            self._tokenizer.enable_truncation(max_length=_MAX_SEQUENCE_LENGTH)
            self._session = ort.InferenceSession(
                str(self._model_path), providers=["CPUExecutionProvider"]
            )
            self._input_names = {i.name for i in self._session.get_inputs()}
            output_dim = self._session.get_outputs()[0].shape[-1]
            self.dim = output_dim if isinstance(output_dim, int) else _DEFAULT_DIM
        except Exception as exc:
            raise ModelError(f"Cannot load embedding model: {exc}") from exc
        self.device = "cpu"
        logger.info("Embedding model loaded (%s)", self._model_path.name)

    def unload(self) -> None:
        self._session = None
        self._tokenizer = None

    def embed(self, text: str) -> npt.NDArray[np.float32]:
        if self._session is None:
            self.load()
        assert self._session is not None
        assert self._tokenizer is not None
        text = text.strip()
        if not text:
            return np.zeros(self.dim or _DEFAULT_DIM, dtype=np.float32)

        encoding = self._tokenizer.encode(text)
        input_ids = np.asarray([encoding.ids], dtype=np.int64)
        attention_mask = np.asarray([encoding.attention_mask], dtype=np.int64)
        feed: dict[str, npt.NDArray[np.int64]] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.zeros_like(input_ids)

        try:
            outputs = self._session.run(None, feed)
        except Exception as exc:
            raise ModelError(f"Embedding inference failed: {exc}") from exc

        token_embeddings = outputs[0][0]  # (seq_len, hidden) — batch size is always 1 here
        mask = attention_mask[0].astype(np.float32)[:, None]
        pooled = (token_embeddings * mask).sum(axis=0) / np.clip(mask.sum(), 1e-9, None)
        norm = np.linalg.norm(pooled)
        normalized = pooled / norm if norm > 0 else pooled
        result: npt.NDArray[np.float32] = np.asarray(normalized, dtype=np.float32)
        return result
