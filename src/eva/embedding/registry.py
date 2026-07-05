"""Embedding provider registry: id -> factory(settings, model_files)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from eva.config.settings import Settings
from eva.core.errors import ModelError
from eva.core.registry import Registry
from eva.embedding.base import EmbeddingProvider

# Factories receive the resolved file paths of the active embedding model.
EmbeddingFactory = Callable[[Settings, dict[str, Path]], EmbeddingProvider]

embedding_registry: Registry[EmbeddingFactory] = Registry("embedding-engine")


def _make_onnx(_settings: Settings, files: dict[str, Path]) -> EmbeddingProvider:
    from eva.embedding.onnx import OnnxEmbeddingProvider

    try:
        return OnnxEmbeddingProvider(files["model"], files["tokenizer"])
    except KeyError as exc:
        raise ModelError(f"Embedding model requires a '{exc.args[0]}' model file") from exc


def register_builtins() -> None:
    if "onnx-embedding" not in embedding_registry:
        embedding_registry.register("onnx-embedding", _make_onnx)


def create_embedding_provider(settings: Settings, files: dict[str, Path]) -> EmbeddingProvider:
    register_builtins()
    return embedding_registry.get(settings.memory.embedding_engine)(settings, files)
