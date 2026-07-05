"""Memory subsystem benchmark — real measurements, not estimates (M4 Part 16).

No embedding model download required for the storage/retrieval numbers:
synthetic pre-normalized vectors exercise the exact same
`NumpyMemoryRetriever`/`MemoryStore` code paths a real embedding would feed
(ADR-020 §5 — the retrieval algorithm doesn't care where the vector came
from). Context-composition timing uses the real `ContextBuilder` with a
fixed-vector fake `EmbeddingProvider` so the number reflects SQLite +
retrieval + formatting overhead, not embedding-model inference time (a
separate, already-benchmarked concern — see `eva.benchmark.pipeline`).
"""

from __future__ import annotations

import logging
import time

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict

from eva.config.settings import Settings
from eva.conversation.context_builder import ContextBuilder
from eva.embedding.base import EmbeddingProvider
from eva.memory.base import MemoryStore
from eva.memory.models import Speaker
from eva.memory.retriever import NumpyMemoryRetriever

logger = logging.getLogger(__name__)


class MemoryBenchmarkReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    turn_count: int
    embedding_dim: int
    db_size_bytes: int
    search_text_ms_p50: float
    search_text_ms_p95: float
    retrieval_ms_p50: float
    retrieval_ms_p95: float
    context_build_ms_p50: float
    context_build_ms_p95: float

    def render(self) -> str:
        lines = [
            f"Synthetic turns:        {self.turn_count}",
            f"Embedding dimension:    {self.embedding_dim}",
            f"Database size:          {self.db_size_bytes / 1_048_576:.2f} MB",
            "",
            f"{'Keyword search (FTS)':<28} p50 {self.search_text_ms_p50:>7.2f} ms   "
            f"p95 {self.search_text_ms_p95:>7.2f} ms",
            f"{'Semantic retrieval':<28} p50 {self.retrieval_ms_p50:>7.2f} ms   "
            f"p95 {self.retrieval_ms_p95:>7.2f} ms",
            f"{'Context composition':<28} p50 {self.context_build_ms_p50:>7.2f} ms   "
            f"p95 {self.context_build_ms_p95:>7.2f} ms",
        ]
        return "\n".join(lines)


def _percentile(samples: list[float], pct: float) -> float:
    if not samples:
        return 0.0
    return float(np.percentile(samples, pct))


def _random_unit_vector(rng: np.random.Generator, dim: int) -> npt.NDArray[np.float32]:
    vec = rng.standard_normal(dim).astype(np.float32)
    norm: npt.NDArray[np.float32] = vec / np.linalg.norm(vec)
    return norm


class _FixedDimEmbeddingProvider(EmbeddingProvider):
    """Stands in for a real embedding model in the context-composition
    benchmark: same random-unit-vector output, none of the ONNX/tokenizer
    overhead — isolates SQLite + retrieval + formatting cost specifically."""

    def __init__(self, dim: int, rng: np.random.Generator) -> None:
        self.dim = dim
        self._rng = rng
        self.device = "cpu"

    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass

    def embed(self, text: str) -> npt.NDArray[np.float32]:
        return _random_unit_vector(self._rng, self.dim)


def run_memory_benchmark(
    store: MemoryStore,
    *,
    turn_count: int = 1000,
    turns_per_conversation: int = 40,
    embedding_dim: int = 384,
    search_rounds: int = 20,
    seed: int = 0,
) -> MemoryBenchmarkReport:
    """Populate `store` with `turn_count` synthetic turns spread across many
    `turns_per_conversation`-sized conversations (a real usage pattern —
    sessions end and new ones start; history doesn't accumulate as one
    ever-growing conversation), then measure real search/retrieval/context-
    composition latency against it. Retrieval and context composition are
    measured unscoped (`conversation_id=None`), matching `ContextBuilder`'s
    actual behavior (ADR-021: semantic memory recalls *past* conversations,
    not just the active one) — measured against a single giant conversation
    instead, SQLite's query planner cannot use `embeddings`' own primary-key
    order to satisfy the recency `LIMIT` and falls back to sorting every
    matching row, which is a different (and see this module's git history —
    much slower) case than what the live pipeline hits.

    `store` should be empty/dedicated — this benchmark's synthetic data is
    not meant to coexist with real conversations in the same database.
    """
    rng = np.random.default_rng(seed)
    conversation_ids: list[str] = []
    remaining = turn_count
    while remaining > 0:
        conversation = store.start_conversation(title="benchmark")
        conversation_ids.append(conversation.id)
        for i in range(min(turns_per_conversation, remaining)):
            speaker: Speaker = "user" if i % 2 == 0 else "assistant"
            turn = store.add_turn(conversation.id, speaker, f"synthetic turn number {i}")
            assert turn.id is not None
            vector = _random_unit_vector(rng, embedding_dim)
            store.store_embedding(turn.id, "benchmark-model", vector.tobytes(), embedding_dim)
        remaining -= turns_per_conversation
    active_conversation_id = conversation_ids[-1]

    retriever = NumpyMemoryRetriever(store)
    search_times: list[float] = []
    retrieval_times: list[float] = []
    for _ in range(search_rounds):
        start = time.perf_counter()
        store.search_text("synthetic", limit=20)
        search_times.append((time.perf_counter() - start) * 1000)

        query = _random_unit_vector(rng, embedding_dim).tobytes()
        start = time.perf_counter()
        retriever.retrieve(query, top_k=5, conversation_id=None)
        retrieval_times.append((time.perf_counter() - start) * 1000)

    builder = ContextBuilder(
        Settings(),
        store,
        retriever=retriever,
        embedding_provider=_FixedDimEmbeddingProvider(embedding_dim, rng),
    )
    build_times: list[float] = []
    for _ in range(search_rounds):
        start = time.perf_counter()
        builder.build(active_conversation_id, "what did we discuss earlier?")
        build_times.append((time.perf_counter() - start) * 1000)

    stats = store.stats()
    return MemoryBenchmarkReport(
        turn_count=turn_count,
        embedding_dim=embedding_dim,
        db_size_bytes=stats.db_size_bytes,
        search_text_ms_p50=_percentile(search_times, 50),
        search_text_ms_p95=_percentile(search_times, 95),
        retrieval_ms_p50=_percentile(retrieval_times, 50),
        retrieval_ms_p95=_percentile(retrieval_times, 95),
        context_build_ms_p50=_percentile(build_times, 50),
        context_build_ms_p95=_percentile(build_times, 95),
    )
