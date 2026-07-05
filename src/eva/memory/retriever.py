"""Brute-force numpy semantic retrieval (ADR-020).

No vector database: at personal-assistant scale, a single matrix-vector
cosine-similarity product is faster than the LLM/ASR/TTS stages it feeds.
`MemoryRetriever` stays a real port so a future FAISS/sqlite-vec adapter is a
drop-in replacement if scale ever demands it — see ADR-020's rationale.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from eva.memory.base import MemoryRetriever, MemoryStore
from eva.memory.models import MemorySearchResult


class NumpyMemoryRetriever(MemoryRetriever):
    def __init__(
        self,
        store: MemoryStore,
        *,
        recency_half_life_days: float = 14.0,
        pinned_boost: float = 0.3,
        favorite_boost: float = 0.15,
        scan_limit: int = 2000,
    ) -> None:
        self._store = store
        self._half_life_days = recency_half_life_days
        self._pinned_boost = pinned_boost
        self._favorite_boost = favorite_boost
        self._scan_limit = scan_limit

    def retrieve(
        self,
        query_vector: bytes,
        *,
        top_k: int,
        conversation_id: str | None = None,
    ) -> list[MemorySearchResult]:
        rows = self._store.embeddings_for(conversation_id, limit=self._scan_limit)
        if not rows:
            return []

        # Guard against a stale embedding dimension left over from a prior
        # embedding-model choice (e.g. the user switched models): keep only
        # the dimension the current query actually has.
        query = np.frombuffer(query_vector, dtype=np.float32)
        rows = [r for r in rows if r[2] == query.shape[0]]
        if not rows:
            return []

        turn_ids = [r[0] for r in rows]
        matrix = np.frombuffer(b"".join(r[1] for r in rows), dtype=np.float32).reshape(
            len(rows), query.shape[0]
        )
        # Embeddings are already L2-normalized at write time (EmbeddingProvider
        # contract), so the dot product alone is cosine similarity.
        similarities = matrix @ query

        # One bulk fetch, not one query per candidate (measured: an N+1
        # get_turn() loop here made retrieval scale linearly with total
        # embedded turns instead of staying flat — see eva.benchmark.memory).
        turns_by_id = {turn.id: turn for turn in self._store.get_turns(turn_ids) if turn.id}

        now = datetime.now(UTC)
        scored: list[MemorySearchResult] = []
        for turn_id, similarity in zip(turn_ids, similarities, strict=True):
            turn = turns_by_id.get(turn_id)
            if turn is None:
                continue  # forgotten between embeddings_for() and here
            age_days = max((now - turn.created_at).total_seconds() / 86400.0, 0.0)
            recency = (
                0.5 ** (age_days / self._half_life_days) if self._half_life_days > 0 else 1.0
            )
            boost = (self._pinned_boost if turn.pinned else 0.0) + (
                self._favorite_boost if turn.favorite else 0.0
            )
            score = float(similarity) * recency + boost
            scored.append(MemorySearchResult(turn=turn, score=score, match_reason="semantic"))

        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]
