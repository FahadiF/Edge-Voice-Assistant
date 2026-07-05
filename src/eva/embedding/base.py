"""Embedding provider port (ADR-020).

`embed()` is blocking, called from a worker thread by callers (mirrors every
other engine port in this codebase — `TTSEngine.synthesize()`,
`ASREngine.transcribe()` — engines stay simple synchronous functions;
callers own concurrency).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class EmbeddingProvider(ABC):
    device: str = "unloaded"
    """Device the model actually landed on ("cuda"/"cpu"); set by load()."""

    dim: int = 0
    """Output vector dimensionality; set by load()."""

    @abstractmethod
    def load(self) -> None:
        """Load model weights. Idempotent."""

    @abstractmethod
    def unload(self) -> None:
        """Release model resources (hot-swap support)."""

    @abstractmethod
    def embed(self, text: str) -> npt.NDArray[np.float32]:
        """Return an L2-normalized embedding vector of length `self.dim`."""
