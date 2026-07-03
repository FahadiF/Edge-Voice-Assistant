"""Aggregates 10 ms stream frames into the chunk size a VAD engine requires."""

from __future__ import annotations

import numpy as np

from eva.audio.frames import Frame


class FrameChunker:
    def __init__(self, chunk_samples: int) -> None:
        if chunk_samples <= 0:
            raise ValueError("chunk_samples must be positive")
        self._chunk_samples = chunk_samples
        self._buffer = np.empty(0, dtype=np.int16)

    def push(self, frame: Frame) -> list[Frame]:
        """Add one frame; return every complete chunk now available."""
        self._buffer = np.concatenate([self._buffer, frame])
        chunks: list[Frame] = []
        while self._buffer.shape[0] >= self._chunk_samples:
            chunks.append(self._buffer[: self._chunk_samples])
            self._buffer = self._buffer[self._chunk_samples :]
        return chunks

    def reset(self) -> None:
        self._buffer = np.empty(0, dtype=np.int16)
