"""VAD engine port."""

from __future__ import annotations

from abc import ABC, abstractmethod

from eva.audio.frames import Frame


class VADEngine(ABC):
    """Frame-synchronous speech-probability estimator.

    Implementations are stateful across consecutive chunks of one stream and
    must be `reset()` between independent streams. They are used from a single
    consumer thread; thread safety is not required.
    """

    @property
    @abstractmethod
    def chunk_samples(self) -> int:
        """Exact chunk size (samples at 16 kHz) `process()` requires."""

    @abstractmethod
    def process(self, chunk: Frame) -> float:
        """Speech probability [0, 1] for one chunk of `chunk_samples` samples."""

    @abstractmethod
    def reset(self) -> None:
        """Clear internal state before a new independent audio stream."""
