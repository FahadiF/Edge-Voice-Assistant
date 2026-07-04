"""TTS engine port.

`synthesize()` is blocking, called per sentence segment in a worker thread —
sentence-granular synthesis is what gives the pipeline streaming playback and
sub-sentence interruption without requiring engines to stream internally.
Output is pipeline-format audio (16 kHz mono int16); engines resample at their
own boundary.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from eva.audio.frames import Frame


class TTSEngine(ABC):
    device: str = "unloaded"
    """Device the model actually landed on ("cuda"/"cpu"); set by load()."""

    @abstractmethod
    def load(self) -> None:
        """Load model weights. Idempotent."""

    @abstractmethod
    def unload(self) -> None:
        """Release model resources (hot-swap support)."""

    @abstractmethod
    def synthesize(self, text: str, *, voice: str, speed: float = 1.0) -> Frame:
        """Render one text segment to 16 kHz mono int16 PCM."""

    @abstractmethod
    def voices(self) -> list[str]:
        """Available voice ids (capability discovery for the UI)."""
