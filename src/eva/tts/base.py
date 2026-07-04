"""TTS engine port.

`synthesize()` is blocking, called per sentence segment in a worker thread —
sentence-granular synthesis is what gives the pipeline streaming playback and
sub-sentence interruption without requiring engines to stream internally.
Output is pipeline-format audio (16 kHz mono int16); engines resample at their
own boundary.

`synthesize_stream()` (ADR-018) is an additive, non-abstract capability: an
engine that can render sub-sentence chunks incrementally (e.g. Kokoro via
kokoro-onnx's phoneme-batch streaming) overrides it to cut both time-to-first-
audio and barge-in latency. The default implementation yields exactly one
chunk via `synthesize()`, so every existing and future adapter behaves
unchanged unless it explicitly opts in.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

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

    def synthesize_stream(self, text: str, *, voice: str, speed: float = 1.0) -> Iterator[Frame]:
        """Yield PCM chunks as they become available.

        Default: one chunk via `synthesize()`. Override for engines that can
        render incrementally.
        """
        yield self.synthesize(text, voice=voice, speed=speed)

    @abstractmethod
    def voices(self) -> list[str]:
        """Available voice ids (capability discovery for the UI)."""
