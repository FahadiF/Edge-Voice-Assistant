"""Silero VAD adapter (ONNX runtime via pysilero-vad; no torch dependency)."""

from __future__ import annotations

from pysilero_vad import SileroVoiceActivityDetector

from eva.audio.frames import Frame
from eva.core.errors import ModelError
from eva.vad.base import VADEngine


class SileroVAD(VADEngine):
    def __init__(self) -> None:
        try:
            self._detector = SileroVoiceActivityDetector()
        except Exception as exc:
            raise ModelError(f"Cannot initialize Silero VAD: {exc}") from exc
        self._chunk_samples = int(self._detector.chunk_samples())

    @property
    def chunk_samples(self) -> int:
        return self._chunk_samples

    def process(self, chunk: Frame) -> float:
        if chunk.shape[0] != self._chunk_samples:
            raise ValueError(
                f"Silero VAD requires {self._chunk_samples}-sample chunks, got {chunk.shape[0]}"
            )
        return float(self._detector.process_chunk(chunk.tobytes()))

    def reset(self) -> None:
        self._detector.reset()
