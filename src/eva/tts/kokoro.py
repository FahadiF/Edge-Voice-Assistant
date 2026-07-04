"""Kokoro TTS adapter via kokoro-onnx (ADR-004, ADR-012, ADR-018).

Runs on onnxruntime/CPU — perceptually identical to the PyTorch build and keeps
torch out of the product. Kokoro renders at 24 kHz; the adapter resamples to
the 16 kHz pipeline format.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from eva.audio.frames import Frame, float_to_int16
from eva.audio.resample import resample_int16
from eva.core.errors import ModelError
from eva.tts.base import TTSEngine

logger = logging.getLogger(__name__)

_PIPELINE_RATE = 16_000


class KokoroTTS(TTSEngine):
    def __init__(self, model_path: Path, voices_path: Path) -> None:
        self._model_path = model_path
        self._voices_path = voices_path
        self._kokoro: Any = None

    def load(self) -> None:
        if self._kokoro is not None:
            return
        for path in (self._model_path, self._voices_path):
            if not path.exists():
                raise ModelError(f"Kokoro model file not found: {path}")
        try:
            from kokoro_onnx import Kokoro

            self._kokoro = Kokoro(str(self._model_path), str(self._voices_path))
        except Exception as exc:
            raise ModelError(f"Cannot load Kokoro TTS: {exc}") from exc
        self.device = "cpu"
        logger.info("Kokoro TTS loaded (%s)", self._model_path.name)

    def unload(self) -> None:
        self._kokoro = None

    def synthesize(self, text: str, *, voice: str, speed: float = 1.0) -> Frame:
        if self._kokoro is None:
            self.load()
        assert self._kokoro is not None
        text = text.strip()
        if not text:
            return np.zeros(0, dtype=np.int16)
        try:
            samples, sample_rate = self._kokoro.create(text, voice=voice, speed=speed)
        except Exception as exc:
            raise ModelError(f"Kokoro synthesis failed: {exc}") from exc
        pcm = float_to_int16(np.asarray(samples, dtype=np.float32))
        return resample_int16(pcm, int(sample_rate), _PIPELINE_RATE)

    def synthesize_stream(self, text: str, *, voice: str, speed: float = 1.0) -> Iterator[Frame]:
        """Yield PCM chunks as kokoro-onnx's phoneme-batch streaming produces them.

        Drives `Kokoro.create_stream()` (an async generator) one `__anext__()`
        at a time over a dedicated event loop. The loop and its background
        synthesis task persist across calls to this generator's `next()`, so
        batch N+1 keeps rendering while the caller consumes/plays batch N.
        """
        if self._kokoro is None:
            self.load()
        assert self._kokoro is not None
        text = text.strip()
        if not text:
            return
        loop = asyncio.new_event_loop()
        agen = self._kokoro.create_stream(text, voice=voice, speed=speed)
        try:
            while True:
                try:
                    samples, sample_rate = loop.run_until_complete(agen.__anext__())
                except StopAsyncIteration:
                    break
                except Exception as exc:
                    raise ModelError(f"Kokoro streaming synthesis failed: {exc}") from exc
                pcm = float_to_int16(np.asarray(samples, dtype=np.float32))
                yield resample_int16(pcm, int(sample_rate), _PIPELINE_RATE)
        finally:
            loop.run_until_complete(agen.aclose())
            loop.close()

    def voices(self) -> list[str]:
        if self._kokoro is None:
            self.load()
        assert self._kokoro is not None
        try:
            return sorted(self._kokoro.get_voices())
        except Exception:  # capability discovery must not break the pipeline
            return []
