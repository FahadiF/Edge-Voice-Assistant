"""Kokoro TTS adapter via kokoro-onnx (ADR-004, ADR-012, ADR-018).

Runs on onnxruntime/CPU — perceptually identical to the PyTorch build and keeps
torch out of the product. Kokoro renders at 24 kHz; the adapter resamples to
the 16 kHz pipeline format.
"""

from __future__ import annotations

import asyncio
import contextlib
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

# Conversation language (BCP-47 primary subtag) → espeak-ng phonemizer voice
# (M5.6). Kokoro's text frontend is espeak-based: feeding non-English text
# through the default "en-us" phonemizer is why non-English replies sounded
# recognizably wrong. Languages absent here fall back to en-us phonemization
# (least-bad option: the model was trained mostly on these phoneme sets).
_ESPEAK_LANG = {
    "en": "en-us",
    "es": "es",
    "de": "de",
    "fi": "fi",
    "sv": "sv",
    "fr": "fr-fr",
    "it": "it",
    "pt": "pt-br",
    "hi": "hi",
    "bn": "bn",
}


def _espeak_lang(language: str | None) -> str:
    if language is None:
        return "en-us"
    return _ESPEAK_LANG.get(language.split("-")[0].lower(), "en-us")


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
        self._warm_up()

    def _warm_up(self) -> None:
        """Synthesize a throwaway word once at load (M5.6).

        onnxruntime finishes kernel initialization and memory-arena growth on
        the FIRST inference, not at session creation — without this, that
        one-time cost (high hundreds of ms) landed inside the first reply's
        time-to-first-audio. Loading happens on a preload worker thread in
        parallel with the LLM load (ADR-026), so warming here is free in
        wall-clock terms. Best-effort: a failure must never block loading."""
        assert self._kokoro is not None
        try:
            voices = self._kokoro.get_voices()
            if voices:
                self._kokoro.create("Hi.", voice=sorted(voices)[0], speed=1.0)
                logger.debug("Kokoro warm-up synthesis complete")
        except Exception:
            logger.debug("Kokoro warm-up failed (non-fatal)", exc_info=True)

    def unload(self) -> None:
        self._kokoro = None

    def synthesize(
        self, text: str, *, voice: str, speed: float = 1.0, language: str | None = None
    ) -> Frame:
        if self._kokoro is None:
            self.load()
        assert self._kokoro is not None
        text = text.strip()
        if not text:
            return np.zeros(0, dtype=np.int16)
        try:
            samples, sample_rate = self._kokoro.create(
                text, voice=voice, speed=speed, lang=_espeak_lang(language)
            )
        except Exception as exc:
            raise ModelError(f"Kokoro synthesis failed: {exc}") from exc
        pcm = float_to_int16(np.asarray(samples, dtype=np.float32))
        return resample_int16(pcm, int(sample_rate), _PIPELINE_RATE)

    def synthesize_stream(
        self, text: str, *, voice: str, speed: float = 1.0, language: str | None = None
    ) -> Iterator[Frame]:
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
        agen = self._kokoro.create_stream(
            text, voice=voice, speed=speed, lang=_espeak_lang(language)
        )
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
            # Cleanup must never crash the speak worker or leak the loop
            # (M5.5, ADR-026). This runs on the single thread that owns the
            # whole stream (the orchestrator's per-stream executor), so the
            # loop is guaranteed NOT to be running here — but guard anyway:
            # run_until_complete on a running loop is the one call that can
            # never be allowed.
            with contextlib.suppress(Exception):
                if not loop.is_running():
                    loop.run_until_complete(agen.aclose())
            with contextlib.suppress(Exception):
                if not loop.is_running():
                    loop.run_until_complete(loop.shutdown_asyncgens())
            with contextlib.suppress(Exception):
                loop.close()

    def voices(self) -> list[str]:
        if self._kokoro is None:
            self.load()
        assert self._kokoro is not None
        try:
            return sorted(self._kokoro.get_voices())
        except Exception:  # capability discovery must not break the pipeline
            return []
