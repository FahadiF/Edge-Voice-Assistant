"""faster-whisper (CTranslate2) ASR adapter (ADR-003)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from eva.asr.base import ASREngine, TranscriptionResult
from eva.audio.frames import Frame, int16_to_float
from eva.core.errors import ModelError

logger = logging.getLogger(__name__)


class FasterWhisperASR(ASREngine):
    """Whisper via CTranslate2. `model` is a size or path ("small", "base", …)."""

    def __init__(
        self,
        model: str,
        *,
        device: str = "auto",
        compute_type: str = "auto",
        download_root: Path | None = None,
    ) -> None:
        self._model_name = model
        self._device = device
        self._compute_type = "int8" if compute_type == "auto" else compute_type
        self._download_root = download_root
        self._model: Any = None

    def load(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel

        root = str(self._download_root) if self._download_root else None
        attempts = (
            [("cuda", self._compute_type), ("cpu", "int8")]
            if self._device in ("auto", "cuda")
            else [("cpu", self._compute_type)]
        )
        # Offline-first (M5.7): once the model is cached, huggingface_hub still
        # makes a HEAD request per file on every load to check for updates
        # unless told the files are local. Try a fully offline load first; only
        # if the model isn't cached (first run) do we permit the network. This
        # keeps a fully-installed EVA from touching Hugging Face at startup.
        last_error: Exception | None = None
        for local_files_only in (True, False):
            if not local_files_only:
                logger.info(
                    "faster-whisper '%s' not in local cache — downloading", self._model_name
                )
            for device, compute_type in attempts:
                try:
                    self._model = WhisperModel(
                        self._model_name,
                        device=device,
                        compute_type=compute_type,
                        download_root=root,
                        local_files_only=local_files_only,
                    )
                    self.device = device
                    logger.info(
                        "faster-whisper '%s' loaded (%s, %s%s)",
                        self._model_name,
                        device,
                        compute_type,
                        ", offline" if local_files_only else "",
                    )
                    return
                except Exception as exc:  # not cached, cuda missing, bad type, …
                    # The offline pass failing is expected on first run — stay
                    # quiet there and let the online pass report real trouble.
                    if not local_files_only:
                        logger.warning("faster-whisper load failed on %s: %s", device, exc)
                    last_error = exc
        raise ModelError(f"Cannot load faster-whisper '{self._model_name}': {last_error}")

    def unload(self) -> None:
        self._model = None

    def transcribe(self, audio: Frame, language: str | None = None) -> TranscriptionResult:
        if self._model is None:
            self.load()
        assert self._model is not None
        segments, info = self._model.transcribe(
            int16_to_float(audio),
            language=language,
            beam_size=1,  # greedy: ~2x faster than default beam 5; quality parity on clean speech
            condition_on_previous_text=False,  # avoids repetition loops on short utterances
            vad_filter=False,  # VAD already applied upstream
        )
        text = "".join(segment.text for segment in segments).strip()
        return TranscriptionResult(text=text, language=getattr(info, "language", None))
