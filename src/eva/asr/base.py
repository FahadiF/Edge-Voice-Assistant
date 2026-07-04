"""ASR engine port.

`transcribe()` is a blocking call executed by the orchestrator in a worker
thread. Cancellation model: utterance transcription is short (hundreds of ms);
stale results are dropped by epoch rather than interrupted mid-decode. Engines
whose decode is long-running should still check nothing — keeping the port
minimal — because staleness is enforced one level up.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict

from eva.audio.frames import Frame


class TranscriptionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    language: str | None = None


class ASREngine(ABC):
    """Utterance-level speech recognizer over 16 kHz mono int16 audio."""

    device: str = "unloaded"
    """Device the model actually landed on ("cuda"/"cpu"); set by load()."""

    @abstractmethod
    def load(self) -> None:
        """Load model weights. Idempotent; called before first transcribe."""

    @abstractmethod
    def unload(self) -> None:
        """Release model resources (hot-swap support)."""

    @abstractmethod
    def transcribe(self, audio: Frame, language: str | None = None) -> TranscriptionResult:
        """Transcribe one utterance. `audio` may be partial (for live partials)."""
