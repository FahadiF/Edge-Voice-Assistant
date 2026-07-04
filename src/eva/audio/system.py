"""AudioSystem: composition root of the audio subsystem.

Wires processor + playback queue + duplex stream + VAD + segmenter + capture
pipeline from `Settings`, and is the single object later milestones (and the
CLI demos) interact with: enqueue speech, stop speech, receive capture events.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from eva.audio.capture import CapturePipeline
from eva.audio.duplex import DuplexAudioStream
from eva.audio.playback import PlaybackQueue
from eva.audio.processor import create_processor
from eva.audio.ring import FrameRing
from eva.audio.segmenter import SegmenterEvent, SpeechSegmenter
from eva.config.settings import Settings
from eva.vad.registry import create_vad

logger = logging.getLogger(__name__)

_CAPTURE_RING_SECONDS = 2.0


class AudioSystem:
    def __init__(
        self,
        settings: Settings,
        on_event: Callable[[SegmenterEvent], None],
        *,
        enable_raw_tap: bool = False,
    ) -> None:
        self._settings = settings
        self.processor = create_processor(settings.audio)
        self.playback = PlaybackQueue(fade_ms=settings.audio.fade_out_ms)
        self.capture_ring = FrameRing(int(_CAPTURE_RING_SECONDS * 100))  # 10 ms frames
        self.raw_tap = FrameRing(int(_CAPTURE_RING_SECONDS * 100)) if enable_raw_tap else None
        self.stream = DuplexAudioStream(
            self.processor,
            self.playback,
            self.capture_ring,
            input_device=settings.audio.input_device,
            output_device=settings.audio.output_device,
            mic_gain=settings.audio.mic_gain,
            speaker_volume=settings.audio.speaker_volume,
            raw_tap=self.raw_tap,
        )
        vad = create_vad(settings.vad.engine)
        partial_interval = (
            settings.asr.partial_interval_ms if settings.asr.partial_transcripts else None
        )
        segmenter = SpeechSegmenter(settings.vad, partial_interval_ms=partial_interval)
        self.pipeline = CapturePipeline(
            self.capture_ring,
            vad,
            segmenter,
            playback_active=lambda: self.playback.is_active,
            on_event=on_event,
        )

    def start(self) -> None:
        logger.info("Audio system starting (processor: %s)", self.processor.name)
        self.stream.start()
        self.pipeline.start()

    def stop(self) -> None:
        self.pipeline.stop()
        self.stream.stop()

    def say(self, pcm: npt.NDArray[np.int16]) -> None:
        """Queue PCM for playback (16 kHz mono int16)."""
        self.playback.enqueue(pcm)
        self.playback.flush_pending()

    def stop_speaking(self) -> None:
        """Fade out and flush current playback (the barge-in action)."""
        self.playback.stop()

    @property
    def is_speaking(self) -> bool:
        return self.playback.is_active
