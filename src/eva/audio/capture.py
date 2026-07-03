"""Capture pipeline: cleaned frames → VAD chunks → segmenter events.

Runs on its own consumer thread so VAD inference (~1 ms per chunk on CPU) and
event handling never execute inside the audio callback. Event callbacks are
invoked on this thread; handlers must be quick or hand off to their own queue.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

from eva.audio.chunker import FrameChunker
from eva.audio.frames import rms_dbfs
from eva.audio.ring import FrameRing
from eva.audio.segmenter import SegmenterEvent, SpeechSegmenter
from eva.vad.base import VADEngine

logger = logging.getLogger(__name__)

_IDLE_SLEEP_S = 0.002


class CapturePipeline:
    def __init__(
        self,
        capture_ring: FrameRing,
        vad: VADEngine,
        segmenter: SpeechSegmenter,
        *,
        playback_active: Callable[[], bool],
        on_event: Callable[[SegmenterEvent], None],
    ) -> None:
        self._ring = capture_ring
        self._vad = vad
        self._segmenter = segmenter
        self._chunker = FrameChunker(vad.chunk_samples)
        self._playback_active = playback_active
        self._on_event = on_event
        self._thread: threading.Thread | None = None
        self._stop_flag = threading.Event()
        self._level_dbfs = -120.0

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_flag.clear()
        self._vad.reset()
        self._chunker.reset()
        self._segmenter.reset()
        self._thread = threading.Thread(target=self._run, name="eva-capture", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_flag.set()
        self._thread.join(timeout=2.0)
        self._thread = None

    @property
    def level_dbfs(self) -> float:
        """Most recent input level (for UI meters)."""
        return self._level_dbfs

    def _run(self) -> None:
        logger.debug("Capture pipeline started")
        while not self._stop_flag.is_set():
            frame = self._ring.pop()
            if frame is None:
                time.sleep(_IDLE_SLEEP_S)
                continue
            self._level_dbfs = rms_dbfs(frame)
            for chunk in self._chunker.push(frame):
                prob = self._vad.process(chunk)
                for event in self._segmenter.feed(chunk, prob, self._playback_active()):
                    try:
                        self._on_event(event)
                    except Exception:
                        logger.exception("Capture event handler failed")
        logger.debug("Capture pipeline stopped")
