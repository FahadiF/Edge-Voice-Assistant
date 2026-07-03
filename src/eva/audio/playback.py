"""Playback queue consumed by the duplex audio callback.

Producers enqueue arbitrary-length int16 PCM; the callback pulls exactly one
10 ms frame per tick. `stop()` starts a short linear fade (default 40 ms) and
then flushes — an instant cut clicks audibly, a fade does not. This is the
mechanism barge-in uses to silence the assistant.

Locking: a single short mutex guards the frame deque and fade state. The
callback holds it for microseconds (deque ops + slice copy); producers never
hold it while synthesizing.
"""

from __future__ import annotations

import threading
from collections import deque

import numpy as np
import numpy.typing as npt

from eva.audio.frames import FRAME_MS, FRAME_SAMPLES, Frame, silence_frame


class PlaybackQueue:
    def __init__(self, fade_ms: int = 40) -> None:
        self._fade_frames_total = max(1, fade_ms // FRAME_MS)
        self._lock = threading.Lock()
        self._frames: deque[Frame] = deque()
        self._pending: Frame | None = None  # partial tail of the last enqueue
        self._fading = False
        self._fade_frames_left = 0

    def enqueue(self, pcm: npt.NDArray[np.int16]) -> None:
        """Append PCM (any length) for playback; splits into 10 ms frames."""
        if pcm.ndim != 1:
            raise ValueError("PlaybackQueue expects mono 1-D int16 PCM")
        with self._lock:
            if self._fading:
                # A stop is in progress; new audio belongs to a newer turn and
                # must wait until the fade completes and the queue is flushed.
                # Callers coordinate ordering via the turn epoch (ADR-006).
                self._frames.clear()
                self._fading = False
            if self._pending is not None:
                pcm = np.concatenate([self._pending, pcm])
                self._pending = None
            full, rest = divmod(pcm.shape[0], FRAME_SAMPLES)
            for i in range(full):
                self._frames.append(pcm[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES])
            if rest:
                self._pending = pcm[full * FRAME_SAMPLES :].copy()

    def flush_pending(self) -> None:
        """Zero-pad and queue the partial tail (call at end of an utterance)."""
        with self._lock:
            if self._pending is not None:
                frame = silence_frame()
                frame[: self._pending.shape[0]] = self._pending
                self._frames.append(frame)
                self._pending = None

    def stop(self) -> None:
        """Begin fade-out; the queue flushes itself when the fade completes."""
        with self._lock:
            if not self._frames and self._pending is None:
                return
            self._pending = None
            if not self._fading:
                self._fading = True
                self._fade_frames_left = self._fade_frames_total

    def next_frame(self) -> Frame:
        """Called by the audio callback every 10 ms. Always returns a frame."""
        with self._lock:
            if self._fading:
                if self._fade_frames_left <= 0 or not self._frames:
                    self._frames.clear()
                    self._fading = False
                    return silence_frame()
                frame = self._frames.popleft()
                # Linear ramp across the remaining fade window.
                start = self._fade_frames_left / self._fade_frames_total
                end = (self._fade_frames_left - 1) / self._fade_frames_total
                ramp = np.linspace(start, end, FRAME_SAMPLES, dtype=np.float32)
                self._fade_frames_left -= 1
                faded: Frame = (frame.astype(np.float32) * ramp).astype(np.int16)
                if self._fade_frames_left <= 0:
                    self._frames.clear()
                    self._fading = False
                return faded
            frame_or_none = self._frames.popleft() if self._frames else None
        return frame_or_none if frame_or_none is not None else silence_frame()

    @property
    def is_active(self) -> bool:
        """True while there is audio queued or a fade in progress."""
        with self._lock:
            return bool(self._frames) or self._fading or self._pending is not None

    def queued_seconds(self) -> float:
        with self._lock:
            return len(self._frames) * FRAME_MS / 1000.0
