"""Playback queue consumed by the duplex audio callback.

Producers enqueue arbitrary-length int16 PCM; the callback pulls exactly one
10 ms frame per tick. `stop()` starts a short linear fade (default 40 ms) and
then flushes — an instant cut clicks audibly, a fade does not. This is the
mechanism barge-in uses to silence the assistant.

Markers (M7.1, ADR-028) let a producer ask "tell me when *this* audio is
actually heard". A marker is bound to the next frame the queue will append and
fires when the audio callback pulls that frame — i.e. on the audio clock, not
when the audio was queued. That distinction is the whole point: the queue
routinely holds seconds of synthesized lead, so "queued" and "audible" are far
apart. A fired marker therefore means "the user heard this": audio a `stop()`
cuts off fires nothing, including audio that would only start sounding during
the fade-out.

Locking: a single short mutex guards the frame deque, fade state, and marker
bookkeeping. The callback holds it for microseconds (deque ops + slice copy);
producers never hold it while synthesizing, and marker callbacks are invoked
*after* the lock is released.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from eva.audio.frames import FRAME_MS, FRAME_SAMPLES, Frame, silence_frame

logger = logging.getLogger(__name__)


class PlaybackQueue:
    def __init__(
        self, fade_ms: int = 40, *, on_marker: Callable[[object], None] | None = None
    ) -> None:
        self._fade_frames_total = max(1, fade_ms // FRAME_MS)
        self._lock = threading.Lock()
        self._frames: deque[Frame] = deque()
        self._pending: Frame | None = None  # partial tail of the last enqueue
        self._fading = False
        self._fade_frames_left = 0
        # Marker bookkeeping: (frame index, marker), in nondecreasing index
        # order. `_appended`/`_played` count frames over the queue's lifetime,
        # so a marker's index can be compared against the audio clock.
        self._markers: deque[tuple[int, object]] = deque()
        self._appended = 0
        self._played = 0
        self.on_marker: Callable[[object], None] | None = on_marker
        """Called (on the audio callback thread) when a marked frame plays.
        Must be trivial and non-blocking. Assign before playback starts."""

    def enqueue(self, pcm: npt.NDArray[np.int16], *, marker: object | None = None) -> None:
        """Append PCM (any length) for playback; splits into 10 ms frames.

        `marker` is bound to the next frame this queue appends, and is handed
        to `on_marker` when that frame is played (see the module docstring).
        """
        if pcm.ndim != 1:
            raise ValueError("PlaybackQueue expects mono 1-D int16 PCM")
        with self._lock:
            if self._fading:
                # A stop is in progress; new audio belongs to a newer turn and
                # must wait until the fade completes and the queue is flushed.
                # Callers coordinate ordering via the turn epoch (ADR-006).
                self._drop_queued()
                self._fading = False
            if marker is not None:
                # Registered after any flush above, so a superseding enqueue
                # never discards its own marker along with the old audio.
                self._markers.append((self._appended, marker))
            if self._pending is not None:
                pcm = np.concatenate([self._pending, pcm])
                self._pending = None
            full, rest = divmod(pcm.shape[0], FRAME_SAMPLES)
            for i in range(full):
                self._frames.append(pcm[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES])
            self._appended += full
            if rest:
                self._pending = pcm[full * FRAME_SAMPLES :].copy()

    def flush_pending(self) -> None:
        """Zero-pad and queue the partial tail (call at end of an utterance)."""
        with self._lock:
            if self._pending is not None:
                frame = silence_frame()
                frame[: self._pending.shape[0]] = self._pending
                self._frames.append(frame)
                self._appended += 1
                self._pending = None

    def stop(self) -> None:
        """Begin fade-out; the queue flushes itself when the fade completes."""
        with self._lock:
            # Every marker still waiting belongs to audio being cut off. The
            # fade-out is the tail of what the user already heard, so a
            # sentence that would first become audible *inside* it was never
            # really spoken — and neither was anything after it. Dropping them
            # here is what keeps "a marker fired" == "the user heard it".
            self._markers.clear()
            if not self._frames and self._pending is None:
                return
            self._pending = None
            if not self._fading:
                self._fading = True
                self._fade_frames_left = self._fade_frames_total

    def next_frame(self) -> Frame:
        """Called by the audio callback every 10 ms. Always returns a frame."""
        with self._lock:
            frame, due = self._pop_locked()
        # Outside the lock: a marker handler hands off to another thread (event
        # publication), which must never happen while the callback holds the
        # queue mutex. One call per marked frame, never per frame.
        for marker in due:
            self._fire(marker)
        return frame

    @property
    def is_active(self) -> bool:
        """True while there is audio queued or a fade in progress."""
        with self._lock:
            return bool(self._frames) or self._fading or self._pending is not None

    def queued_seconds(self) -> float:
        with self._lock:
            return len(self._frames) * FRAME_MS / 1000.0

    # ── internals (all called with the lock held, except _fire) ──

    def _pop_locked(self) -> tuple[Frame, list[object]]:
        """Produce the next frame plus the markers it completes."""
        if self._fading:
            if self._fade_frames_left <= 0 or not self._frames:
                self._drop_queued()
                self._fading = False
                return silence_frame(), []
            frame = self._frames.popleft()
            self._played += 1
            due = self._take_due_markers()  # before any flush below
            # Linear ramp across the remaining fade window.
            start = self._fade_frames_left / self._fade_frames_total
            end = (self._fade_frames_left - 1) / self._fade_frames_total
            ramp = np.linspace(start, end, FRAME_SAMPLES, dtype=np.float32)
            self._fade_frames_left -= 1
            faded: Frame = (frame.astype(np.float32) * ramp).astype(np.int16)
            if self._fade_frames_left <= 0:
                self._drop_queued()
                self._fading = False
            return faded, due
        if not self._frames:
            return silence_frame(), []
        frame = self._frames.popleft()
        self._played += 1
        return frame, self._take_due_markers()

    def _take_due_markers(self) -> list[object]:
        due: list[object] = []
        while self._markers and self._markers[0][0] < self._played:
            due.append(self._markers.popleft()[1])
        return due

    def _drop_queued(self) -> None:
        """Discard queued audio and the markers riding on it — a barge-in must
        not announce speech the user never heard. `_played` absorbs the dropped
        frames so markers registered afterwards still line up with the clock.
        """
        self._frames.clear()
        self._markers.clear()
        self._played = self._appended

    def _fire(self, marker: object) -> None:
        handler = self.on_marker
        if handler is None:
            return
        try:
            handler(marker)
        except Exception:  # a broken consumer must never break playback
            logger.exception("Playback marker handler failed")
