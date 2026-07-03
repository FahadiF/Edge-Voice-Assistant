"""Bounded frame queue between the audio callback and consumer threads.

Built on `collections.deque(maxlen=…)`: append and popleft are O(1) and
GIL-atomic, which is sufficient for the single-producer (audio callback) /
single-consumer (capture pipeline) pattern without taking locks in the
callback. When the consumer falls behind, the **oldest** frames are dropped —
for live audio, fresh data always wins — and the drop count is exposed for
diagnostics.
"""

from __future__ import annotations

from collections import deque

from eva.audio.frames import Frame


class FrameRing:
    def __init__(self, capacity_frames: int) -> None:
        if capacity_frames <= 0:
            raise ValueError("capacity_frames must be positive")
        self._frames: deque[Frame] = deque(maxlen=capacity_frames)
        self._capacity = capacity_frames
        self._pushed = 0
        self._popped = 0

    def push(self, frame: Frame) -> None:
        """Called from the audio callback. Never blocks; drops oldest when full."""
        self._pushed += 1
        self._frames.append(frame)

    def pop(self) -> Frame | None:
        """Called from the consumer thread. Returns None when empty."""
        try:
            frame = self._frames.popleft()
        except IndexError:
            return None
        self._popped += 1
        return frame

    def clear(self) -> None:
        self._frames.clear()

    @property
    def dropped(self) -> int:
        """Approximate count of frames lost to overflow."""
        return max(0, self._pushed - self._popped - len(self._frames))

    def __len__(self) -> int:
        return len(self._frames)
