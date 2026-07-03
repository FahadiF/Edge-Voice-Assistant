from __future__ import annotations

import numpy as np

from eva.audio.frames import FRAME_SAMPLES
from eva.audio.playback import PlaybackQueue


def _pcm(n_frames: int, value: int = 1000) -> np.ndarray:
    return np.full(n_frames * FRAME_SAMPLES, value, dtype=np.int16)


def test_enqueue_and_drain() -> None:
    q = PlaybackQueue()
    q.enqueue(_pcm(3))
    assert q.is_active
    assert q.queued_seconds() == 0.03
    for _ in range(3):
        frame = q.next_frame()
        assert frame[0] == 1000
    assert not q.is_active
    assert np.all(q.next_frame() == 0)  # silence when empty


def test_partial_tail_flushed_zero_padded() -> None:
    q = PlaybackQueue()
    q.enqueue(np.full(FRAME_SAMPLES + 10, 500, dtype=np.int16))
    q.flush_pending()
    q.next_frame()
    tail = q.next_frame()
    assert tail[0] == 500
    assert np.all(tail[10:] == 0)


def test_stop_fades_then_silences() -> None:
    q = PlaybackQueue(fade_ms=20)  # 2 frames of fade
    q.enqueue(_pcm(10, value=10_000))
    q.stop()
    f1 = q.next_frame()
    f2 = q.next_frame()
    # Fade must be monotonically decreasing toward zero.
    assert abs(int(f1[0])) <= 10_000
    assert abs(int(f2[-1])) < abs(int(f1[0]))
    assert not q.is_active  # flushed after fade
    assert np.all(q.next_frame() == 0)


def test_stop_on_empty_queue_is_noop() -> None:
    q = PlaybackQueue()
    q.stop()
    assert not q.is_active


def test_enqueue_after_stop_supersedes_old_audio() -> None:
    q = PlaybackQueue(fade_ms=20)
    q.enqueue(_pcm(10, value=1))
    q.stop()
    q.enqueue(_pcm(2, value=7))  # new turn's audio arrives mid-fade
    frames = [q.next_frame() for _ in range(2)]
    assert all(f[0] == 7 for f in frames)
    assert not q.is_active


def test_rejects_non_mono() -> None:
    q = PlaybackQueue()
    try:
        q.enqueue(np.zeros((10, 2), dtype=np.int16))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
