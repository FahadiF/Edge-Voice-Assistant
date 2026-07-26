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


def test_streamed_chunks_join_without_gaps() -> None:
    """ADR-018: streamed TTS chunks are enqueued without flushing between
    them (`AudioSystem.say()` per chunk), then flushed once at the end
    (`finish_utterance()`). Neither chunk length here is a multiple of
    FRAME_SAMPLES, so if a flush happened between chunks (the old per-call
    behavior), the first chunk's partial tail frame would be zero-padded
    before the second chunk started — an audible gap. This must not happen."""
    q = PlaybackQueue()
    chunk1 = np.arange(1, 251, dtype=np.int16)  # 250 samples: not frame-aligned
    chunk2 = np.arange(1000, 1100, dtype=np.int16)  # 100 samples

    q.enqueue(chunk1)  # no flush between chunks
    q.enqueue(chunk2)
    q.flush_pending()  # one flush, after the last chunk of the utterance

    played = []
    while q.is_active:
        played.append(q.next_frame())
    played_flat = np.concatenate(played)

    expected = np.concatenate([chunk1, chunk2])
    total = expected.size
    assert np.array_equal(played_flat[:total], expected)  # contiguous, no gap
    assert np.all(played_flat[total:] == 0)  # only the final tail is padded


def test_flush_between_chunks_would_insert_a_gap() -> None:
    """Contrast case: flushing after every chunk (the pre-ADR-018 `say()`
    behavior) zero-pads each partial tail — proving the fix in the test above
    actually matters."""
    q = PlaybackQueue()
    chunk1 = np.arange(1, 251, dtype=np.int16)
    chunk2 = np.arange(1000, 1100, dtype=np.int16)

    q.enqueue(chunk1)
    q.flush_pending()  # old behavior: flush after every chunk
    q.enqueue(chunk2)
    q.flush_pending()

    played = np.concatenate([q.next_frame() for _ in range(4)])
    # chunk1 occupies frames 0-1 fully; frame 1's tail (samples 90:160) is
    # zero-padding inserted *before* chunk2 starts — the gap this ADR avoids.
    gap = played[FRAME_SAMPLES + 90 : FRAME_SAMPLES * 2]
    assert np.all(gap == 0)
    assert played[FRAME_SAMPLES * 2] == 1000  # chunk2 only starts on the next frame


def test_marker_fires_on_the_audio_clock_not_on_enqueue() -> None:
    """M7.1/ADR-028: the point of markers is that queued ≠ heard. Sentence 2's
    audio is enqueued while sentence 1 is still playing, so its marker must
    stay silent until playback actually reaches it — otherwise a display
    following markers would run ahead by the whole buffered lead."""
    fired: list[object] = []
    q = PlaybackQueue(on_marker=fired.append)

    q.enqueue(_pcm(3), marker="one")
    q.enqueue(_pcm(2), marker="two")  # queued early, must not fire early
    assert fired == []

    assert q.queued_seconds() == 0.05  # a full 50 ms of lead is buffered
    q.next_frame()  # frame 0: sentence one becomes audible
    assert fired == ["one"]
    q.next_frame()
    q.next_frame()
    assert fired == ["one"]  # still inside sentence one
    q.next_frame()  # frame 3: sentence two becomes audible
    assert fired == ["one", "two"]


def test_marker_survives_a_partial_tail_boundary() -> None:
    """Sentences are flushed with `finish_utterance()`, so the next sentence's
    marker lands on a frame boundary — but a marker registered while a partial
    tail is still pending must fire too, on the frame that carries it."""
    fired: list[object] = []
    q = PlaybackQueue(on_marker=fired.append)

    q.enqueue(np.full(FRAME_SAMPLES + 10, 500, dtype=np.int16), marker="one")
    q.next_frame()
    assert fired == ["one"]
    q.enqueue(_pcm(1), marker="two")  # merges with the 10-sample pending tail
    assert fired == ["one"]
    q.next_frame()
    assert fired == ["one", "two"]


def test_markers_are_dropped_for_audio_a_barge_in_cut_off() -> None:
    """A cancelled turn must never announce speech the user never heard."""
    fired: list[object] = []
    q = PlaybackQueue(fade_ms=20, on_marker=fired.append)

    q.enqueue(_pcm(2), marker="heard")
    q.enqueue(_pcm(10), marker="cut-off")
    q.next_frame()  # "heard" becomes audible
    q.stop()  # barge-in
    while q.is_active:
        q.next_frame()
    assert fired == ["heard"]


def test_marker_index_stays_aligned_after_a_flush() -> None:
    """Dropped frames are accounted for, so the *next* turn's marker fires on
    its own first frame — not immediately, and not never."""
    fired: list[object] = []
    q = PlaybackQueue(fade_ms=20, on_marker=fired.append)

    q.enqueue(_pcm(10), marker="old")
    q.stop()
    while q.is_active:
        q.next_frame()  # fade completes, the rest of the queue is discarded
    fired.clear()

    q.enqueue(_pcm(2), marker="new")
    assert fired == []  # not yet audible
    q.next_frame()
    assert fired == ["new"]


def test_stop_on_an_empty_queue_discards_a_stranded_marker() -> None:
    """A sentence whose audio never materialized leaves a marker with no frame
    to ride on; stopping the turn must clear it so it cannot later fire
    against the next turn's audio."""
    fired: list[object] = []
    q = PlaybackQueue(on_marker=fired.append)

    q.enqueue(np.zeros(0, dtype=np.int16), marker="stranded")
    q.stop()  # nothing queued: the early-return path
    q.enqueue(_pcm(1), marker="next-turn")
    q.next_frame()
    assert fired == ["next-turn"]


def test_marker_handler_failure_never_breaks_playback() -> None:
    def boom(_marker: object) -> None:
        raise RuntimeError("consumer is broken")

    q = PlaybackQueue(on_marker=boom)
    q.enqueue(_pcm(2), marker="x")
    assert q.next_frame()[0] == 1000  # audio keeps flowing
    assert q.next_frame()[0] == 1000


def test_unmarked_audio_needs_no_handler() -> None:
    q = PlaybackQueue()  # no on_marker configured
    q.enqueue(_pcm(1), marker="ignored")
    assert q.next_frame()[0] == 1000


def test_rejects_non_mono() -> None:
    q = PlaybackQueue()
    try:
        q.enqueue(np.zeros((10, 2), dtype=np.int16))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
