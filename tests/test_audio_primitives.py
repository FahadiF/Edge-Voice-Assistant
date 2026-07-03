from __future__ import annotations

import numpy as np

from eva.audio.chunker import FrameChunker
from eva.audio.frames import (
    FRAME_SAMPLES,
    float_to_int16,
    int16_to_float,
    rms_dbfs,
    silence_frame,
)
from eva.audio.ring import FrameRing


def _frame(value: int = 0) -> np.ndarray:
    return np.full(FRAME_SAMPLES, value, dtype=np.int16)


class TestFrames:
    def test_silence_frame_shape(self) -> None:
        f = silence_frame()
        assert f.shape == (FRAME_SAMPLES,)
        assert f.dtype == np.int16

    def test_float_int16_round_trip(self) -> None:
        f = np.linspace(-1, 1, FRAME_SAMPLES, dtype=np.float32)
        back = int16_to_float(float_to_int16(f))
        assert np.allclose(back, f, atol=1e-3)

    def test_float_to_int16_clips(self) -> None:
        f = np.array([2.0, -2.0], dtype=np.float32)
        out = float_to_int16(f)
        assert out[0] == 32767
        assert out[1] == -32767

    def test_rms_dbfs_silence(self) -> None:
        assert rms_dbfs(silence_frame()) == -120.0

    def test_rms_dbfs_full_scale(self) -> None:
        full = np.full(FRAME_SAMPLES, 32767, dtype=np.int16)
        assert abs(rms_dbfs(full)) < 0.1  # ≈ 0 dBFS


class TestFrameRing:
    def test_fifo_order(self) -> None:
        ring = FrameRing(4)
        ring.push(_frame(1))
        ring.push(_frame(2))
        first = ring.pop()
        assert first is not None and first[0] == 1
        second = ring.pop()
        assert second is not None and second[0] == 2
        assert ring.pop() is None

    def test_overflow_drops_oldest(self) -> None:
        ring = FrameRing(2)
        for v in (1, 2, 3):
            ring.push(_frame(v))
        assert ring.dropped == 1
        first = ring.pop()
        assert first is not None and first[0] == 2  # frame 1 was dropped

    def test_clear(self) -> None:
        ring = FrameRing(4)
        ring.push(_frame())
        ring.clear()
        assert len(ring) == 0
        assert ring.pop() is None


class TestFrameChunker:
    def test_aggregates_to_chunk_size(self) -> None:
        chunker = FrameChunker(512)
        chunks: list[np.ndarray] = []
        for _ in range(10):  # 10 x 160 = 1600 samples → 3 chunks of 512, 64 left
            chunks.extend(chunker.push(_frame()))
        assert len(chunks) == 3
        assert all(c.shape == (512,) for c in chunks)

    def test_preserves_sample_order(self) -> None:
        chunker = FrameChunker(4)
        out = chunker.push(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int16))
        assert [list(c) for c in out] == [[1, 2, 3, 4], [5, 6, 7, 8]]
        out2 = chunker.push(np.array([10, 11, 12], dtype=np.int16))
        assert [list(c) for c in out2] == [[9, 10, 11, 12]]

    def test_reset_discards_partial(self) -> None:
        chunker = FrameChunker(512)
        chunker.push(_frame())
        chunker.reset()
        assert chunker.push(np.zeros(512, dtype=np.int16))[0].shape == (512,)
