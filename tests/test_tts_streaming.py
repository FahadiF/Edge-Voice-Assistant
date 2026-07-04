"""TTS streaming synthesis tests (ADR-018).

Covers the ABC's default single-chunk fallback and the Kokoro adapter's real
streaming path (mocked `kokoro_onnx.Kokoro`, no model files required).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from eva.audio.frames import Frame
from eva.core.errors import ModelError
from eva.tts.base import TTSEngine
from eva.tts.kokoro import KokoroTTS


class _MinimalTTS(TTSEngine):
    """Only implements the abstract methods — exercises the default
    `synthesize_stream()` fallback defined on the ABC."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def load(self) -> None: ...

    def unload(self) -> None: ...

    def synthesize(self, text: str, *, voice: str, speed: float = 1.0) -> Frame:
        self.calls.append(text)
        return np.full(800, 7, dtype=np.int16)

    def voices(self) -> list[str]:
        return ["only-voice"]


def test_default_synthesize_stream_yields_one_chunk_via_synthesize() -> None:
    engine = _MinimalTTS()
    chunks = list(engine.synthesize_stream("hello there", voice="only-voice"))
    assert len(chunks) == 1
    assert np.array_equal(chunks[0], np.full(800, 7, dtype=np.int16))
    assert engine.calls == ["hello there"]


class _FakeKokoroStreaming:
    """Stands in for `kokoro_onnx.Kokoro`: `create_stream` yields several
    chunks at a fake 24 kHz so the resample-to-16kHz path is exercised too."""

    def __init__(self, batches: list[np.ndarray], fail: bool = False) -> None:
        self._batches = batches
        self._fail = fail
        self.closed_streams = 0

    async def create_stream(self, text: str, voice: str, speed: float = 1.0) -> Any:
        # Real kokoro_onnx.Kokoro.create_stream is `async def` + `yield` in the
        # same body, making it an async-generator function: calling it returns
        # the generator directly (no coroutine to await first). Match that shape.
        try:
            for batch in self._batches:
                if self._fail:
                    raise RuntimeError("synthesis exploded")
                yield (batch.astype(np.float32) / 32768.0, 24_000)
        finally:
            self.closed_streams += 1

    def get_voices(self) -> list[str]:
        return ["af_heart"]


def _make_kokoro(fake: _FakeKokoroStreaming) -> KokoroTTS:
    engine = KokoroTTS(Path("unused.onnx"), Path("unused.bin"))
    engine._kokoro = fake  # bypass load(); no real model files needed
    engine.device = "cpu"
    return engine


def test_kokoro_streams_multiple_chunks_and_resamples() -> None:
    batches = [np.full(2400, 1000, dtype=np.int16), np.full(2400, -1000, dtype=np.int16)]
    fake = _FakeKokoroStreaming(batches)
    engine = _make_kokoro(fake)

    chunks = list(engine.synthesize_stream("Hello. World.", voice="af_heart"))

    assert len(chunks) == 2
    # 24kHz -> 16kHz: input length * (16000/24000)
    for chunk in chunks:
        assert chunk.dtype == np.int16
        assert chunk.shape[0] == pytest.approx(2400 * 16_000 / 24_000, abs=1)
    assert fake.closed_streams == 1


def test_kokoro_stream_closes_underlying_generator_on_early_stop() -> None:
    batches = [np.full(2400, 1000, dtype=np.int16) for _ in range(5)]
    fake = _FakeKokoroStreaming(batches)
    engine = _make_kokoro(fake)

    gen = engine.synthesize_stream("A long reply with many clauses.", voice="af_heart")
    first = next(gen)
    assert first.dtype == np.int16
    gen.close()  # simulates barge-in: caller stops consuming mid-stream

    assert fake.closed_streams == 1


def test_kokoro_stream_empty_text_yields_nothing() -> None:
    engine = _make_kokoro(_FakeKokoroStreaming([]))
    assert list(engine.synthesize_stream("   ", voice="af_heart")) == []


def test_kokoro_stream_wraps_synthesis_errors() -> None:
    fake = _FakeKokoroStreaming([np.zeros(100, dtype=np.int16)], fail=True)
    engine = _make_kokoro(fake)
    with pytest.raises(ModelError, match="streaming synthesis failed"):
        list(engine.synthesize_stream("boom", voice="af_heart"))


def test_kokoro_stream_reusable_after_previous_call_closed() -> None:
    """Each call gets its own event loop — no state leaks between calls."""
    engine = _make_kokoro(_FakeKokoroStreaming([np.full(1600, 1, dtype=np.int16)]))
    first = list(engine.synthesize_stream("first", voice="af_heart"))
    second = list(engine.synthesize_stream("second", voice="af_heart"))
    assert len(first) == 1
    assert len(second) == 1
