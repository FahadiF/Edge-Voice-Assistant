from __future__ import annotations

import time

import numpy as np

from eva.audio.capture import CapturePipeline
from eva.audio.frames import FRAME_SAMPLES
from eva.audio.ring import FrameRing
from eva.audio.segmenter import SegmenterEvent, SpeechSegmenter, SpeechStart, UtteranceEnd
from eva.config.settings import VADSettings
from eva.vad.base import VADEngine


class ScriptedVAD(VADEngine):
    """Returns a scripted probability per chunk; loud chunks count as speech."""

    def __init__(self) -> None:
        self.reset_count = 0

    @property
    def chunk_samples(self) -> int:
        return 512

    def process(self, chunk: np.ndarray) -> float:
        return 0.9 if int(np.abs(chunk).max()) > 500 else 0.05

    def reset(self) -> None:
        self.reset_count += 1


def _run_pipeline_with(frames: list[np.ndarray], timeout_s: float = 3.0) -> list[SegmenterEvent]:
    ring = FrameRing(4096)
    events: list[SegmenterEvent] = []
    vad = ScriptedVAD()
    segmenter = SpeechSegmenter(
        VADSettings(silence_timeout_ms=160, min_speech_ms=64, barge_in_confirm_ms=64)
    )
    pipeline = CapturePipeline(
        ring, vad, segmenter, playback_active=lambda: False, on_event=events.append
    )
    for f in frames:
        ring.push(f)
    pipeline.start()
    deadline = time.time() + timeout_s
    while time.time() < deadline and not any(isinstance(e, UtteranceEnd) for e in events):
        time.sleep(0.01)
    pipeline.stop()
    return events


def test_pipeline_detects_utterance_from_ring() -> None:
    speech = [np.full(FRAME_SAMPLES, 2000, dtype=np.int16)] * 40  # 400 ms loud
    silence = [np.zeros(FRAME_SAMPLES, dtype=np.int16)] * 40  # 400 ms quiet
    events = _run_pipeline_with(speech + silence)
    assert any(isinstance(e, SpeechStart) for e in events)
    ends = [e for e in events if isinstance(e, UtteranceEnd)]
    assert len(ends) == 1
    assert ends[0].audio.size > 0


def test_pipeline_silence_only_emits_nothing() -> None:
    silence = [np.zeros(FRAME_SAMPLES, dtype=np.int16)] * 30
    events = _run_pipeline_with(silence, timeout_s=0.5)
    assert events == []


def test_pipeline_vad_reset_on_start() -> None:
    ring = FrameRing(16)
    vad = ScriptedVAD()
    segmenter = SpeechSegmenter(VADSettings())
    pipeline = CapturePipeline(
        ring, vad, segmenter, playback_active=lambda: False, on_event=lambda e: None
    )
    pipeline.start()
    pipeline.stop()
    assert vad.reset_count == 1
