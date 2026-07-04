"""SpeechSegmenter state-machine tests with synthetic probability sequences.

Chunks are 512 samples = 32 ms. Settings used here: threshold 0.5,
silence timeout 320 ms (10 chunks), min speech 96 ms (3 chunks),
max utterance 5 s, barge-in confirm 96 ms (3 chunks).
"""

from __future__ import annotations

import numpy as np

from eva.audio.segmenter import (
    BargeIn,
    SegmenterEvent,
    SpeechSegmenter,
    SpeechStart,
    UtteranceDiscarded,
    UtteranceEnd,
)
from eva.config.settings import VADSettings

CHUNK = 512  # samples → 32 ms
CHUNK_MS = 32


def make_segmenter(**overrides: object) -> SpeechSegmenter:
    defaults: dict[str, object] = {
        "threshold": 0.5,
        "silence_timeout_ms": 320,
        "min_speech_ms": 96,
        "max_utterance_s": 5,
        "barge_in_enabled": True,
        "barge_in_confirm_ms": 96,
    }
    defaults.update(overrides)
    return SpeechSegmenter(VADSettings(**defaults))  # type: ignore[arg-type]


def feed(
    seg: SpeechSegmenter, probs: list[float], playing: bool = False, value: int = 100
) -> list[SegmenterEvent]:
    events: list[SegmenterEvent] = []
    for p in probs:
        chunk = np.full(CHUNK, value, dtype=np.int16)
        events.extend(seg.feed(chunk, p, playing))
    return events


def test_silence_produces_no_events() -> None:
    seg = make_segmenter()
    assert feed(seg, [0.0] * 50) == []
    assert not seg.utterance_active


def test_basic_utterance() -> None:
    seg = make_segmenter()
    events = feed(seg, [0.9] * 10 + [0.0] * 10)
    assert isinstance(events[0], SpeechStart)
    ends = [e for e in events if isinstance(e, UtteranceEnd)]
    assert len(ends) == 1
    end = ends[0]
    assert end.speech_ms == 10 * CHUNK_MS
    assert not end.forced
    # 10 speech + 10 silence chunks, no pre-roll (stream started with speech)
    assert end.duration_ms == 20 * CHUNK_MS
    assert end.audio.shape[0] == 20 * CHUNK
    assert not seg.utterance_active


def test_short_burst_discarded_as_noise() -> None:
    seg = make_segmenter()
    events = feed(seg, [0.9] * 2 + [0.0] * 10)  # 64 ms speech < 96 ms gate
    assert isinstance(events[-1], UtteranceDiscarded)
    assert not any(isinstance(e, UtteranceEnd) for e in events)


def test_pre_roll_included_in_utterance() -> None:
    seg = make_segmenter()
    feed(seg, [0.0] * 20)  # fills pre-roll (capped at ~300 ms ≈ 9 chunks)
    events = feed(seg, [0.9] * 10 + [0.0] * 10)
    end = next(e for e in events if isinstance(e, UtteranceEnd))
    # Audio must contain more than the speech+trailing-silence chunks.
    assert end.audio.shape[0] > 20 * CHUNK
    # Pre-roll is capped: no more than ~10 extra chunks.
    assert end.audio.shape[0] <= 30 * CHUNK


def test_mid_utterance_pause_tolerated() -> None:
    seg = make_segmenter()
    # 5 speech, 5 silence (160 ms < 320 ms timeout), 5 speech, then end.
    events = feed(seg, [0.9] * 5 + [0.0] * 5 + [0.9] * 5 + [0.0] * 10)
    ends = [e for e in events if isinstance(e, UtteranceEnd)]
    assert len(ends) == 1
    assert ends[0].speech_ms == 10 * CHUNK_MS


def test_max_utterance_forces_end() -> None:
    seg = make_segmenter()
    events = feed(seg, [0.9] * 200)  # 6.4 s of continuous speech > 5 s cap
    ends = [e for e in events if isinstance(e, UtteranceEnd)]
    assert len(ends) == 1
    assert ends[0].forced


def test_barge_in_fires_once_after_confirm_window() -> None:
    seg = make_segmenter()
    events = feed(seg, [0.9] * 10, playing=True)
    barges = [e for e in events if isinstance(e, BargeIn)]
    assert len(barges) == 1
    assert barges[0].speech_ms == 96  # 3rd chunk crosses the 96 ms confirm window


def test_no_barge_in_when_not_playing() -> None:
    seg = make_segmenter()
    events = feed(seg, [0.9] * 10, playing=False)
    assert not any(isinstance(e, BargeIn) for e in events)


def test_no_barge_in_when_disabled() -> None:
    seg = make_segmenter(barge_in_enabled=False)
    events = feed(seg, [0.9] * 10, playing=True)
    assert not any(isinstance(e, BargeIn) for e in events)


def test_echo_blip_does_not_barge_in() -> None:
    seg = make_segmenter()
    # Two speech chunks (64 ms < 96 ms confirm) then silence — no barge-in.
    events = feed(seg, [0.9] * 2 + [0.0] * 10, playing=True)
    assert not any(isinstance(e, BargeIn) for e in events)


def test_repeated_utterances_and_barge_ins() -> None:
    seg = make_segmenter()
    for _ in range(3):
        events = feed(seg, [0.9] * 10 + [0.0] * 10, playing=True)
        assert sum(isinstance(e, BargeIn) for e in events) == 1
        assert sum(isinstance(e, UtteranceEnd) for e in events) == 1


def test_reset_clears_active_utterance() -> None:
    seg = make_segmenter()
    feed(seg, [0.9] * 5)
    assert seg.utterance_active
    seg.reset()
    assert not seg.utterance_active
    assert feed(seg, [0.0] * 5) == []


def test_utterance_progress_emitted_at_interval() -> None:
    seg = SpeechSegmenter(
        VADSettings(silence_timeout_ms=320, min_speech_ms=96),
        partial_interval_ms=128,  # 4 chunks
    )
    events = feed(seg, [0.9] * 12 + [0.0] * 12)
    from eva.audio.segmenter import UtteranceProgress

    progress = [e for e in events if isinstance(e, UtteranceProgress)]
    assert len(progress) >= 2
    # Snapshots grow monotonically and include everything so far.
    assert progress[0].audio.shape[0] < progress[-1].audio.shape[0]
    # No progress events without the option.
    seg_off = make_segmenter()
    events_off = feed(seg_off, [0.9] * 12 + [0.0] * 12)
    assert not any(isinstance(e, UtteranceProgress) for e in events_off)


def test_barge_in_speech_is_kept_for_transcription() -> None:
    """The audio that triggered a barge-in must appear in the utterance."""
    seg = make_segmenter()
    events = feed(seg, [0.9] * 10 + [0.0] * 10, playing=True, value=1234)
    end = next(e for e in events if isinstance(e, UtteranceEnd))
    assert np.any(end.audio == 1234)
    assert end.speech_ms == 10 * CHUNK_MS
