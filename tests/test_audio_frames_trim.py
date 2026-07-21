"""trim_edge_silence tests.

Added alongside the sentence-boundary silence trim (orchestrator speak_worker):
Kokoro's per-utterance output was measured (real synthesis, real audio) to
carry ~40-100ms of genuine dead air at each edge. These tests lock the
bounded-trim contract: only real silence is cut, up to the cap, and real
speech is never touched.
"""

from __future__ import annotations

import numpy as np

from eva.audio.frames import SAMPLE_RATE, trim_edge_silence

SPEECH = 8000  # well above the silence threshold


def _clip(*, lead_silence_ms: int, speech_ms: int, trail_silence_ms: int) -> np.ndarray:
    lead = np.zeros(int(lead_silence_ms * SAMPLE_RATE / 1000), dtype=np.int16)
    speech = np.full(int(speech_ms * SAMPLE_RATE / 1000), SPEECH, dtype=np.int16)
    trail = np.zeros(int(trail_silence_ms * SAMPLE_RATE / 1000), dtype=np.int16)
    return np.concatenate([lead, speech, trail])


class TestNoOp:
    def test_zero_caps_return_input_unchanged(self) -> None:
        clip = _clip(lead_silence_ms=50, speech_ms=100, trail_silence_ms=50)
        trimmed = trim_edge_silence(clip, max_leading_ms=0, max_trailing_ms=0)
        assert trimmed is clip  # identity, not just equal — no copy on the no-op path

    def test_empty_input(self) -> None:
        empty = np.zeros(0, dtype=np.int16)
        assert trim_edge_silence(empty, max_leading_ms=100, max_trailing_ms=100).size == 0

    def test_all_silence_returned_unchanged(self) -> None:
        silence = np.zeros(1600, dtype=np.int16)
        trimmed = trim_edge_silence(silence, max_leading_ms=100, max_trailing_ms=100)
        assert trimmed.size == silence.size  # nothing meaningful to trim around


class TestTrailingTrim:
    def test_trims_trailing_silence_up_to_actual_amount(self) -> None:
        clip = _clip(lead_silence_ms=0, speech_ms=100, trail_silence_ms=60)
        trimmed = trim_edge_silence(clip, max_trailing_ms=150)
        expected_speech_samples = int(100 * SAMPLE_RATE / 1000)
        assert trimmed.size == expected_speech_samples  # all 60ms of real trailing silence cut

    def test_never_cuts_more_than_the_cap(self) -> None:
        clip = _clip(lead_silence_ms=0, speech_ms=100, trail_silence_ms=200)
        trimmed = trim_edge_silence(clip, max_trailing_ms=50)
        expected_cap_samples = int(50 * SAMPLE_RATE / 1000)
        assert trimmed.size == clip.size - expected_cap_samples
        # Only the capped 50ms is cut; the remaining 150ms of real trailing
        # silence beyond the cap is deliberately left in place.


class TestLeadingTrim:
    def test_trims_leading_silence_up_to_actual_amount(self) -> None:
        clip = _clip(lead_silence_ms=40, speech_ms=100, trail_silence_ms=0)
        trimmed = trim_edge_silence(clip, max_leading_ms=80)
        expected_speech_samples = int(100 * SAMPLE_RATE / 1000)
        assert trimmed.size == expected_speech_samples

    def test_never_cuts_more_than_the_cap(self) -> None:
        clip = _clip(lead_silence_ms=150, speech_ms=100, trail_silence_ms=0)
        trimmed = trim_edge_silence(clip, max_leading_ms=80)
        expected_cap_samples = int(80 * SAMPLE_RATE / 1000)
        assert trimmed.size == clip.size - expected_cap_samples
        # Only the capped 80ms is cut; the remaining 70ms of real leading
        # silence beyond the cap is deliberately left in place.


class TestNeverClipsRealSpeech:
    def test_a_quiet_leading_consonant_is_not_mistaken_for_silence_beyond_the_cap(self) -> None:
        # Real speech starts immediately (0ms lead) — even a large cap must not
        # eat into it, since there's no silence there to trim.
        clip = _clip(lead_silence_ms=0, speech_ms=200, trail_silence_ms=0)
        trimmed = trim_edge_silence(clip, max_leading_ms=100, max_trailing_ms=100)
        assert trimmed.size == clip.size

    def test_trims_both_edges_independently(self) -> None:
        clip = _clip(lead_silence_ms=30, speech_ms=100, trail_silence_ms=90)
        trimmed = trim_edge_silence(clip, max_leading_ms=80, max_trailing_ms=150)
        expected_speech_samples = int(100 * SAMPLE_RATE / 1000)
        assert trimmed.size == expected_speech_samples
