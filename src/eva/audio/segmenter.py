"""Utterance segmentation and barge-in detection — pure logic, no I/O.

Consumes (chunk, speech probability, playback-active) triples and emits
endpointing events. Deliberately free of audio/VAD imports beyond the frame
type so the entire state machine is unit-testable with synthetic sequences.

Behavioral contract:
- A pre-roll buffer keeps ~300 ms of audio from *before* speech onset, so soft
  first syllables (and the start of a barge-in phrase) are never lost.
- `BargeIn` fires once per utterance after `barge_in_confirm_ms` of cumulative
  speech while playback is active — long enough to ignore coughs and residual
  echo blips, short enough to feel instant.
- An utterance ends after `silence_timeout_ms` of continuous non-speech, or
  forcibly at `max_utterance_s`. Utterances with less than `min_speech_ms` of
  speech are discarded as noise (thesis-tuned gate).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from eva.audio.frames import SAMPLE_RATE, Frame
from eva.config.settings import VADSettings

_PRE_ROLL_MS = 300


@dataclass(frozen=True)
class SpeechStart:
    """First speech chunk of a new utterance."""


@dataclass(frozen=True)
class BargeIn:
    """User speech confirmed while the assistant was speaking."""

    speech_ms: int


@dataclass(frozen=True)
class UtteranceEnd:
    """Complete utterance ready for ASR (includes pre-roll and trailing silence)."""

    audio: Frame
    duration_ms: int
    speech_ms: int
    forced: bool  # True when ended by the max-utterance safety timeout


@dataclass(frozen=True)
class UtteranceDiscarded:
    """Utterance dropped by the noise gate."""

    speech_ms: int


@dataclass(frozen=True)
class UtteranceProgress:
    """Periodic snapshot of an in-progress utterance (drives partial ASR)."""

    audio: Frame
    duration_ms: int


SegmenterEvent = SpeechStart | BargeIn | UtteranceEnd | UtteranceDiscarded | UtteranceProgress


class SpeechSegmenter:
    def __init__(self, settings: VADSettings, *, partial_interval_ms: int | None = None) -> None:
        self._partial_interval_ms = partial_interval_ms
        self._threshold = settings.threshold
        self._silence_timeout_ms = settings.silence_timeout_ms
        self._min_speech_ms = settings.min_speech_ms
        self._max_utterance_ms = settings.max_utterance_s * 1000
        self._barge_in_enabled = settings.barge_in_enabled
        self._barge_in_confirm_ms = settings.barge_in_confirm_ms
        self._pre_roll: deque[Frame] = deque()
        self._pre_roll_ms = 0
        self.reset()

    def reset(self) -> None:
        self._active = False
        self._buffer: list[Frame] = []
        self._speech_ms = 0
        self._silence_ms = 0
        self._total_ms = 0
        self._barge_in_fired = False
        self._since_progress_ms = 0
        self._pre_roll.clear()
        self._pre_roll_ms = 0

    @property
    def utterance_active(self) -> bool:
        return self._active

    def feed(self, chunk: Frame, speech_prob: float, playback_active: bool) -> list[SegmenterEvent]:
        """Advance the state machine by one chunk; returns emitted events."""
        chunk_ms = chunk.shape[0] * 1000 // SAMPLE_RATE
        is_speech = speech_prob >= self._threshold
        events: list[SegmenterEvent] = []

        if not self._active:
            if is_speech:
                self._active = True
                self._buffer = [*self._pre_roll, chunk]
                self._pre_roll.clear()
                self._pre_roll_ms = 0
                self._speech_ms = chunk_ms
                self._silence_ms = 0
                self._total_ms = chunk_ms
                self._barge_in_fired = False
                self._since_progress_ms = 0
                events.append(SpeechStart())
                events.extend(self._maybe_barge_in(playback_active))
            else:
                self._push_pre_roll(chunk, chunk_ms)
            return events

        # Active utterance
        self._buffer.append(chunk)
        self._total_ms += chunk_ms
        if is_speech:
            self._speech_ms += chunk_ms
            self._silence_ms = 0
            events.extend(self._maybe_barge_in(playback_active))
        else:
            self._silence_ms += chunk_ms

        if self._silence_ms >= self._silence_timeout_ms:
            events.append(self._finish(forced=False))
        elif self._total_ms >= self._max_utterance_ms:
            events.append(self._finish(forced=True))
        elif self._partial_interval_ms is not None:
            self._since_progress_ms += chunk_ms
            if self._since_progress_ms >= self._partial_interval_ms:
                self._since_progress_ms = 0
                events.append(
                    UtteranceProgress(
                        audio=np.concatenate(self._buffer), duration_ms=self._total_ms
                    )
                )
        return events

    def _maybe_barge_in(self, playback_active: bool) -> list[SegmenterEvent]:
        if (
            self._barge_in_enabled
            and playback_active
            and not self._barge_in_fired
            and self._speech_ms >= self._barge_in_confirm_ms
        ):
            self._barge_in_fired = True
            return [BargeIn(speech_ms=self._speech_ms)]
        return []

    def _finish(self, *, forced: bool) -> SegmenterEvent:
        speech_ms = self._speech_ms
        total_ms = self._total_ms
        buffer = self._buffer
        self._active = False
        self._buffer = []
        self._speech_ms = 0
        self._silence_ms = 0
        self._total_ms = 0
        if speech_ms < self._min_speech_ms:
            return UtteranceDiscarded(speech_ms=speech_ms)
        audio: Frame = np.concatenate(buffer)
        return UtteranceEnd(audio=audio, duration_ms=total_ms, speech_ms=speech_ms, forced=forced)

    def _push_pre_roll(self, chunk: Frame, chunk_ms: int) -> None:
        self._pre_roll.append(chunk)
        self._pre_roll_ms += chunk_ms
        while self._pre_roll_ms > _PRE_ROLL_MS and len(self._pre_roll) > 1:
            dropped = self._pre_roll.popleft()
            self._pre_roll_ms -= dropped.shape[0] * 1000 // SAMPLE_RATE
