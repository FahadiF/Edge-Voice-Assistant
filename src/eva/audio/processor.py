"""Capture-path signal processing: echo cancellation, noise suppression, AGC.

`WebRtcAudioProcessor` wraps the WebRTC Audio Processing Module (via the
livekit SDK, the maintained cross-platform binding). The duplex stream feeds
every played frame to `process_render()` (the far-end reference) and passes
every mic frame through `process_capture()`; APM subtracts the assistant's own
voice from the mic signal, which is what makes barge-in on speakers possible
(ADR-005).

`PassthroughProcessor` is the fallback when APM is unavailable or disabled —
the capture pipeline then relies on the half-duplex / push-to-talk ladder.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from eva.audio.frames import FRAME_SAMPLES, SAMPLE_RATE, Frame
from eva.config.settings import AudioSettings

logger = logging.getLogger(__name__)


class AudioProcessor(Protocol):
    """10 ms-frame signal processor with a far-end reference path."""

    name: str

    def process_capture(self, frame: Frame) -> Frame:
        """Clean one mic frame (in place or copy); returns the cleaned frame."""
        ...

    def process_render(self, frame: Frame) -> None:
        """Feed one frame that is about to be played (far-end reference)."""
        ...

    def set_stream_delay_ms(self, delay_ms: int) -> None:
        """Report the render→capture loop delay estimate to the canceller."""
        ...


class PassthroughProcessor:
    """No-op processor: frames pass unchanged."""

    name = "passthrough"

    def process_capture(self, frame: Frame) -> Frame:
        return frame

    def process_render(self, frame: Frame) -> None:
        return None

    def set_stream_delay_ms(self, delay_ms: int) -> None:
        return None


class WebRtcAudioProcessor:
    """WebRTC APM wrapper. Frames must be 10 ms / 16 kHz / mono int16."""

    name = "webrtc-apm"

    def __init__(
        self,
        *,
        echo_cancellation: bool = True,
        noise_suppression: bool = True,
        auto_gain_control: bool = True,
    ) -> None:
        import numpy as np
        from livekit import rtc
        from livekit.rtc.apm import AudioProcessingModule

        self._np = np
        self._rtc = rtc
        self._apm = AudioProcessingModule(
            echo_cancellation=echo_cancellation,
            noise_suppression=noise_suppression,
            high_pass_filter=True,
            auto_gain_control=auto_gain_control,
        )

    def _to_rtc_frame(self, frame: Frame) -> Any:
        if frame.shape[0] != FRAME_SAMPLES:
            raise ValueError(f"APM requires {FRAME_SAMPLES}-sample frames, got {frame.shape[0]}")
        return self._rtc.AudioFrame(
            data=frame.tobytes(),
            sample_rate=SAMPLE_RATE,
            num_channels=1,
            samples_per_channel=FRAME_SAMPLES,
        )

    def process_capture(self, frame: Frame) -> Frame:
        rtc_frame = self._to_rtc_frame(frame)
        self._apm.process_stream(rtc_frame)
        cleaned: Frame = self._np.frombuffer(rtc_frame.data, dtype=self._np.int16).copy()
        return cleaned

    def process_render(self, frame: Frame) -> None:
        self._apm.process_reverse_stream(self._to_rtc_frame(frame))

    def set_stream_delay_ms(self, delay_ms: int) -> None:
        self._apm.set_stream_delay_ms(max(0, delay_ms))


def create_processor(settings: AudioSettings) -> AudioProcessor:
    """Build the best available processor for the current settings.

    Degrades to passthrough (with a warning) if APM cannot be constructed, so a
    broken native dependency downgrades the experience instead of killing the app.
    """
    wants_apm = (
        settings.echo_cancellation or settings.noise_suppression or settings.auto_gain_control
    )
    if not wants_apm:
        return PassthroughProcessor()
    try:
        return WebRtcAudioProcessor(
            echo_cancellation=settings.echo_cancellation,
            noise_suppression=settings.noise_suppression,
            auto_gain_control=settings.auto_gain_control,
        )
    except Exception:
        logger.warning(
            "WebRTC APM unavailable — falling back to passthrough audio. "
            "Echo cancellation disabled; consider half-duplex or push-to-talk mode.",
            exc_info=True,
        )
        return PassthroughProcessor()
