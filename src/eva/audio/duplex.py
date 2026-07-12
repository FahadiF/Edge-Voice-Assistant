"""Full-duplex audio stream: one PortAudio stream, one clock (ADR-005).

Capture and playback share a single `sounddevice.Stream`, so the far-end
reference fed to the echo canceller is aligned with the mic signal by
construction — the failure mode that breaks WebRTC AEC in two-stream designs.

Callback discipline: the callback only moves frames — pull one playback frame,
feed it to the processor's render path, clean the mic frame, push both into
their rings. APM's native processing is well under 1 ms per 10 ms frame. No
allocation-heavy work, no logging, no locks beyond the queues' micro-mutexes.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import numpy.typing as npt

from eva.audio.frames import FRAME_SAMPLES, SAMPLE_RATE, Frame
from eva.audio.playback import PlaybackQueue
from eva.audio.processor import AudioProcessor
from eva.audio.ring import FrameRing
from eva.core.errors import AudioError

logger = logging.getLogger(__name__)


class DuplexAudioStream:
    """Owns the sound devices; produces cleaned capture frames, consumes playback."""

    def __init__(
        self,
        processor: AudioProcessor,
        playback: PlaybackQueue,
        capture_ring: FrameRing,
        *,
        input_device: str | int | None = None,
        output_device: str | int | None = None,
        mic_gain: float = 1.0,
        speaker_volume: float = 1.0,
        raw_tap: FrameRing | None = None,
    ) -> None:
        self._processor = processor
        self._playback = playback
        self._capture_ring = capture_ring
        self._raw_tap = raw_tap  # pre-APM frames, for diagnostics (echo test)
        self._input_device = input_device
        self._output_device = output_device
        self._mic_gain = float(mic_gain)
        self._volume = float(speaker_volume)
        self._stream: Any = None
        self._callback_errors = 0

    def start(self, *, playback_only: bool = False) -> None:
        """Open the audio device(s).

        `playback_only` (M5.6) opens ONLY the output device — used when the
        microphone permission is off (ADR-025): speech synthesis still plays,
        but no input device is ever opened (a revoked mic permission must not
        even touch the microphone). Previously mic-off skipped audio startup
        entirely, which left the playback queue undrained and wedged every
        typed turn in the "speaking" state forever.
        """
        import sounddevice as sd

        if self._stream is not None:
            raise AudioError("Duplex stream already started")
        try:
            if playback_only:
                self._stream = sd.OutputStream(
                    samplerate=SAMPLE_RATE,
                    blocksize=FRAME_SAMPLES,
                    channels=1,
                    dtype="int16",
                    device=self._output_device,
                    callback=self._output_callback,
                )
            else:
                self._stream = sd.Stream(
                    samplerate=SAMPLE_RATE,
                    blocksize=FRAME_SAMPLES,
                    channels=1,
                    dtype="int16",
                    device=(self._input_device, self._output_device),
                    callback=self._callback,
                )
            self._stream.start()
        except Exception as exc:
            self._stream = None
            raise AudioError(f"Cannot open duplex audio stream: {exc}") from exc

        if playback_only:
            logger.info("Playback-only stream started (microphone permission is off)")
            return
        # Report the render→capture loop latency to the echo canceller.
        try:
            in_lat, out_lat = self._stream.latency
            delay_ms = int((in_lat + out_lat) * 1000)
            self._processor.set_stream_delay_ms(delay_ms)
            logger.info("Duplex stream started (loop delay ≈ %d ms)", delay_ms)
        except Exception:
            logger.info("Duplex stream started (loop delay unknown)")

    def stop(self) -> None:
        if self._stream is None:
            return
        try:
            self._stream.stop()
            self._stream.close()
        finally:
            self._stream = None
        logger.info("Duplex stream stopped (callback errors: %d)", self._callback_errors)

    @property
    def running(self) -> bool:
        return self._stream is not None

    def _output_callback(
        self,
        outdata: npt.NDArray[np.int16],
        frames: int,
        _time: Any,
        status: Any,
    ) -> None:
        """Render-only callback (playback-only mode): no capture ring, no
        echo-canceller render feed — there is no mic signal to clean."""
        try:
            out_frame = self._playback.next_frame()
            if self._volume != 1.0:
                out_frame = (out_frame.astype(np.float32) * self._volume).astype(np.int16)
            outdata[:, 0] = out_frame
        except Exception:
            self._callback_errors += 1
            outdata.fill(0)

    def _callback(
        self,
        indata: npt.NDArray[np.int16],
        outdata: npt.NDArray[np.int16],
        frames: int,
        _time: Any,
        status: Any,
    ) -> None:
        try:
            # ── render path ──
            out_frame = self._playback.next_frame()
            if self._volume != 1.0:
                out_frame = (out_frame.astype(np.float32) * self._volume).astype(np.int16)
            outdata[:, 0] = out_frame
            self._processor.process_render(out_frame)

            # ── capture path ──
            mic: Frame = indata[:, 0].copy()
            if self._mic_gain != 1.0:
                mic = np.clip(mic.astype(np.float32) * self._mic_gain, -32768, 32767).astype(
                    np.int16
                )
            if self._raw_tap is not None:
                self._raw_tap.push(mic)
            self._capture_ring.push(self._processor.process_capture(mic))
        except Exception:
            self._callback_errors += 1
            outdata.fill(0)
