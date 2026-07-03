"""Model-backed tests that need no audio hardware and no network.

The Silero model ships inside pysilero-vad; WebRTC APM ships inside livekit.
Both run on CPU in milliseconds, so these stay in the default test run.
"""

from __future__ import annotations

import numpy as np
import pytest

from eva.audio.frames import FRAME_SAMPLES, SAMPLE_RATE, float_to_int16
from eva.audio.processor import PassthroughProcessor, WebRtcAudioProcessor, create_processor
from eva.config.settings import AudioSettings
from eva.vad.registry import create_vad, register_builtins, vad_registry


def _speechlike(seconds: float, seed: int = 7) -> np.ndarray:
    """Amplitude-modulated band-limited noise — enough to excite AEC adaptation."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    carrier = rng.standard_normal(n).astype(np.float32)
    envelope = (0.5 + 0.5 * np.sin(2 * np.pi * 3.1 * t)).astype(np.float32)
    return float_to_int16(0.3 * carrier * envelope)


class TestSileroVAD:
    def test_registry_provides_silero(self) -> None:
        register_builtins()
        assert "silero" in vad_registry

    def test_silence_scores_low_speech_scores_are_valid(self) -> None:
        vad = create_vad("silero")
        silence = np.zeros(vad.chunk_samples, dtype=np.int16)
        prob = vad.process(silence)
        assert 0.0 <= prob < 0.3
        noisy = _speechlike(0.5)[: vad.chunk_samples]
        assert 0.0 <= vad.process(noisy) <= 1.0

    def test_wrong_chunk_size_rejected(self) -> None:
        vad = create_vad("silero")
        with pytest.raises(ValueError):
            vad.process(np.zeros(100, dtype=np.int16))

    def test_reset(self) -> None:
        vad = create_vad("silero")
        vad.process(np.zeros(vad.chunk_samples, dtype=np.int16))
        vad.reset()  # must not raise


class TestProcessorSelection:
    def test_all_disabled_gives_passthrough(self) -> None:
        s = AudioSettings(echo_cancellation=False, noise_suppression=False, auto_gain_control=False)
        assert isinstance(create_processor(s), PassthroughProcessor)

    def test_default_gives_webrtc(self) -> None:
        assert isinstance(create_processor(AudioSettings()), WebRtcAudioProcessor)

    def test_passthrough_is_identity(self) -> None:
        p = PassthroughProcessor()
        frame = np.arange(FRAME_SAMPLES, dtype=np.int16)
        assert np.array_equal(p.process_capture(frame), frame)


class TestWebRtcApmEchoCancellation:
    def test_apm_attenuates_pure_echo(self) -> None:
        """Feed a signal as render and a delayed copy as capture: after the
        canceller converges, cleaned output must be much quieter than input."""
        apm = WebRtcAudioProcessor(
            echo_cancellation=True, noise_suppression=False, auto_gain_control=False
        )
        apm.set_stream_delay_ms(30)
        signal = _speechlike(6.0)
        delay = int(0.030 * SAMPLE_RATE)
        echo = np.concatenate([np.zeros(delay, dtype=np.int16), signal])[: signal.shape[0]]
        echo = (echo.astype(np.float32) * 0.7).astype(np.int16)

        n_frames = signal.shape[0] // FRAME_SAMPLES
        in_energy = 0.0
        out_energy = 0.0
        measure_from = n_frames // 2  # skip the convergence period
        for i in range(n_frames):
            sl = slice(i * FRAME_SAMPLES, (i + 1) * FRAME_SAMPLES)
            apm.process_render(signal[sl])
            cleaned = apm.process_capture(echo[sl].copy())
            if i >= measure_from:
                in_energy += float(np.mean(echo[sl].astype(np.float64) ** 2))
                out_energy += float(np.mean(cleaned.astype(np.float64) ** 2))

        assert in_energy > 0
        erle_db = 10 * np.log10(in_energy / max(out_energy, 1e-9))
        assert erle_db > 10.0, f"expected >10 dB echo attenuation, got {erle_db:.1f} dB"

    def test_wrong_frame_size_rejected(self) -> None:
        apm = WebRtcAudioProcessor()
        with pytest.raises(ValueError):
            apm.process_capture(np.zeros(100, dtype=np.int16))
