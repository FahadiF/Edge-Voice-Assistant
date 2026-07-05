from __future__ import annotations

import numpy as np
import pytest

from eva.audio.resample import resample_int16


class TestResample:
    def test_identity_when_rates_match(self) -> None:
        audio = np.arange(100, dtype=np.int16)
        assert resample_int16(audio, 16000, 16000) is audio

    def test_24k_to_16k_length(self) -> None:
        audio = np.zeros(24_000, dtype=np.int16)  # 1 second
        out = resample_int16(audio, 24_000, 16_000)
        assert abs(out.shape[0] - 16_000) <= 1

    def test_preserves_dc_level(self) -> None:
        audio = np.full(2400, 1000, dtype=np.int16)
        out = resample_int16(audio, 24_000, 16_000)
        assert np.all(out == 1000)

    def test_sine_frequency_preserved(self) -> None:
        t = np.arange(24_000) / 24_000
        sine = (np.sin(2 * np.pi * 440 * t) * 10_000).astype(np.int16)
        out = resample_int16(sine, 24_000, 16_000)
        # Zero crossings per second: about 2x the frequency
        crossings = int(np.sum(np.abs(np.diff(np.sign(out.astype(np.int32)))) > 0))
        assert abs(crossings - 880) < 40

    def test_empty_audio(self) -> None:
        empty = np.zeros(0, dtype=np.int16)
        assert resample_int16(empty, 24_000, 16_000).size == 0

    def test_invalid_rate_rejected(self) -> None:
        with pytest.raises(ValueError):
            resample_int16(np.zeros(10, dtype=np.int16), 0, 16_000)
