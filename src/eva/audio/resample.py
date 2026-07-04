"""Sample-rate conversion for engine boundaries (e.g. Kokoro 24 kHz → 16 kHz).

Linear interpolation: adequate for downsampling speech to the pipeline rate
(playback is speech, not music) and dependency-free. A windowed-sinc polyphase
upgrade is scheduled with the M7 performance pass if listening tests warrant it.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def resample_int16(
    audio: npt.NDArray[np.int16], source_rate: int, target_rate: int
) -> npt.NDArray[np.int16]:
    if source_rate == target_rate or audio.size == 0:
        return audio
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be positive")
    duration = audio.shape[0] / source_rate
    target_length = max(1, round(duration * target_rate))
    source_positions = np.arange(audio.shape[0], dtype=np.float64)
    target_positions = np.linspace(0, audio.shape[0] - 1, target_length, dtype=np.float64)
    resampled = np.interp(target_positions, source_positions, audio.astype(np.float64))
    return resampled.astype(np.int16)
