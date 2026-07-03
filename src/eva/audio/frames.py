"""Canonical audio format of the pipeline.

Everything between the sound card and the ASR engine is **16 kHz mono int16**
in **10 ms frames (160 samples)** — the frame size WebRTC APM requires and the
greatest common divisor of what the downstream consumers need (Silero VAD reads
512-sample chunks assembled from these frames). Adapters that need other rates
resample at their own boundary.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

SAMPLE_RATE = 16_000
FRAME_MS = 10
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 160
FRAME_BYTES = FRAME_SAMPLES * 2  # int16

Frame = npt.NDArray[np.int16]
"""One 10 ms mono frame: shape (160,), dtype int16."""


def silence_frame() -> Frame:
    return np.zeros(FRAME_SAMPLES, dtype=np.int16)


def float_to_int16(samples: npt.NDArray[np.float32]) -> Frame:
    """Convert [-1, 1] float audio to int16 with clipping."""
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)


def int16_to_float(samples: npt.NDArray[np.int16]) -> npt.NDArray[np.float32]:
    return (samples.astype(np.float32) / 32768.0).astype(np.float32)


def rms_dbfs(samples: npt.NDArray[np.int16]) -> float:
    """RMS level in dBFS; -120.0 for digital silence."""
    if samples.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(int16_to_float(samples) ** 2)))
    if rms <= 1e-6:
        return -120.0
    return float(20.0 * np.log10(rms))
