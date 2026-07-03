"""Audio subsystem: full-duplex stream, echo cancellation, capture pipeline.

Import concrete classes from their modules (`eva.audio.duplex`, …) rather than
re-exporting them here — importing this package must stay side-effect free and
must not require PortAudio to be present (server/CI environments).
"""

from eva.audio.frames import FRAME_MS, FRAME_SAMPLES, SAMPLE_RATE

__all__ = ["FRAME_MS", "FRAME_SAMPLES", "SAMPLE_RATE"]
