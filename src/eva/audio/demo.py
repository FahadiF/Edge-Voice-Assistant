"""Interactive audio diagnostics used by the CLI (`eva listen`, `eva echo-test`).

These are validation tools for the M1 exit criteria, runnable on any machine
with a microphone and speakers:

- `run_listen` shows live VAD/segmentation events and the input level.
- `run_echo_test` measures how well echo cancellation prevents self-triggering:
  it records a short speech sample from the user, plays it back over the
  speakers while capturing, and reports raw vs cleaned echo level and any VAD
  events caused by the playback (there should be none while the user is silent).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from eva.audio.chunker import FrameChunker
from eva.audio.duplex import DuplexAudioStream
from eva.audio.frames import Frame, rms_dbfs
from eva.audio.playback import PlaybackQueue
from eva.audio.processor import create_processor
from eva.audio.ring import FrameRing
from eva.audio.segmenter import (
    BargeIn,
    SegmenterEvent,
    SpeechSegmenter,
    SpeechStart,
    UtteranceDiscarded,
    UtteranceEnd,
)
from eva.audio.system import AudioSystem
from eva.config.settings import Settings
from eva.vad.registry import create_vad

_RING_CAPACITY_FRAMES = 3000  # 30 s of 10 ms frames


def _describe(event: SegmenterEvent) -> str:
    match event:
        case SpeechStart():
            return "speech started"
        case BargeIn(speech_ms=ms):
            return f"BARGE-IN confirmed after {ms} ms of speech"
        case UtteranceEnd(duration_ms=d, speech_ms=s, forced=f):
            suffix = " (forced by max-utterance timeout)" if f else ""
            return f"utterance ended: {d} ms total, {s} ms speech{suffix}"
        case UtteranceDiscarded(speech_ms=s):
            return f"utterance discarded as noise ({s} ms speech)"


def run_listen(settings: Settings, seconds: float) -> int:
    """Live VAD/segmenter monitor."""
    t0 = time.perf_counter()

    def on_event(event: SegmenterEvent) -> None:
        print(f"\r[{time.perf_counter() - t0:6.2f}s] {_describe(event)}" + " " * 20)

    system = AudioSystem(settings, on_event)
    system.start()
    print(f"Listening for {seconds:.0f} s — speak into the microphone. Ctrl+C to stop.")
    try:
        while time.perf_counter() - t0 < seconds:
            level = system.pipeline.level_dbfs
            bar = "#" * max(0, int((level + 60) / 3))
            print(f"\rlevel {level:7.1f} dBFS |{bar:<20}|", end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        system.stop()
        print()
    return 0


@dataclass
class EchoTestReport:
    raw_level_dbfs: float
    cleaned_level_dbfs: float
    attenuation_db: float
    vad_events: int
    processor: str

    @property
    def passed(self) -> bool:
        return self.vad_events == 0


def run_echo_test(settings: Settings, record_seconds: float = 4.0, loops: int = 2) -> int:
    """Measure echo suppression on the real speaker/microphone path."""
    processor = create_processor(settings.audio)
    playback = PlaybackQueue()
    clean_ring = FrameRing(_RING_CAPACITY_FRAMES)
    raw_ring = FrameRing(_RING_CAPACITY_FRAMES)
    stream = DuplexAudioStream(
        processor,
        playback,
        clean_ring,
        input_device=settings.audio.input_device,
        output_device=settings.audio.output_device,
        raw_tap=raw_ring,
    )
    vad = create_vad(settings.vad.engine)
    segmenter = SpeechSegmenter(settings.vad)
    chunker = FrameChunker(vad.chunk_samples)

    stream.start()
    try:
        # ── Phase 1: record a speech sample from the user ──
        print(f"Recording {record_seconds:.0f} s — please speak normally...")
        recorded: list[Frame] = []
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < record_seconds:
            frame = clean_ring.pop()
            if frame is None:
                time.sleep(0.002)
                continue
            recorded.append(frame)
        sample = np.concatenate(recorded) if recorded else np.zeros(0, dtype=np.int16)
        sample_level = rms_dbfs(sample)
        if sample_level < -50.0:
            print(f"Recorded level too low ({sample_level:.1f} dBFS) — check the microphone.")
            return 1
        print(f"Sample recorded ({sample_level:.1f} dBFS).")

        # ── Phase 2: play the sample back; measure echo on the mic path ──
        print(f"Playing the sample back {loops}x — please stay silent...")
        clean_ring.clear()
        raw_ring.clear()
        vad.reset()
        segmenter.reset()
        chunker.reset()

        for _ in range(loops):
            playback.enqueue(sample)
        playback.flush_pending()

        raw_energy: list[float] = []
        clean_energy: list[float] = []
        vad_events = 0
        while playback.is_active:
            raw = raw_ring.pop()
            if raw is not None:
                raw_energy.append(float(np.mean(raw.astype(np.float64) ** 2)))
            clean = clean_ring.pop()
            if clean is not None:
                clean_energy.append(float(np.mean(clean.astype(np.float64) ** 2)))
                for chunk in chunker.push(clean):
                    prob = vad.process(chunk)
                    for event in segmenter.feed(chunk, prob, True):
                        if isinstance(event, SpeechStart | BargeIn):
                            vad_events += 1
                            print(f"  self-trigger: {_describe(event)}")
            if raw is None and clean is None:
                time.sleep(0.002)
    finally:
        stream.stop()

    def _to_dbfs(mean_sq: list[float]) -> float:
        if not mean_sq:
            return -120.0
        rms = float(np.sqrt(np.mean(mean_sq))) / 32768.0
        return 20.0 * float(np.log10(max(rms, 1e-6)))

    report = EchoTestReport(
        raw_level_dbfs=_to_dbfs(raw_energy),
        cleaned_level_dbfs=_to_dbfs(clean_energy),
        attenuation_db=_to_dbfs(raw_energy) - _to_dbfs(clean_energy),
        vad_events=vad_events,
        processor=processor.name,
    )
    print("\nEcho test report")
    print("----------------")
    print(f"Processor:          {report.processor}")
    print(f"Raw echo level:     {report.raw_level_dbfs:7.1f} dBFS")
    print(f"Cleaned echo level: {report.cleaned_level_dbfs:7.1f} dBFS")
    print(f"Echo attenuation:   {report.attenuation_db:7.1f} dB")
    print(f"VAD self-triggers:  {report.vad_events}")
    print(f"Result:             {'PASS' if report.passed else 'FAIL'}")
    return 0 if report.passed else 1
