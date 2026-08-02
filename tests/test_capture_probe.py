"""Capture-probe measurement logic (M7.2 diagnostic).

The fully device-dependent half (`run_capture_test`) needs a microphone and is
exercised by hand, like `eva listen` / `eva echo-test`. Everything a
conclusion is drawn from — the signal metrics, WER, and the verdict the report
prints — is pure and tested here, because those are what the investigation
will actually be believed on.

`_record`'s stop-condition logic (Batch 4A) is the exception: it is tested
below by faking `DuplexAudioStream` and the VAD engine so the real
`SpeechSegmenter` state machine can be driven deterministically, without a
microphone or real wall-clock delay.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import ClassVar

import numpy as np
import pytest

from eva.audio.capture_probe import (
    AsrConfig,
    CaptureReport,
    Environment,
    SegmentationReport,
    SignalMetrics,
    VariantReport,
    _asr_config,
    _record,
    _spy_on_transcribe,
    normalize_words,
    signal_metrics,
    word_error_rate,
)
from eva.audio.frames import SAMPLE_RATE
from eva.audio.ring import FrameRing
from eva.config.settings import Settings


def _tone(seconds: float, freq: float, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (np.sin(2 * np.pi * freq * t) * amplitude * 32767).astype(np.int16)


class TestSignalMetrics:
    def test_empty_input_reads_as_silence(self) -> None:
        m = signal_metrics(np.zeros(0, dtype=np.int16))
        assert m.duration_s == 0.0
        assert m.peak_dbfs == -120.0
        assert m.clipping_percent == 0.0

    def test_duration_and_sample_rate(self) -> None:
        m = signal_metrics(_tone(0.5, 440.0))
        assert m.duration_s == 0.5
        assert m.sample_rate == SAMPLE_RATE

    def test_peak_and_rms_of_a_known_tone(self) -> None:
        m = signal_metrics(_tone(0.2, 440.0, amplitude=0.5))
        assert -6.5 < m.peak_dbfs < -5.5  # 0.5 full scale ≈ -6 dBFS
        assert m.rms_dbfs < m.peak_dbfs  # RMS of a sine is ~3 dB below peak

    def test_clipping_is_detected_and_quantified(self) -> None:
        pcm = np.zeros(1000, dtype=np.int16)
        pcm[:100] = 32767
        m = signal_metrics(pcm)
        assert 9.9 < m.clipping_percent < 10.1

    def test_clean_signal_reports_no_clipping(self) -> None:
        assert signal_metrics(_tone(0.2, 440.0, amplitude=0.3)).clipping_percent == 0.0

    def test_high_frequency_energy_separates_fricative_content(self) -> None:
        """The metric the whole investigation turns on: a low-frequency-only
        signal must read near zero, a high-frequency one near 100 %."""
        low = signal_metrics(_tone(0.3, 300.0)).high_freq_energy_percent
        high = signal_metrics(_tone(0.3, 6000.0)).high_freq_energy_percent
        assert low < 1.0
        assert high > 99.0


class TestWordErrorRate:
    def test_identical_text_scores_zero(self) -> None:
        assert word_error_rate("I said fox, not box.", "I said fox not box") == (0.0, 0, 5)

    def test_punctuation_and_case_are_ignored(self) -> None:
        wer, _edits, _n = word_error_rate("No, fox.", "no FOX")
        assert wer == 0.0

    def test_one_substitution_in_five_words(self) -> None:
        wer, edits, n = word_error_rate("I said fox, not box.", "I said box, not box.")
        assert (edits, n) == (1, 5)
        assert wer == 20.0

    def test_empty_reference_scores_zero(self) -> None:
        assert word_error_rate("", "anything at all") == (0.0, 0, 0)

    def test_empty_hypothesis_is_all_deletions(self) -> None:
        assert word_error_rate("one two three", "") == (100.0, 3, 3)

    def test_normalization_drops_punctuation(self) -> None:
        assert normalize_words("It's a Fox, not a Box!") == ["its", "a", "fox", "not", "a", "box"]


def _report(
    *,
    raw_wer: float | None,
    proc_wer: float | None,
    raw_rms: float = -20.0,
    reference: str | None = "I said fox, not box.",
) -> CaptureReport:
    def variant(name: str, wer: float | None, rms: float) -> VariantReport:
        return VariantReport(
            name=name,
            path=f"{name}.wav",
            metrics=SignalMetrics(1.0, SAMPLE_RATE, -6.0, rms, 0.0, 2.0),
            transcript="whatever",
            decode_ms=100,
            wer_percent=wer,
        )

    return CaptureReport(
        recorded_at="20260726-120000",
        reference=reference,
        processor="webrtc-apm",
        echo_cancellation=True,
        noise_suppression=True,
        auto_gain_control=True,
        mic_gain=1.0,
        input_device=None,
        device_native_sample_rate=44100.0,
        capture_ms=6000,
        frames_dropped=0,
        segmentation=SegmentationReport(True, False, 1500, 900),
        asr=_ASR_CONFIG,
        environment=_ENVIRONMENT,
        variants=[variant("raw", raw_wer, raw_rms), variant("processed", proc_wer, -20.0)],
    )


_ASR_CONFIG = AsrConfig(
    engine="faster-whisper",
    model_id="faster-whisper/small",
    model_name="small",
    device="cuda",
    compute_type_configured="auto",
    compute_type_resolved="int8",
    language="en",
    beam_size=1,
    best_of=5,
    temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    initial_prompt=None,
    vad_filter=False,
    condition_on_previous_text=False,
    compression_ratio_threshold=2.4,
    log_prob_threshold=-1.0,
    no_speech_threshold=0.6,
)

_ENVIRONMENT = Environment(
    timestamp_utc="2026-07-26T12:00:00+00:00",
    eva_version="0.5.0a1",
    git_commit="abc1234",
    git_dirty=False,
    backend="cuda",
    gpu_name="NVIDIA GeForce RTX 3060 Laptop GPU",
    gpu_vram_mb=6144,
    cuda_device_count=1,
    faster_whisper_version="1.1.0",
    ctranslate2_version="4.8.1",
    python_version="3.12.0",
    platform="Windows 11 (win32)",
)


class TestVerdict:
    """The report's conclusion is the product — it must not mislead."""

    def test_both_wrong_points_outside_eva(self) -> None:
        text = _report(raw_wer=20.0, proc_wer=20.0).render()
        assert "OUTSIDE EVA" in text

    def test_raw_right_processed_wrong_points_at_the_front_end(self) -> None:
        text = _report(raw_wer=0.0, proc_wer=20.0).render()
        assert "FRONT END" in text

    def test_both_right_points_at_segmentation(self) -> None:
        text = _report(raw_wer=0.0, proc_wer=0.0).render()
        assert "segmentation or runtime" in text

    def test_front_end_helping_is_reported_honestly(self) -> None:
        text = _report(raw_wer=20.0, proc_wer=0.0).render()
        assert "HELPING" in text

    def test_a_too_quiet_recording_refuses_to_conclude(self) -> None:
        """A silent-microphone run must not be read as a front-end result."""
        text = _report(raw_wer=100.0, proc_wer=100.0, raw_rms=-70.0).render()
        assert "too quiet to conclude" in text
        assert "OUTSIDE EVA" not in text

    def test_without_a_reference_no_verdict_is_invented(self) -> None:
        text = _report(raw_wer=None, proc_wer=None, reference=None).render()
        assert "No reference text supplied" in text


class TestReportSerialization:
    def test_round_trips_to_json_friendly_types(self) -> None:
        import json

        data = _report(raw_wer=0.0, proc_wer=20.0).to_dict()
        assert json.loads(json.dumps(data))["variants"][1]["wer_percent"] == 20.0
        assert data["segmentation"]["speech_ms"] == 900

    def test_apm_flag_follows_the_processor(self) -> None:
        report = _report(raw_wer=0.0, proc_wer=0.0)
        assert report.apm_enabled is True
        report.processor = "passthrough"
        assert report.apm_enabled is False
        assert "APM OFF" in report.render()

    def test_render_includes_every_measurement_asked_for(self) -> None:
        text = _report(raw_wer=0.0, proc_wer=20.0).render()
        for expected in (
            "duration",
            "peak / rms",
            "clipping",
            "decode",
            "transcript",
            "WER",
            "Device native SR",
            "Frames dropped",
            "Segmentation",
        ):
            assert expected in text, f"missing {expected!r} from the report"


class TestProvenance:
    """A report has to be reproducible months later, so every field that
    determines the result must be recorded — and recorded truthfully."""

    def test_json_carries_the_full_decode_configuration(self) -> None:
        import json

        data = json.loads(json.dumps(_report(raw_wer=0.0, proc_wer=0.0).to_dict()))
        asr = data["asr"]
        for field_name in (
            "model_id",
            "language",
            "compute_type_resolved",
            "beam_size",
            "initial_prompt",
            "vad_filter",
            "temperature",
            "best_of",
            "compression_ratio_threshold",
            "log_prob_threshold",
            "no_speech_threshold",
        ):
            assert field_name in asr, f"{field_name} missing from the JSON report"
        assert asr["temperature"] == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        assert asr["initial_prompt"] is None  # null, not absent

    def test_json_carries_the_environment(self) -> None:
        import json

        env = json.loads(json.dumps(_report(raw_wer=0.0, proc_wer=0.0).to_dict()))["environment"]
        assert env["gpu_name"].startswith("NVIDIA")
        assert env["backend"] == "cuda"
        assert env["git_commit"] == "abc1234"
        assert env["timestamp_utc"].endswith("+00:00")
        assert env["ctranslate2_version"] == "4.8.1"

    def test_render_shows_config_and_environment(self) -> None:
        text = _report(raw_wer=0.0, proc_wer=0.0).render()
        for expected in (
            "ASR configuration",
            "beam_size:         1",
            "no_speech_threshold:         0.6",
            "int8  (settings: auto)",  # resolved value, and what produced it
            "Environment",
            "RTX 3060",
            "abc1234",
        ):
            assert expected in text, f"missing {expected!r} from the report"

    def test_a_dirty_checkout_is_flagged(self) -> None:
        """Measurements taken with uncommitted changes are not reproducible
        from the commit alone; the report must say so."""
        report = _report(raw_wer=0.0, proc_wer=0.0)
        assert report.environment is not None
        report.environment = replace(report.environment, git_dirty=True)
        assert "dirty" in report.render()

    def test_spy_records_the_arguments_the_adapter_really_passes(self) -> None:
        """The whole point of observing the call: these values come from
        `FasterWhisperASR`, not from a copy of its defaults kept here. If the
        adapter changes its decode settings, this test changes with it."""
        pytest.importorskip("faster_whisper")
        from faster_whisper import WhisperModel

        recorded: dict[str, object] = {}
        with _spy_on_transcribe(recorded), contextlib.suppress(Exception):
            WhisperModel.transcribe(  # no real model needed: the spy runs first
                object(),  # type: ignore[arg-type]
                np.zeros(16000, dtype=np.float32),
                language="en",
                beam_size=1,
                condition_on_previous_text=False,
                vad_filter=False,
                initial_prompt=None,
            )

        assert recorded["beam_size"] == 1
        assert recorded["condition_on_previous_text"] is False
        # Never passed by the adapter — filled in from the library's defaults,
        # which is exactly what makes the report complete.
        assert recorded["best_of"] == 5
        assert recorded["no_speech_threshold"] == 0.6
        assert recorded["compression_ratio_threshold"] == 2.4

    def test_spy_always_restores_the_original_method(self) -> None:
        pytest.importorskip("faster_whisper")
        from faster_whisper import WhisperModel

        before = WhisperModel.transcribe
        with contextlib.suppress(RuntimeError), _spy_on_transcribe({}):
            raise RuntimeError("boom")
        assert WhisperModel.transcribe is before

    def test_asr_config_prefers_observed_values_over_settings(self) -> None:
        """`compute_type` is "auto" in settings; the report must show what the
        run actually used."""
        settings = Settings()
        observed = {
            "language": "en",
            "beam_size": 1,
            "_compute_type": "float16",
            "_device": "cuda",
        }
        config = _asr_config(settings, _StubAsr(), observed)
        assert config.compute_type_configured == "auto"
        assert config.compute_type_resolved == "float16"
        assert config.device == "cuda"
        assert config.model_name == "small"

    def test_asr_config_survives_a_non_faster_whisper_engine(self) -> None:
        """A different ASR adapter records nothing; the report degrades to
        nulls instead of inventing values."""
        config = _asr_config(Settings(), _StubAsr(), {})
        assert config.beam_size is None
        assert config.compute_type_resolved is None
        assert config.model_id == Settings().asr.model


class _StubAsr:
    _model_name = "small"
    device = "cpu"


# ──────────────── _record() stop-condition harness ────────────────
#
# `_record` drives the real `SpeechSegmenter` live over the processed stream,
# so these tests fake only the device boundary (`DuplexAudioStream`, the VAD
# engine) and a controllable clock — the segmenter itself, `FrameChunker`, and
# `FrameRing` are the genuine article. A chunk's speech/silence classification
# is a marker (`chunk[0]`), not real signal analysis: that decision already has
# its own test suite (Silero/`vad.base`); here only the segmenter's state
# transitions and the timing around them are under test.

_CHUNK_SAMPLES = 160  # 10 ms @ 16 kHz — chunk_ms comes out to a clean integer


def _speech_chunk(n: int = _CHUNK_SAMPLES) -> np.ndarray:
    return np.ones(n, dtype=np.int16)


def _silence_chunk(n: int = _CHUNK_SAMPLES) -> np.ndarray:
    return np.zeros(n, dtype=np.int16)


class _FakeVad:
    chunk_samples = _CHUNK_SAMPLES

    def process(self, chunk: np.ndarray) -> float:
        return 1.0 if chunk[0] == 1 else 0.0

    def reset(self) -> None:
        pass


class _FakeDuplexStream:
    """Replaces `DuplexAudioStream`: no device, just exposes the real rings
    `_record` constructed so the fake clock/sleep can push scripted frames."""

    instances: ClassVar[list[_FakeDuplexStream]] = []
    pending_straggler_raw: ClassVar[np.ndarray | None] = None
    pending_straggler_clean: ClassVar[np.ndarray | None] = None

    def __init__(
        self,
        processor: object,
        playback: object,
        capture_ring: FrameRing,
        *,
        input_device: object = None,
        output_device: object = None,
        mic_gain: float = 1.0,
        raw_tap: FrameRing | None = None,
    ) -> None:
        assert raw_tap is not None
        self.capture_ring = capture_ring
        self.raw_ring = raw_tap
        self.stopped = False
        _FakeDuplexStream.instances.append(self)

    def start(self) -> None:
        pass

    def stop(self) -> None:
        # Simulates the audio-callback thread pushing one more frame in the
        # brief window between the main loop's last pop and the stream
        # actually closing — the reason the post-loop drain exists at all.
        if _FakeDuplexStream.pending_straggler_raw is not None:
            self.raw_ring.push(_FakeDuplexStream.pending_straggler_raw)
        if _FakeDuplexStream.pending_straggler_clean is not None:
            self.capture_ring.push(_FakeDuplexStream.pending_straggler_clean)
        self.stopped = True


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def perf_counter(self) -> float:
        return self.now


def _scripted_sleep(
    clock: _FakeClock, script: Sequence[tuple[float, np.ndarray | None]]
) -> Callable[[float], None]:
    """Each real `time.sleep()` call in `_record`'s idle branch consumes one
    `(advance_seconds, frame_or_None)` entry: advances the fake clock, and —
    if a frame is given — pushes it into the most recently constructed fake
    stream's rings, standing in for the next audio-callback tick. Once the
    script runs out, sleep just advances the clock, like a quiet microphone."""
    state = {"i": 0}

    def fake_sleep(requested_s: float) -> None:
        i = state["i"]
        if i >= len(script):
            clock.now += requested_s
            return
        advance, frame = script[i]
        state["i"] += 1
        clock.now += advance
        if frame is not None:
            stream = _FakeDuplexStream.instances[-1]
            stream.raw_ring.push(frame)
            stream.capture_ring.push(frame)

    return fake_sleep


def _vad_settings(**overrides: object) -> object:
    return Settings().vad.model_copy(update=overrides)


def _run_record(
    monkeypatch: pytest.MonkeyPatch,
    script: Sequence[tuple[float, np.ndarray | None]],
    *,
    seconds: float = 10.0,
    vad_overrides: dict[str, object] | None = None,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    import eva.audio.capture_probe as capture_probe_module

    _FakeDuplexStream.instances = []
    clock = _FakeClock()
    settings = Settings().model_copy(update={"vad": _vad_settings(**(vad_overrides or {}))})
    monkeypatch.setattr(time, "perf_counter", clock.perf_counter)
    monkeypatch.setattr(time, "sleep", _scripted_sleep(clock, script))
    monkeypatch.setattr(capture_probe_module, "DuplexAudioStream", _FakeDuplexStream)
    monkeypatch.setattr("eva.vad.registry.create_vad", lambda engine_id: _FakeVad())
    return _record(settings, seconds)


class TestRecordingStopCondition:
    """Batch 4A: `_record` stops on the segmenter's terminal-state invariant
    (`utterance_active` True → False) instead of severing a fixed wall-clock
    window mid-utterance."""

    @pytest.fixture(autouse=True)
    def _reset_stragglers(self) -> None:
        _FakeDuplexStream.pending_straggler_raw = None
        _FakeDuplexStream.pending_straggler_clean = None

    def test_stops_right_after_utterance_end(self, monkeypatch: pytest.MonkeyPatch) -> None:
        script = [(0.001, _speech_chunk()) for _ in range(5)] + [
            (0.001, _silence_chunk()) for _ in range(5)
        ]
        raw, clean, capture_ms, dropped = _run_record(
            monkeypatch, script, vad_overrides={"silence_timeout_ms": 50, "min_speech_ms": 20}
        )
        assert clean.size == 10 * _CHUNK_SAMPLES  # exactly what was needed, nothing more
        assert raw.size == clean.size
        assert capture_ms < 1000  # nowhere near the 10 s ceiling used by default
        assert dropped == 0

    def test_stops_right_after_utterance_discarded_as_noise(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The invariant is `utterance_active` flipping False, not a specific
        event type — a too-short burst (`UtteranceDiscarded`) must stop
        recording exactly as promptly as a real `UtteranceEnd`."""
        script = [(0.001, _speech_chunk()) for _ in range(2)] + [
            (0.001, _silence_chunk()) for _ in range(5)
        ]
        _raw, clean, capture_ms, _dropped = _run_record(
            monkeypatch, script, vad_overrides={"silence_timeout_ms": 50, "min_speech_ms": 30}
        )
        assert clean.size == 7 * _CHUNK_SAMPLES
        assert capture_ms < 1000

    def test_ceiling_does_not_cut_off_speech_in_progress(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The bug this batch fixes: a fixed window severed speech that was
        still in progress. Here the `seconds` ceiling elapses while the
        segmenter is still mid-utterance (no frame pushed, clock alone
        advances past it) — recording must keep going and capture every
        speech chunk regardless."""
        script = (
            [(0.001, _speech_chunk()) for _ in range(3)]
            + [(0.5, None)]  # ceiling (50 ms) elapses here — still mid-utterance
            + [(0.001, _silence_chunk()) for _ in range(5)]
        )
        raw, clean, capture_ms, dropped = _run_record(
            monkeypatch,
            script,
            seconds=0.05,
            vad_overrides={"silence_timeout_ms": 50, "min_speech_ms": 10},
        )
        assert capture_ms > 50  # ran past the ceiling instead of stopping at it
        assert clean.size == 8 * _CHUNK_SAMPLES  # all 3 speech + 5 silence chunks preserved
        assert raw.size == clean.size
        assert dropped == 0

    def test_ceiling_stops_recording_when_no_utterance_is_active(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No speech ever starts: the ceiling is the only thing that can
        stop recording, exactly like the old fixed-window behavior."""
        script = [(0.03, None)]  # nothing but silence/no data, past the ceiling
        raw, clean, capture_ms, _dropped = _run_record(monkeypatch, script, seconds=0.02)
        assert capture_ms >= 20
        assert raw.size == 0
        assert clean.size == 0

    def test_max_utterance_s_forced_finish_is_the_ultimate_backstop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Continuous speech that never pauses is bounded by the segmenter's
        own `max_utterance_s` safety timeout, not by the recording ceiling —
        proving recording can never hang even if speech never stops."""
        chunk_count = 500  # 500 * 10 ms = 5 s = the minimum allowed max_utterance_s
        script = [(0.0001, _speech_chunk()) for _ in range(chunk_count)]
        raw, clean, _capture_ms, dropped = _run_record(
            monkeypatch,
            script,
            seconds=0.01,
            vad_overrides={"max_utterance_s": 5, "silence_timeout_ms": 800},
        )
        assert clean.size == chunk_count * _CHUNK_SAMPLES
        assert raw.size == clean.size
        assert dropped == 0

    def test_only_the_first_utterance_is_captured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The probe records one utterance per run: once the first completes,
        a second utterance's frames must never even be consumed."""
        first = [(0.001, _speech_chunk()) for _ in range(5)] + [
            (0.001, _silence_chunk()) for _ in range(5)
        ]
        second = [(0.001, _speech_chunk()) for _ in range(3)]
        _raw, clean, _capture_ms, _dropped = _run_record(
            monkeypatch,
            first + second,
            vad_overrides={"silence_timeout_ms": 50, "min_speech_ms": 20},
        )
        assert clean.size == 10 * _CHUNK_SAMPLES  # not 13 — the second batch was never touched

    def test_ring_capacity_covers_ceiling_plus_max_utterance(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The true worst case is speech starting just before the ceiling and
        running the full `max_utterance_s` — the ring must be sized for that
        sum, not for the ceiling alone, or a long utterance could overflow it."""
        import eva.audio.capture_probe as capture_probe_module

        capacities: list[int] = []

        def spy_ring(capacity_frames: int) -> FrameRing:
            capacities.append(capacity_frames)
            return FrameRing(capacity_frames)

        monkeypatch.setattr(capture_probe_module, "FrameRing", spy_ring)
        _run_record(monkeypatch, [(0.01, None)], seconds=6.0, vad_overrides={"max_utterance_s": 30})
        assert capacities == [int((6.0 + 30) * 100) + 200] * 2  # raw ring, clean ring

    def test_post_stop_drain_still_collects_a_straggler_frame(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A frame the fake callback pushes only inside `stop()` — modeling
        the real race between the main loop's last pop and the stream
        actually closing — must still end up in the final recording."""
        script = [(0.001, _speech_chunk()) for _ in range(5)] + [
            (0.001, _silence_chunk()) for _ in range(5)
        ]
        _FakeDuplexStream.pending_straggler_raw = _silence_chunk()
        _FakeDuplexStream.pending_straggler_clean = _silence_chunk()
        raw, clean, _capture_ms, _dropped = _run_record(
            monkeypatch, script, vad_overrides={"silence_timeout_ms": 50, "min_speech_ms": 20}
        )
        assert clean.size == 11 * _CHUNK_SAMPLES
        assert raw.size == 11 * _CHUNK_SAMPLES

    def test_raw_and_clean_stay_aligned_when_only_one_ring_gets_a_straggler(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A race that lands a straggler in only one ring must not desync the
        two recordings — the existing `min(...)` trim has to absorb it."""
        script = [(0.001, _speech_chunk()) for _ in range(5)] + [
            (0.001, _silence_chunk()) for _ in range(5)
        ]
        _FakeDuplexStream.pending_straggler_raw = _silence_chunk()  # clean ring gets none
        raw, clean, _capture_ms, _dropped = _run_record(
            monkeypatch, script, vad_overrides={"silence_timeout_ms": 50, "min_speech_ms": 20}
        )
        assert raw.size == clean.size == 10 * _CHUNK_SAMPLES

    def test_keyboard_interrupt_still_stops_the_stream_cleanly(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ctrl+C must still work exactly as before: caught, the stream is
        stopped, and whatever was captured so far is returned rather than
        raising out of `_record`."""
        import eva.audio.capture_probe as capture_probe_module

        clock = _FakeClock()
        _FakeDuplexStream.instances = []
        _FakeDuplexStream.pending_straggler_raw = None
        _FakeDuplexStream.pending_straggler_clean = None

        def raising_sleep(_requested_s: float) -> None:
            stream = _FakeDuplexStream.instances[-1]
            stream.raw_ring.push(_speech_chunk())
            stream.capture_ring.push(_speech_chunk())
            raise KeyboardInterrupt

        monkeypatch.setattr(time, "perf_counter", clock.perf_counter)
        monkeypatch.setattr(time, "sleep", raising_sleep)
        monkeypatch.setattr(capture_probe_module, "DuplexAudioStream", _FakeDuplexStream)
        monkeypatch.setattr("eva.vad.registry.create_vad", lambda engine_id: _FakeVad())

        raw, clean, _capture_ms, _dropped = _record(Settings(), 10.0)

        assert _FakeDuplexStream.instances[-1].stopped is True
        assert raw.size == _CHUNK_SAMPLES  # the one frame pushed before the interrupt
        assert clean.size == _CHUNK_SAMPLES
