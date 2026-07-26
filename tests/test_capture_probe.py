"""Capture-probe measurement logic (M7.2 diagnostic).

The device-dependent half (`_record`, `run_capture_test`) needs a microphone
and is exercised by hand, like `eva listen` / `eva echo-test`. Everything a
conclusion is drawn from — the signal metrics, WER, and the verdict the report
prints — is pure and tested here, because those are what the investigation
will actually be believed on.
"""

from __future__ import annotations

import contextlib
from dataclasses import replace

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
    _spy_on_transcribe,
    normalize_words,
    signal_metrics,
    word_error_rate,
)
from eva.audio.frames import SAMPLE_RATE
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
