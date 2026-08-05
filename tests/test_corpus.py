"""Audio corpus recorder (Batch 4B).

Workflow tests (`TestRunCorpusRecordWorkflow`) fake only the recording
boundary (`eva.audio.corpus._capture_one`) — everything about prompt
sequencing, resuming, overwrite confirmation, and file writing is real.
`TestReusesTheRealCapturePipeline` goes one layer deeper: it fakes the same
device/VAD boundary `test_capture_probe.py` already fakes for `_record`
itself, proving `run_corpus_record` truly drives the real capture pipeline —
`DuplexAudioStream`, the VAD, and the real `SpeechSegmenter` state machine —
rather than a lookalike.
"""

from __future__ import annotations

import time
import wave
from pathlib import Path

import numpy as np
import pytest

from eva.audio.corpus import (
    _MANIFEST_FILENAME,
    _hash_prompts_file,
    _load_manifest,
    _resume_index,
    _stem,
    _stem_width,
    _verify_prompt_integrity,
    read_prompts,
    run_corpus_record,
)
from eva.audio.frames import SAMPLE_RATE
from eva.config.settings import Settings
from eva.core.errors import ConfigError


def _prompts_file(tmp_path: Path, lines: list[str]) -> Path:
    path = tmp_path / "prompts.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _scripted_input(monkeypatch: pytest.MonkeyPatch, responses: list[str]) -> None:
    it = iter(responses)

    def fake_input(_prompt: str = "") -> str:
        return next(it)

    monkeypatch.setattr("builtins.input", fake_input)


def _tone(n: int, value: int = 500) -> np.ndarray:
    return np.full(n, value, dtype=np.int16)


class TestReadPrompts:
    def test_reads_non_blank_lines_stripped(self, tmp_path: Path) -> None:
        path = tmp_path / "prompts.txt"
        path.write_text("Hello EVA.\n\n  Good morning.  \n\nHow are you?\n", encoding="utf-8")
        assert read_prompts(path) == ["Hello EVA.", "Good morning.", "How are you?"]

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError):
            read_prompts(tmp_path / "nope.txt")

    def test_utf8_content_is_preserved_exactly(self, tmp_path: Path) -> None:
        path = tmp_path / "prompts.txt"
        path.write_text("Café au lait, s'il vous plaît.\n", encoding="utf-8")
        assert read_prompts(path) == ["Café au lait, s'il vous plaît."]


class TestStemWidth:
    def test_default_is_three_digits(self) -> None:
        assert _stem_width(150) == 3
        assert _stem(23, 3) == "023"

    def test_widens_for_a_corpus_over_999(self) -> None:
        assert _stem_width(1200) == 4
        assert _stem(23, 4) == "0023"


class TestResumeIndex:
    def test_empty_directories_start_at_one(self, tmp_path: Path) -> None:
        speech, transcripts = tmp_path / "speech", tmp_path / "transcripts"
        speech.mkdir()
        transcripts.mkdir()
        assert _resume_index(speech, transcripts, 5, 3) == 1

    def test_skips_complete_pairs(self, tmp_path: Path) -> None:
        speech, transcripts = tmp_path / "speech", tmp_path / "transcripts"
        speech.mkdir()
        transcripts.mkdir()
        (speech / "001.wav").write_bytes(b"x")
        (transcripts / "001.txt").write_text("a", encoding="utf-8")
        assert _resume_index(speech, transcripts, 5, 3) == 2

    def test_a_wav_missing_its_transcript_is_not_skipped(self, tmp_path: Path) -> None:
        speech, transcripts = tmp_path / "speech", tmp_path / "transcripts"
        speech.mkdir()
        transcripts.mkdir()
        (speech / "001.wav").write_bytes(b"x")  # transcript never written
        assert _resume_index(speech, transcripts, 5, 3) == 1

    def test_all_complete_returns_one_past_the_end(self, tmp_path: Path) -> None:
        speech, transcripts = tmp_path / "speech", tmp_path / "transcripts"
        speech.mkdir()
        transcripts.mkdir()
        for i in (1, 2):
            (speech / f"00{i}.wav").write_bytes(b"x")
            (transcripts / f"00{i}.txt").write_text("a", encoding="utf-8")
        assert _resume_index(speech, transcripts, 2, 3) == 3


class TestRunCorpusRecordWorkflow:
    def test_no_prompts_is_a_clean_failure(self, tmp_path: Path) -> None:
        path = tmp_path / "prompts.txt"
        path.write_text("\n\n", encoding="utf-8")
        code = run_corpus_record(Settings(), prompts_path=path, fixtures_dir=tmp_path / "fixtures")
        assert code == 1

    def test_records_all_prompts_in_order(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        prompts_path = _prompts_file(tmp_path, ["Hello EVA.", "Good morning."])
        fixtures_dir = tmp_path / "fixtures"
        seen_seconds: list[float] = []

        def fake_capture(settings: Settings, seconds: float) -> np.ndarray:
            seen_seconds.append(seconds)
            return _tone(1600)

        monkeypatch.setattr("eva.audio.corpus._capture_one", fake_capture)
        _scripted_input(monkeypatch, ["", "n", "", "n"])  # Enter,Next, Enter,Next

        code = run_corpus_record(
            Settings(), prompts_path=prompts_path, fixtures_dir=fixtures_dir, seconds=7.0
        )

        assert code == 0
        assert seen_seconds == [7.0, 7.0]
        assert (fixtures_dir / "speech" / "001.wav").exists()
        assert (fixtures_dir / "speech" / "002.wav").exists()
        assert (fixtures_dir / "transcripts" / "001.txt").read_text(
            encoding="utf-8"
        ) == "Hello EVA."
        assert (fixtures_dir / "transcripts" / "002.txt").read_text(
            encoding="utf-8"
        ) == "Good morning."

    def test_wav_format_matches_capture_test_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        prompts_path = _prompts_file(tmp_path, ["Hello EVA."])
        fixtures_dir = tmp_path / "fixtures"
        pcm = _tone(1600, 777)
        monkeypatch.setattr("eva.audio.corpus._capture_one", lambda settings, seconds: pcm)
        _scripted_input(monkeypatch, ["", "n"])

        run_corpus_record(Settings(), prompts_path=prompts_path, fixtures_dir=fixtures_dir)

        with wave.open(str(fixtures_dir / "speech" / "001.wav"), "rb") as handle:
            assert handle.getnchannels() == 1
            assert handle.getsampwidth() == 2
            assert handle.getframerate() == SAMPLE_RATE
            assert handle.getnframes() == pcm.size

    def test_no_audio_captured_retries_the_same_prompt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        prompts_path = _prompts_file(tmp_path, ["Hello EVA."])
        fixtures_dir = tmp_path / "fixtures"
        takes = iter([np.zeros(0, dtype=np.int16), _tone(800, 300)])
        monkeypatch.setattr("eva.audio.corpus._capture_one", lambda settings, seconds: next(takes))
        _scripted_input(monkeypatch, ["", "", "n"])  # Enter (empty), Enter (retry), Next

        code = run_corpus_record(Settings(), prompts_path=prompts_path, fixtures_dir=fixtures_dir)

        assert code == 0
        assert (fixtures_dir / "speech" / "001.wav").exists()

    def test_re_record_retakes_the_same_prompt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        prompts_path = _prompts_file(tmp_path, ["Hello EVA."])
        fixtures_dir = tmp_path / "fixtures"
        takes = iter([_tone(100, 1), _tone(200, 2)])
        monkeypatch.setattr("eva.audio.corpus._capture_one", lambda settings, seconds: next(takes))
        _scripted_input(monkeypatch, ["", "r", "", "n"])  # Enter, Re-record, Enter, Next

        run_corpus_record(Settings(), prompts_path=prompts_path, fixtures_dir=fixtures_dir)

        with wave.open(str(fixtures_dir / "speech" / "001.wav"), "rb") as handle:
            assert handle.getnframes() == 200  # the second take, not the discarded first

    def test_quit_stops_early_and_leaves_the_rest_unrecorded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        prompts_path = _prompts_file(tmp_path, ["Hello EVA.", "Good morning.", "Goodbye."])
        fixtures_dir = tmp_path / "fixtures"
        monkeypatch.setattr(
            "eva.audio.corpus._capture_one", lambda settings, seconds: _tone(500, 9)
        )
        _scripted_input(monkeypatch, ["", "n", "", "q"])  # record 1, next, record 2, quit

        code = run_corpus_record(Settings(), prompts_path=prompts_path, fixtures_dir=fixtures_dir)

        assert code == 0
        assert (fixtures_dir / "speech" / "001.wav").exists()
        assert (fixtures_dir / "speech" / "002.wav").exists()
        assert not (fixtures_dir / "speech" / "003.wav").exists()

    def test_resuming_continues_after_the_last_completed_prompt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        prompts_path = _prompts_file(tmp_path, ["Hello EVA.", "Good morning.", "Goodbye."])
        fixtures_dir = tmp_path / "fixtures"
        (fixtures_dir / "speech").mkdir(parents=True)
        (fixtures_dir / "transcripts").mkdir(parents=True)
        (fixtures_dir / "speech" / "001.wav").write_bytes(b"already recorded")
        (fixtures_dir / "transcripts" / "001.txt").write_text("Hello EVA.", encoding="utf-8")
        monkeypatch.setattr(
            "eva.audio.corpus._capture_one", lambda settings, seconds: _tone(500, 9)
        )
        _scripted_input(monkeypatch, ["", "n", "", "n"])

        code = run_corpus_record(Settings(), prompts_path=prompts_path, fixtures_dir=fixtures_dir)

        assert code == 0
        # Prompt 1 was never touched — still the placeholder bytes, not a real WAV.
        assert (fixtures_dir / "speech" / "001.wav").read_bytes() == b"already recorded"
        assert (fixtures_dir / "speech" / "002.wav").exists()
        assert (fixtures_dir / "speech" / "003.wav").exists()

    def test_declining_overwrite_keeps_the_existing_file_and_advances(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Migration/safety case: a file present before this run started is
        never silently replaced."""
        prompts_path = _prompts_file(tmp_path, ["Hello EVA.", "Good morning."])
        fixtures_dir = tmp_path / "fixtures"
        (fixtures_dir / "speech").mkdir(parents=True)
        (fixtures_dir / "transcripts").mkdir(parents=True)
        # An orphaned WAV with no matching transcript — _resume_index lands here.
        (fixtures_dir / "speech" / "001.wav").write_bytes(b"orphaned take")
        monkeypatch.setattr(
            "eva.audio.corpus._capture_one", lambda settings, seconds: _tone(500, 9)
        )
        # record, DECLINE overwrite, record 2, next
        _scripted_input(monkeypatch, ["", "n", "", "n"])

        code = run_corpus_record(Settings(), prompts_path=prompts_path, fixtures_dir=fixtures_dir)

        assert code == 0
        assert (fixtures_dir / "speech" / "001.wav").read_bytes() == b"orphaned take"
        assert not (fixtures_dir / "transcripts" / "001.txt").exists()
        assert (fixtures_dir / "speech" / "002.wav").exists()

    def test_confirming_overwrite_replaces_the_existing_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        prompts_path = _prompts_file(tmp_path, ["Hello EVA."])
        fixtures_dir = tmp_path / "fixtures"
        (fixtures_dir / "speech").mkdir(parents=True)
        (fixtures_dir / "speech" / "001.wav").write_bytes(b"orphaned take")
        monkeypatch.setattr(
            "eva.audio.corpus._capture_one", lambda settings, seconds: _tone(500, 9)
        )
        _scripted_input(monkeypatch, ["", "y", "n"])  # record, CONFIRM overwrite, next

        code = run_corpus_record(Settings(), prompts_path=prompts_path, fixtures_dir=fixtures_dir)

        assert code == 0
        assert (fixtures_dir / "speech" / "001.wav").read_bytes() != b"orphaned take"
        assert (fixtures_dir / "transcripts" / "001.txt").read_text(
            encoding="utf-8"
        ) == "Hello EVA."

    def test_transcript_only_orphan_also_requires_confirmation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Symmetric overwrite guard: a transcript with no paired WAV must be
        confirmed too, not silently replaced just because the WAV is absent."""
        prompts_path = _prompts_file(tmp_path, ["Hello EVA.", "Good morning."])
        fixtures_dir = tmp_path / "fixtures"
        (fixtures_dir / "transcripts").mkdir(parents=True)
        (fixtures_dir / "transcripts" / "001.txt").write_text("stale text", encoding="utf-8")
        monkeypatch.setattr(
            "eva.audio.corpus._capture_one", lambda settings, seconds: _tone(500, 9)
        )
        # record, DECLINE overwrite, record 2, next
        _scripted_input(monkeypatch, ["", "n", "", "n"])

        code = run_corpus_record(Settings(), prompts_path=prompts_path, fixtures_dir=fixtures_dir)

        assert code == 0
        assert not (fixtures_dir / "speech" / "001.wav").exists()
        assert (fixtures_dir / "transcripts" / "001.txt").read_text(
            encoding="utf-8"
        ) == "stale text"
        assert (fixtures_dir / "speech" / "002.wav").exists()

    def test_transcript_only_orphan_overwrite_confirmed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        prompts_path = _prompts_file(tmp_path, ["Hello EVA."])
        fixtures_dir = tmp_path / "fixtures"
        (fixtures_dir / "transcripts").mkdir(parents=True)
        (fixtures_dir / "transcripts" / "001.txt").write_text("stale text", encoding="utf-8")
        monkeypatch.setattr(
            "eva.audio.corpus._capture_one", lambda settings, seconds: _tone(500, 9)
        )
        _scripted_input(monkeypatch, ["", "y", "n"])  # record, CONFIRM overwrite, next

        code = run_corpus_record(Settings(), prompts_path=prompts_path, fixtures_dir=fixtures_dir)

        assert code == 0
        assert (fixtures_dir / "speech" / "001.wav").exists()
        assert (fixtures_dir / "transcripts" / "001.txt").read_text(
            encoding="utf-8"
        ) == "Hello EVA."

    def test_summary_after_saving_shows_index_filename_duration_and_transcript(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        prompts_path = _prompts_file(tmp_path, ["Hello EVA."])
        fixtures_dir = tmp_path / "fixtures"
        monkeypatch.setattr(
            "eva.audio.corpus._capture_one", lambda settings, seconds: _tone(SAMPLE_RATE * 2)
        )
        _scripted_input(monkeypatch, ["", "n"])

        code = run_corpus_record(Settings(), prompts_path=prompts_path, fixtures_dir=fixtures_dir)

        assert code == 0
        out = capsys.readouterr().out
        assert "[1/1]" in out
        assert "001.wav" in out
        assert "2.00s" in out
        assert "Hello EVA." in out

    def test_invalid_menu_choice_reprompts_without_re_recording(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        prompts_path = _prompts_file(tmp_path, ["Hello EVA."])
        fixtures_dir = tmp_path / "fixtures"
        calls = 0

        def fake_capture(settings: Settings, seconds: float) -> np.ndarray:
            nonlocal calls
            calls += 1
            return _tone(500, 9)

        monkeypatch.setattr("eva.audio.corpus._capture_one", fake_capture)
        _scripted_input(monkeypatch, ["", "xyz", "n"])  # Enter, garbage, then Next

        code = run_corpus_record(Settings(), prompts_path=prompts_path, fixtures_dir=fixtures_dir)

        assert code == 0
        assert calls == 1  # the garbage menu answer never triggered a re-record

    def test_keyboard_interrupt_stops_gracefully(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        prompts_path = _prompts_file(tmp_path, ["Hello EVA.", "Good morning."])
        fixtures_dir = tmp_path / "fixtures"

        def raise_interrupt(_prompt: str = "") -> str:
            raise KeyboardInterrupt

        monkeypatch.setattr("builtins.input", raise_interrupt)

        code = run_corpus_record(Settings(), prompts_path=prompts_path, fixtures_dir=fixtures_dir)

        assert code == 1
        assert not (fixtures_dir / "speech" / "001.wav").exists()


class TestReusesTheRealCapturePipeline:
    """Fakes only the device/VAD boundary `test_capture_probe.py` already
    fakes for `_record` itself — everything above that (the real
    `DuplexAudioStream` call shape, the real `SpeechSegmenter`, the real
    `_write_wav`) runs for real, proving `run_corpus_record` is a caller of
    the existing pipeline, not a lookalike."""

    def test_end_to_end_through_the_real_record_function(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import eva.audio.capture_probe as capture_probe_module
        from tests.test_capture_probe import (
            _FakeClock,
            _FakeDuplexStream,
            _FakeVad,
            _scripted_sleep,
            _silence_chunk,
            _speech_chunk,
        )

        _FakeDuplexStream.instances = []
        _FakeDuplexStream.pending_straggler_raw = None
        _FakeDuplexStream.pending_straggler_clean = None
        clock = _FakeClock()
        script: list[tuple[float, np.ndarray | None]] = [
            (0.001, _speech_chunk()) for _ in range(5)
        ] + [(0.001, _silence_chunk()) for _ in range(5)]
        monkeypatch.setattr(time, "perf_counter", clock.perf_counter)
        monkeypatch.setattr(time, "sleep", _scripted_sleep(clock, script))
        monkeypatch.setattr(capture_probe_module, "DuplexAudioStream", _FakeDuplexStream)
        monkeypatch.setattr("eva.vad.registry.create_vad", lambda engine_id: _FakeVad())

        settings = Settings().model_copy(
            update={
                "vad": Settings().vad.model_copy(
                    update={"silence_timeout_ms": 50, "min_speech_ms": 20}
                )
            }
        )
        prompts_path = _prompts_file(tmp_path, ["Hello EVA."])
        fixtures_dir = tmp_path / "fixtures"
        _scripted_input(monkeypatch, ["", "n"])

        code = run_corpus_record(
            settings, prompts_path=prompts_path, fixtures_dir=fixtures_dir, seconds=10.0
        )

        assert code == 0
        wav_path = fixtures_dir / "speech" / "001.wav"
        assert wav_path.exists()
        with wave.open(str(wav_path), "rb") as handle:
            assert handle.getnframes() == 10 * 160  # 5 speech + 5 silence chunks, 10 ms @ 16 kHz


class TestPromptIntegrityManifest:
    """Editing `prompts.txt` after recording begins silently detaches an
    already-saved transcript from the text now on that line — nothing else
    catches it. The manifest is the guard against that."""

    def _run_one_prompt(
        self, monkeypatch: pytest.MonkeyPatch, prompts_path: Path, fixtures_dir: Path
    ) -> int:
        """Records prompt 1 only, then quits — establishes a manifest baseline
        without needing to script through every remaining prompt."""
        monkeypatch.setattr(
            "eva.audio.corpus._capture_one", lambda settings, seconds: _tone(500, 9)
        )
        _scripted_input(monkeypatch, ["", "q"])
        return run_corpus_record(Settings(), prompts_path=prompts_path, fixtures_dir=fixtures_dir)

    def test_first_session_persists_a_manifest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        prompts_path = _prompts_file(tmp_path, ["Hello EVA.", "Good morning."])
        fixtures_dir = tmp_path / "fixtures"

        self._run_one_prompt(monkeypatch, prompts_path, fixtures_dir)

        manifest = _load_manifest(fixtures_dir)
        assert manifest is not None
        assert manifest.prompt_count == 2
        assert manifest.prompts_sha256 == _hash_prompts_file(prompts_path)
        assert (fixtures_dir / _MANIFEST_FILENAME).exists()

    def test_resume_with_unchanged_prompts_does_not_abort(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        prompts_path = _prompts_file(tmp_path, ["Hello EVA.", "Good morning."])
        fixtures_dir = tmp_path / "fixtures"
        self._run_one_prompt(monkeypatch, prompts_path, fixtures_dir)

        # Second session, prompts.txt untouched.
        monkeypatch.setattr(
            "eva.audio.corpus._capture_one", lambda settings, seconds: _tone(500, 9)
        )
        _scripted_input(monkeypatch, ["", "n"])
        code = run_corpus_record(Settings(), prompts_path=prompts_path, fixtures_dir=fixtures_dir)

        assert code == 0
        assert (fixtures_dir / "speech" / "002.wav").exists()

    def test_editing_prompt_text_after_recording_aborts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        prompts_path = _prompts_file(tmp_path, ["Hello EVA.", "Good morning."])
        fixtures_dir = tmp_path / "fixtures"
        self._run_one_prompt(monkeypatch, prompts_path, fixtures_dir)

        prompts_path.write_text("Hello EVA, changed.\nGood morning.\n", encoding="utf-8")

        with pytest.raises(ConfigError, match="changed since recording began"):
            run_corpus_record(Settings(), prompts_path=prompts_path, fixtures_dir=fixtures_dir)
        # Nothing was recorded as a side effect of the aborted attempt.
        assert not (fixtures_dir / "speech" / "002.wav").exists()

    def test_changing_prompt_count_after_recording_aborts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        prompts_path = _prompts_file(tmp_path, ["Hello EVA.", "Good morning."])
        fixtures_dir = tmp_path / "fixtures"
        self._run_one_prompt(monkeypatch, prompts_path, fixtures_dir)

        prompts_path.write_text("Hello EVA.\nGood morning.\nOne more prompt.\n", encoding="utf-8")

        with pytest.raises(ConfigError, match="was 2 prompts, now 3"):
            run_corpus_record(Settings(), prompts_path=prompts_path, fixtures_dir=fixtures_dir)

    def test_force_bypasses_the_abort_and_rebaselines(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        prompts_path = _prompts_file(tmp_path, ["Hello EVA.", "Good morning."])
        fixtures_dir = tmp_path / "fixtures"
        self._run_one_prompt(monkeypatch, prompts_path, fixtures_dir)

        prompts_path.write_text("Hello EVA, changed.\nGood morning.\n", encoding="utf-8")
        monkeypatch.setattr(
            "eva.audio.corpus._capture_one", lambda settings, seconds: _tone(500, 9)
        )
        _scripted_input(monkeypatch, ["", "n"])

        code = run_corpus_record(
            Settings(), prompts_path=prompts_path, fixtures_dir=fixtures_dir, force=True
        )

        assert code == 0
        manifest = _load_manifest(fixtures_dir)
        assert manifest is not None
        assert manifest.prompts_sha256 == _hash_prompts_file(prompts_path)

        # Re-baselined: an immediate follow-up run without --force no longer aborts.
        monkeypatch.setattr(
            "eva.audio.corpus._capture_one", lambda settings, seconds: _tone(500, 9)
        )
        _scripted_input(monkeypatch, ["", "n"])
        code = run_corpus_record(Settings(), prompts_path=prompts_path, fixtures_dir=fixtures_dir)
        assert code == 0

    def test_a_pre_existing_corpus_with_no_manifest_adopts_a_baseline_without_aborting(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Migration case: a corpus recorded before this check existed has no
        manifest yet. The first run against it must not abort — it has
        nothing to compare against — and existing resume behavior continues
        exactly as before."""
        prompts_path = _prompts_file(tmp_path, ["Hello EVA.", "Good morning."])
        fixtures_dir = tmp_path / "fixtures"
        (fixtures_dir / "speech").mkdir(parents=True)
        (fixtures_dir / "transcripts").mkdir(parents=True)
        (fixtures_dir / "speech" / "001.wav").write_bytes(b"pre-existing")
        (fixtures_dir / "transcripts" / "001.txt").write_text("Hello EVA.", encoding="utf-8")
        monkeypatch.setattr(
            "eva.audio.corpus._capture_one", lambda settings, seconds: _tone(500, 9)
        )
        _scripted_input(monkeypatch, ["", "n"])

        code = run_corpus_record(Settings(), prompts_path=prompts_path, fixtures_dir=fixtures_dir)

        assert code == 0
        assert (fixtures_dir / "speech" / "001.wav").read_bytes() == b"pre-existing"
        assert (fixtures_dir / "speech" / "002.wav").exists()
        assert _load_manifest(fixtures_dir) is not None

    def test_verify_prompt_integrity_directly_matching(self, tmp_path: Path) -> None:
        prompts_path = _prompts_file(tmp_path, ["Hello EVA."])
        fixtures_dir = tmp_path / "fixtures"
        fixtures_dir.mkdir()
        _verify_prompt_integrity(fixtures_dir, prompts_path, 1, force=False)  # writes baseline
        _verify_prompt_integrity(fixtures_dir, prompts_path, 1, force=False)  # no change, no raise

    def test_verify_prompt_integrity_directly_mismatched(self, tmp_path: Path) -> None:
        prompts_path = _prompts_file(tmp_path, ["Hello EVA."])
        fixtures_dir = tmp_path / "fixtures"
        fixtures_dir.mkdir()
        _verify_prompt_integrity(fixtures_dir, prompts_path, 1, force=False)
        with pytest.raises(ConfigError):
            _verify_prompt_integrity(fixtures_dir, prompts_path, 2, force=False)
