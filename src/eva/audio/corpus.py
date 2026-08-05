"""Audio corpus recorder (Batch 4B): the recording tool for the fixture
corpus Batch 4 (H8) deferred — a real recording session plus data-governance
sign-off, not code. Reads a prompt script and records one utterance per
prompt with the real capture pipeline, saving a WAV + transcript pair the
ASR WER harness (M8) can consume later.

This is not the benchmark, and does not measure WER, load the ASR engine, or
touch settings beyond the audio/VAD configuration `_record` already reads.
It only builds the corpus those later batches need.

Recording, VAD, segmentation, and WAV writing are not reimplemented here —
reused directly from `eva.audio.capture_probe`'s private helpers, the same
ones the M7.2 capture-test diagnostic exercises and is tested against, so a
recording made here is byte-for-byte what the live assistant's front end
would have produced for the same speech.
"""

from __future__ import annotations

from pathlib import Path

from eva.audio.capture_probe import _record, _write_wav
from eva.audio.frames import Frame
from eva.config.settings import Settings
from eva.core.errors import ConfigError

_FILENAME_MIN_WIDTH = 3
"""Matches the corpus's own "023.wav" convention for the common case (up to
999 prompts); widens automatically rather than truncating a larger corpus."""


def read_prompts(path: Path) -> list[str]:
    """One prompt per non-blank line, in file order.

    Only trailing whitespace is stripped — the result is written verbatim
    into each transcript file, so no other normalization belongs here.
    """
    if not path.exists():
        raise ConfigError(f"Prompt script not found: {path} (create it with one prompt per line)")
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def _stem_width(total: int) -> int:
    return max(_FILENAME_MIN_WIDTH, len(str(total)))


def _stem(index: int, width: int) -> str:
    return str(index).zfill(width)


def _resume_index(speech_dir: Path, transcripts_dir: Path, total: int, width: int) -> int:
    """The first prompt (1-indexed) missing either half of its pair.

    Never assumes a contiguous run of completed prompts: a WAV removed by
    hand to force a redo is picked back up at its own number rather than
    being skipped past, and nothing already complete is touched.
    """
    for index in range(1, total + 1):
        stem = _stem(index, width)
        if not (speech_dir / f"{stem}.wav").exists():
            return index
        if not (transcripts_dir / f"{stem}.txt").exists():
            return index
    return total + 1


def _capture_one(settings: Settings, seconds: float) -> Frame:
    """The processed stream `_record` produces — the same audio, endpointed
    by the same VAD + `SpeechSegmenter`, the live assistant's ASR would see
    for the same speech. The raw tap `_record` also captures is the capture
    probe's own diagnostic concern, not this tool's."""
    _raw_pcm, clean_pcm, _capture_ms, _dropped = _record(settings, seconds)
    return clean_pcm


def _ask_next_action() -> str:
    while True:
        choice = input("[N] Next   [R] Re-record   [Q] Quit  > ").strip().lower()
        if choice in ("n", ""):
            return "next"
        if choice == "r":
            return "re-record"
        if choice == "q":
            return "quit"
        print("Please answer N, R, or Q.")


def run_corpus_record(
    settings: Settings,
    *,
    prompts_path: Path,
    fixtures_dir: Path,
    seconds: float = 10.0,
) -> int:
    """Interactive prompt-by-prompt recording session. Resumes automatically;
    never overwrites a file that predates this run without asking first."""
    prompts = read_prompts(prompts_path)
    if not prompts:
        print(f"{prompts_path} has no prompts (blank lines don't count).")
        return 1

    speech_dir = fixtures_dir / "speech"
    transcripts_dir = fixtures_dir / "transcripts"
    speech_dir.mkdir(parents=True, exist_ok=True)
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    total = len(prompts)
    width = _stem_width(total)
    index = _resume_index(speech_dir, transcripts_dir, total, width)
    if index > total:
        print(f"All {total} prompts already recorded in {fixtures_dir}.")
        return 0
    if index > 1:
        print(f"Resuming at prompt {index} ({index - 1} already recorded).")

    written_this_run: set[int] = set()
    try:
        while index <= total:
            prompt = prompts[index - 1]
            stem = _stem(index, width)
            wav_path = speech_dir / f"{stem}.wav"
            txt_path = transcripts_dir / f"{stem}.txt"

            print(f"\nPrompt {index} / {total}")
            print(prompt)
            input("Press Enter to record...")

            print("Recording — stops once you finish speaking (Ctrl+C to stop early).")
            pcm = _capture_one(settings, seconds)
            if pcm.size == 0:
                print("No audio captured — is the microphone available and unmuted? Try again.")
                continue

            if wav_path.exists() and index not in written_this_run:
                answer = (
                    input(f"'{stem}' already has a recording. Overwrite? [y/N] ").strip().lower()
                )
                if answer != "y":
                    print(f"Keeping the existing recording for '{stem}'.")
                    index += 1
                    continue

            _write_wav(wav_path, pcm)
            txt_path.write_text(prompt, encoding="utf-8")
            written_this_run.add(index)
            print(f"Saved {wav_path} and {txt_path}")

            action = _ask_next_action()
            if action == "quit":
                print(f"Stopped at prompt {index} / {total}. Run again any time to resume.")
                return 0
            if action == "next":
                index += 1
            # "re-record": loop again with the same index.
    except KeyboardInterrupt:
        print(f"\nInterrupted at prompt {index} / {total}. Run again any time to resume.")
        return 1

    print(f"Done — all {total} prompts recorded in {fixtures_dir}.")
    return 0
