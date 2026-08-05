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

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from eva.audio.capture_probe import _record, _write_wav
from eva.audio.frames import SAMPLE_RATE, Frame
from eva.config.settings import Settings
from eva.core.errors import ConfigError

_FILENAME_MIN_WIDTH = 3
"""Matches the corpus's own "023.wav" convention for the common case (up to
999 prompts); widens automatically rather than truncating a larger corpus."""

_MANIFEST_FILENAME = ".eva-corpus-manifest.json"


def read_prompts(path: Path) -> list[str]:
    """One prompt per non-blank line, in file order.

    Only trailing whitespace is stripped — the result is written verbatim
    into each transcript file, so no other normalization belongs here.
    """
    if not path.exists():
        raise ConfigError(f"Prompt script not found: {path} (create it with one prompt per line)")
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


@dataclass(frozen=True)
class PromptManifest:
    """The prompt list's fingerprint as of the first recording session."""

    prompts_sha256: str
    prompt_count: int


def _hash_prompts_file(path: Path) -> str:
    """SHA-256 of the raw file bytes — deliberately the literal file, not the
    filtered prompt list, so any edit is caught, including one that happens
    not to change which lines are blank."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_path(fixtures_dir: Path) -> Path:
    return fixtures_dir / _MANIFEST_FILENAME


def _load_manifest(fixtures_dir: Path) -> PromptManifest | None:
    path = _manifest_path(fixtures_dir)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return PromptManifest(prompts_sha256=data["prompts_sha256"], prompt_count=data["prompt_count"])


def _write_manifest(fixtures_dir: Path, prompts_path: Path, prompt_count: int) -> None:
    manifest = PromptManifest(
        prompts_sha256=_hash_prompts_file(prompts_path), prompt_count=prompt_count
    )
    _manifest_path(fixtures_dir).write_text(
        json.dumps(
            {"prompts_sha256": manifest.prompts_sha256, "prompt_count": manifest.prompt_count}
        ),
        encoding="utf-8",
    )


def _verify_prompt_integrity(
    fixtures_dir: Path, prompts_path: Path, prompt_count: int, *, force: bool
) -> None:
    """Guard against `prompts.txt` changing after recording begins.

    Numbering is positional: editing, reordering, inserting, or deleting a
    prompt after some have already been recorded silently detaches an
    already-saved transcript from the text now at that line, with nothing
    else able to catch it later. The first session (no manifest yet) simply
    records today's fingerprint as the baseline — this also covers resuming
    a corpus that predates this check, with nothing to compare against yet.
    """
    existing = _load_manifest(fixtures_dir)
    if existing is None:
        _write_manifest(fixtures_dir, prompts_path, prompt_count)
        return

    current_hash = _hash_prompts_file(prompts_path)
    if existing.prompts_sha256 == current_hash and existing.prompt_count == prompt_count:
        return

    if not force:
        raise ConfigError(
            f"{prompts_path} has changed since recording began "
            f"(was {existing.prompt_count} prompts, now {prompt_count}). Editing, "
            "reordering, inserting, or deleting prompts after recording begins can "
            "corrupt the corpus — an already-recorded number would silently point "
            "at different text. Pass --force to proceed anyway (this re-baselines "
            "the manifest against the current prompt list)."
        )
    print(
        f"--force: proceeding despite a change to {prompts_path}. "
        "Re-baselining the integrity manifest against the current prompt list."
    )
    _write_manifest(fixtures_dir, prompts_path, prompt_count)


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
    force: bool = False,
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
    _verify_prompt_integrity(fixtures_dir, prompts_path, total, force=force)
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

            if (wav_path.exists() or txt_path.exists()) and index not in written_this_run:
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
            duration_s = pcm.size / SAMPLE_RATE
            print(f"Saved [{index}/{total}] {wav_path.name}  ({duration_s:.2f}s)")
            print(f"  Transcript: {prompt}")

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
