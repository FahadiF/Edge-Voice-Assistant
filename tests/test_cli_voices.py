"""CLI voice command tests (M4 integration pass) — `_load_tts_engine` is
monkeypatched to a fake engine so these tests need no real TTS model on
disk, matching the existing `tests/test_voice_registry.py` fake-engine
pattern.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from eva.audio.frames import Frame
from eva.cli import main
from eva.config import get_app_paths, load_settings
from eva.tts.base import TTSEngine


class _FakeTTS(TTSEngine):
    def __init__(self) -> None:
        self.loaded = False
        self.unloaded = False

    def load(self) -> None:
        self.loaded = True

    def unload(self) -> None:
        self.unloaded = True

    def synthesize(self, text: str, *, voice: str, speed: float = 1.0) -> Frame:
        return np.zeros(1600, dtype=np.int16)

    def voices(self) -> list[str]:
        return ["af_heart", "bm_george"]


@pytest.fixture(autouse=True)
def _fake_tts_engine(monkeypatch: pytest.MonkeyPatch) -> _FakeTTS:
    fake = _FakeTTS()

    def _load() -> tuple[_FakeTTS, object]:
        paths = get_app_paths()
        paths.ensure_exists()
        settings = load_settings(paths.settings_file)
        fake.load()
        return fake, settings

    monkeypatch.setattr("eva.cli._load_tts_engine", _load)
    return fake


def test_list_shows_voices_and_marks_active(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["voices", "list"]) == 0
    out = capsys.readouterr().out
    assert "af_heart" in out
    assert "bm_george" in out
    assert "* = active voice" in out


def test_use_persists_selection(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["voices", "use", "bm_george"]) == 0
    capsys.readouterr()
    assert main(["config", "show"]) == 0
    body = json.loads(capsys.readouterr().out)
    assert body["tts"]["voice"] == "bm_george"

    assert main(["voices", "list"]) == 0
    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if "bm_george" in line]
    assert lines and lines[0].startswith("*")


def test_preview_writes_wav_file(
    tmp_path_factory: pytest.TempPathFactory,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tempfile

    fake_dir = tmp_path_factory.mktemp("preview")
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(fake_dir))
    assert main(["voices", "preview", "af_heart"]) == 0
    out = capsys.readouterr().out
    assert "Preview written to" in out
    written = list(fake_dir.glob("*.wav"))
    assert len(written) == 1


def test_engine_loaded_and_unloaded(_fake_tts_engine: _FakeTTS) -> None:
    assert main(["voices", "list"]) == 0
    assert _fake_tts_engine.loaded is True
    assert _fake_tts_engine.unloaded is True
