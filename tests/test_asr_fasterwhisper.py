"""faster-whisper adapter: offline-first load (M5.7).

Once a model is cached, huggingface_hub otherwise makes a HEAD request per
file on every load. The adapter must try a fully-offline load first and only
touch the network when the model isn't cached yet.
"""

from __future__ import annotations

import sys
from typing import Any, ClassVar

import pytest

from eva.asr.fasterwhisper import FasterWhisperASR
from eva.core.errors import ModelError


class _FakeWhisperModel:
    """Records every construction attempt; simulates a cache by failing the
    offline (`local_files_only=True`) load until `cached` is True."""

    attempts: ClassVar[list[dict[str, Any]]] = []
    cached = True
    fail_all = False

    def __init__(self, model_size_or_path: str, **kwargs: Any) -> None:
        _FakeWhisperModel.attempts.append({"model": model_size_or_path, **kwargs})
        if _FakeWhisperModel.fail_all:
            raise RuntimeError("unusable model")
        if kwargs.get("local_files_only") and not _FakeWhisperModel.cached:
            raise RuntimeError("LocalEntryNotFoundError: not cached")
        if kwargs.get("device") == "cuda":
            raise RuntimeError("CUDA runtime not available")


@pytest.fixture(autouse=True)
def _fake_whisper(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeWhisperModel.attempts = []
    _FakeWhisperModel.cached = True
    _FakeWhisperModel.fail_all = False
    fake_module = type(sys)("faster_whisper")
    fake_module.WhisperModel = _FakeWhisperModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)


def test_cached_model_loads_offline_without_network() -> None:
    _FakeWhisperModel.cached = True
    asr = FasterWhisperASR("small", device="auto")
    asr.load()
    # cuda offline fails (runtime), cpu offline succeeds — never reaches the
    # online (local_files_only=False) pass at all.
    assert all(a["local_files_only"] for a in _FakeWhisperModel.attempts)
    assert asr.device == "cpu"


def test_uncached_model_falls_back_to_network() -> None:
    _FakeWhisperModel.cached = False
    asr = FasterWhisperASR("small", device="cpu")
    asr.load()
    passes = [a["local_files_only"] for a in _FakeWhisperModel.attempts]
    assert passes == [True, False]  # offline attempted first, then online
    assert asr.device == "cpu"


def test_load_is_idempotent() -> None:
    asr = FasterWhisperASR("small", device="cpu")
    asr.load()
    n = len(_FakeWhisperModel.attempts)
    asr.load()  # already loaded — no new construction
    assert len(_FakeWhisperModel.attempts) == n


def test_total_failure_raises_modelerror() -> None:
    _FakeWhisperModel.fail_all = True  # every device+pass fails
    asr = FasterWhisperASR("small", device="cpu")
    with pytest.raises(ModelError, match="Cannot load faster-whisper"):
        asr.load()
