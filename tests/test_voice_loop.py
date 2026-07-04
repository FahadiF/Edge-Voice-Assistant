"""main_run() Ctrl+C handling (M3 Part 8): a KeyboardInterrupt at any stage —
model loading, audio startup, or the active conversation — must exit cleanly
(no traceback, cleanup still runs), never just at the one place it happened
to be caught before.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from eva.config.settings import Settings
from eva.core.events import EventBus
from eva.metrics.turn import MetricsCollector
from eva.voice_loop import main_run


class _StubEngine:
    device = "cpu"

    def load(self) -> None:
        pass


async def _noop_run() -> None:
    return None


def _make_assistant(**overrides: object) -> SimpleNamespace:
    metrics = MetricsCollector()
    stopped = {"called": False}

    def stop() -> None:
        stopped["called"] = True

    assistant = SimpleNamespace(
        settings=Settings(),
        bus=EventBus(),
        llm=_StubEngine(),
        asr=_StubEngine(),
        tts=_StubEngine(),
        preload=lambda: None,
        start_audio=lambda: None,
        stop=stop,
        orchestrator=SimpleNamespace(metrics=metrics, run=_noop_run),
        _stopped=stopped,
    )
    for key, value in overrides.items():
        setattr(assistant, key, value)
    return assistant


def test_interrupt_during_preload_exits_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    def boom() -> None:
        raise KeyboardInterrupt

    assistant = _make_assistant(preload=boom)
    assert main_run(assistant) == 0
    assert assistant._stopped["called"]
    assert "Stopping" in capsys.readouterr().out


def test_interrupt_during_start_audio_exits_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    def boom() -> None:
        raise KeyboardInterrupt

    assistant = _make_assistant(start_audio=boom)
    assert main_run(assistant) == 0
    assert assistant._stopped["called"]
    assert "Stopping" in capsys.readouterr().out


def test_interrupt_during_voice_loop_exits_cleanly(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def boom(_assistant: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr("eva.voice_loop.run_voice_loop", boom)
    assistant = _make_assistant()
    assert main_run(assistant) == 0
    assert assistant._stopped["called"]


def test_summary_suppressed_when_no_turns_completed(capsys: pytest.CaptureFixture[str]) -> None:
    def boom() -> None:
        raise KeyboardInterrupt

    assistant = _make_assistant(preload=boom)
    main_run(assistant)
    assert "No completed turns" not in capsys.readouterr().out


def test_summary_shown_when_turns_completed(capsys: pytest.CaptureFixture[str]) -> None:
    from eva.metrics.turn import TurnMetrics

    assistant = _make_assistant()
    assistant.orchestrator.metrics.record(TurnMetrics(epoch=1, ttfa_ms=500, total_ms=1000))
    main_run(assistant)
    out = capsys.readouterr().out
    assert "median" in out.lower() or "ttfa" in out.lower() or "turns" in out.lower()
