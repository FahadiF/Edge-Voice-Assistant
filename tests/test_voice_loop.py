"""CLI voice-loop tests.

1. `main_run()` Ctrl+C handling (M3 Part 8): a KeyboardInterrupt at any stage —
   model loading, audio startup, or the active conversation — must exit cleanly
   (no traceback, cleanup still runs), never just at the one place it happened
   to be caught before.
2. `ConsoleRenderer` display policy (M7.1, ADR-028): the console is a display
   surface with the same defect the web UI had — streaming tokens straight to
   stdout writes the whole reply long before the voice gets there.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from eva.config.settings import Settings
from eva.core.events import (
    Event,
    EventBus,
    FinalTranscript,
    LlmFinished,
    LlmSentence,
    LlmToken,
    StateChanged,
    TtsSentenceStarted,
    TurnFinished,
    TurnStarted,
)
from eva.memory.models import MemoryStats
from eva.metrics.turn import MetricsCollector
from eva.voice_loop import ConsoleRenderer, main_run


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

    memory_stats = MemoryStats(
        conversation_count=0,
        turn_count=0,
        embedded_turn_count=0,
        summary_count=0,
        db_size_bytes=0,
        fts_enabled=True,
    )
    assistant = SimpleNamespace(
        settings=Settings(),
        bus=EventBus(),
        llm=_StubEngine(),
        asr=_StubEngine(),
        tts=_StubEngine(),
        memory=SimpleNamespace(stats=lambda: memory_stats),
        profiles=SimpleNamespace(active=lambda: None),
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


# ── ConsoleRenderer: speech-paced vs generation-paced display (M7.1) ──

_FIRST_SPOKEN = TtsSentenceStarted(epoch=1, index=1, text="Hi there.")

# One turn: "Hi there. All good." generated, then spoken sentence by sentence.
# Note the real ordering — LlmFinished lands *before* the last sentence is
# spoken, because generation outruns synthesis. That is the whole problem.
TURN: list[Event] = [
    TurnStarted(epoch=1),
    FinalTranscript(epoch=1, text="how are you", asr_ms=120),
    LlmToken(epoch=1, token="Hi there. "),
    LlmSentence(epoch=1, text="Hi there."),
    LlmToken(epoch=1, token="All good."),
    LlmSentence(epoch=1, text="All good."),
    LlmFinished(epoch=1, text="Hi there. All good.", tokens=2, ttft_ms=300, duration_ms=1000),
    _FIRST_SPOKEN,
    TtsSentenceStarted(epoch=1, index=2, text="All good."),
    TurnFinished(epoch=1),
]


def _render(
    events: list[Event], capsys: pytest.CaptureFixture[str], *, sync_to_speech: bool
) -> str:
    renderer = ConsoleRenderer(sync_to_speech=sync_to_speech)
    for event in events:
        renderer.handle(event)
    return capsys.readouterr().out


def test_sync_mode_writes_the_reply_as_it_is_spoken(capsys: pytest.CaptureFixture[str]) -> None:
    out = _render(TURN, capsys, sync_to_speech=True)

    # The reply is written by the speech events, not by LlmFinished — so it
    # appears exactly once, in speaking order.
    assert out.count("Hi there.") == 1
    assert "Assistant: Hi there. All good." in out
    # The metrics line must not land in the middle of the spoken reply.
    assert out.index("All good.") < out.index("2 tokens")


def test_sync_mode_shows_nothing_before_the_first_sentence_is_spoken(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The gap between transcript and speech is the "thinking" beat — no part
    of the reply may appear during it, however fast generation finishes."""
    out = _render(TURN[: TURN.index(_FIRST_SPOKEN)], capsys, sync_to_speech=True)

    assert "You: how are you" in out
    assert "Hi there." not in out  # generated, not yet spoken
    assert "Assistant" not in out


def test_generation_paced_mode_keeps_the_pre_m7_behavior(
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = _render(TURN, capsys, sync_to_speech=False)

    assert "Assistant: Hi there. All good." in out
    assert out.index("Assistant:") < out.index("2 tokens")
    # Tokens streamed inline as they arrived, before the authoritative line.
    assert out.index("Hi there. ") < out.index("Assistant:")


def test_an_unspoken_reply_still_reaches_the_user(capsys: pytest.CaptureFixture[str]) -> None:
    """With no TTS audio there are no speech events at all; the reply must not
    vanish just because the display follows speech."""
    silent = [e for e in TURN if not isinstance(e, TtsSentenceStarted)]
    out = _render(silent, capsys, sync_to_speech=True)

    assert "Assistant: Hi there. All good." in out
    assert "2 tokens" in out


def test_turn_state_does_not_leak_into_the_next_turn(
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = _render([*TURN, *TURN], capsys, sync_to_speech=True)

    assert out.count("Assistant: Hi there.") == 2  # each turn renders its own
    assert out.count("2 tokens") == 2


def test_unrelated_events_render_as_before(capsys: pytest.CaptureFixture[str]) -> None:
    out = _render([StateChanged(state="listening")], capsys, sync_to_speech=True)
    assert "[listening]" in out


def test_a_failed_turn_reports_its_error(capsys: pytest.CaptureFixture[str]) -> None:
    events: list[Event] = [TurnStarted(epoch=4), TurnFinished(epoch=4, error="asr broken")]
    assert "(turn failed: asr broken)" in _render(events, capsys, sync_to_speech=True)
