"""M3 barge-in stress tests: many rapid-fire interruptions with fake engines.

These use fake ASR/LLM/TTS (no models, no audio hardware) so they run in the
normal CI suite — unlike the manual real-hardware protocol in the team notes
(real mic, real models, a stopwatch), which validates actual audible-stop
latency and is out of scope for an automated test. What this file *can*
validate automatically: the epoch/cancellation machinery stays correct and
leak-free under adversarial timing, however many times it's hit.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from eva.audio.segmenter import BargeIn, UtteranceEnd
from eva.core.events import BargeInLatencyMeasured, TurnCancelled, TurnFinished
from tests.test_orchestrator import FakeLLM, drive, make_orchestrator

AUDIO = np.ones(16_000, dtype=np.int16)

N_INTERRUPTIONS = 20  # M3 exit criterion: 20 consecutive rapid interruptions


def test_twenty_consecutive_interruptions_leave_a_clean_state() -> None:
    async def scenario() -> None:
        llm = FakeLLM(tokens=["tok "] * 200, delay_s=0.005)
        orch, bus, audio, _ = make_orchestrator(llm=llm)
        epoch_at_start = orch.current_epoch

        async def script() -> None:
            for _ in range(N_INTERRUPTIONS):
                orch.feed_audio_event(UtteranceEnd(AUDIO, 1000, 800, False))
                await asyncio.sleep(0.02)
                orch.feed_audio_event(BargeIn(speech_ms=200))
                await asyncio.sleep(0.02)
            # One final, uninterrupted turn must still complete normally.
            orch.feed_audio_event(UtteranceEnd(AUDIO, 1000, 800, False))
            await asyncio.sleep(3.0)

        events = await drive(orch, bus, script, timeout=30)

        # Epoch strictly advanced at least once per interruption (barge-in +
        # the utterance that triggered it both advance it).
        assert orch.current_epoch >= epoch_at_start + N_INTERRUPTIONS * 2
        assert audio.stops >= N_INTERRUPTIONS
        # No orchestrator-level exception surfaced as a failed turn.
        assert not any(
            isinstance(e, TurnFinished) and e.error is not None and "asr broken" not in e.error
            for e in events
        )
        finished_clean = [e for e in events if isinstance(e, TurnFinished) and e.error is None]
        assert finished_clean, "the final, uninterrupted turn must complete"
        cancelled = [e for e in events if isinstance(e, TurnCancelled)]
        assert len(cancelled) >= N_INTERRUPTIONS
        # No background barge-in-latency task leaked past shutdown.
        assert not orch._tasks.active()  # TaskManager owns them now (M5.5)
        assert orch.state in ("listening", "idle")
        assert orch._turn_task is None or orch._turn_task.done()

    asyncio.run(scenario())


def test_barge_in_faster_than_dispatch_still_settles(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fire BargeIn events with no delay between them at all (worst case:
    the segmenter re-confirms speech every chunk during a long interruption).
    Every event is still handled one at a time by `_dispatch` — this proves
    that serialization holds under a true zero-delay burst, not just a
    20 ms-apart approximation of one."""

    async def scenario() -> None:
        llm = FakeLLM(tokens=["tok "] * 200, delay_s=0.005)
        orch, bus, audio, _ = make_orchestrator(llm=llm)

        async def script() -> None:
            orch.feed_audio_event(UtteranceEnd(AUDIO, 1000, 800, False))
            await asyncio.sleep(0.05)
            for _ in range(N_INTERRUPTIONS):
                orch.feed_audio_event(BargeIn(speech_ms=200))  # zero delay between events
            await asyncio.sleep(0.5)

        events = await drive(orch, bus, script, timeout=15)
        assert audio.stops >= 1  # at minimum, the first barge-in stopped playback
        assert not orch._tasks.active()  # TaskManager owns them now (M5.5)
        latency_events = [e for e in events if isinstance(e, BargeInLatencyMeasured)]
        assert latency_events, "at least one barge-in latency measurement must publish"
        assert all(e.detected_to_silent_ms >= 0 for e in latency_events)

    asyncio.run(scenario())


def test_repeated_stress_runs_are_independent() -> None:
    """Running the stress scenario multiple times back-to-back (fresh
    orchestrator each time) must not accumulate state across runs — guards
    against a module-level leak (e.g. a shared queue or counter)."""

    async def one_run() -> tuple[int, int]:
        llm = FakeLLM(tokens=["tok "] * 50, delay_s=0.005)
        orch, bus, _, _ = make_orchestrator(llm=llm)

        async def script() -> None:
            for _ in range(5):
                orch.feed_audio_event(UtteranceEnd(AUDIO, 1000, 800, False))
                await asyncio.sleep(0.02)
                orch.feed_audio_event(BargeIn(speech_ms=200))
                await asyncio.sleep(0.02)

        events = await drive(orch, bus, script, timeout=10)
        cancelled = [e for e in events if e.name == "TurnCancelled"]
        return orch.barge_in_count, len(cancelled)

    async def scenario() -> None:
        first = await one_run()
        second = await one_run()
        # A fresh orchestrator starts its own counters at zero each time.
        assert first[0] == second[0]
        assert first[1] == second[1]

    asyncio.run(scenario())
