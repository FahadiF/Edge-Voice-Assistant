"""Supervised component recovery tests (M5.5, ADR-026): a crashing
component costs one turn (or one sentence), never the assistant."""

from __future__ import annotations

import asyncio

from eva.audio.segmenter import UtteranceEnd
from eva.core.events import TurnFinished
from tests.test_orchestrator import (
    AUDIO,
    FakeASR,
    FakeTTS,
    drive,
    make_orchestrator,
)


class _FlakyASR(FakeASR):
    """Fails N times, then works — the shape of a transient driver crash."""

    def __init__(self, failures: int = 1) -> None:
        super().__init__()
        self.failures = failures
        self.unloads = 0
        self.loads = 0

    def unload(self) -> None:
        self.unloads += 1

    def load(self) -> None:
        self.loads += 1

    def transcribe(self, audio, language=None):  # type: ignore[no-untyped-def]
        if self.failures > 0:
            self.failures -= 1
            raise RuntimeError("asr driver crashed")
        return super().transcribe(audio, language)


class _FlakyTTS(FakeTTS):
    def __init__(self, failures: int = 1) -> None:
        super().__init__()
        self.failures = failures
        self.reloads = 0

    def unload(self) -> None:
        self.reloads += 1

    def synthesize(self, text: str, *, voice: str, speed: float = 1.0):  # type: ignore[no-untyped-def]
        if self.failures > 0:
            self.failures -= 1
            raise RuntimeError("tts backend died")
        return super().synthesize(text, voice=voice, speed=speed)


async def _wait_turn_done(orch) -> None:
    for _ in range(300):
        if orch._turn_task is not None and orch._turn_task.done():
            return
        await asyncio.sleep(0.01)


class TestAsrRecovery:
    def test_asr_crash_errors_one_turn_then_engine_is_reloaded_and_next_turn_works(
        self,
    ) -> None:
        async def scenario() -> None:
            asr = _FlakyASR(failures=1)
            orch, bus, audio, _tts = make_orchestrator(asr=asr)

            async def script() -> None:
                orch.feed_audio_event(UtteranceEnd(AUDIO, 1000, 800, False))
                await _wait_turn_done(orch)
                first_task = orch._turn_task
                # Give the background recovery task a moment to run.
                for _ in range(200):
                    if asr.loads >= 1:
                        break
                    await asyncio.sleep(0.01)
                orch.feed_audio_event(UtteranceEnd(AUDIO, 1000, 800, False))
                # Wait for the SECOND turn's task (a new object) to finish —
                # _wait_turn_done alone would see the old done task.
                for _ in range(300):
                    task = orch._turn_task
                    if task is not None and task is not first_task and task.done():
                        break
                    await asyncio.sleep(0.01)

            events = await drive(orch, bus, script)
            finished = [e for e in events if isinstance(e, TurnFinished)]
            assert len(finished) == 2
            assert finished[0].error is not None  # the crashing turn errored...
            assert finished[1].error is None  # ...the next one succeeded
            assert asr.unloads == 1 and asr.loads == 1  # supervised reload ran
            assert audio.spoken  # the recovered turn actually spoke

        asyncio.run(scenario())

    def test_recovery_is_cooldown_guarded(self) -> None:
        """A persistently broken component gets ONE reload per window, not a
        reload storm."""

        async def scenario() -> None:
            asr = _FlakyASR(failures=99)
            orch, bus, _audio, _tts = make_orchestrator(asr=asr)

            async def script() -> None:
                for _ in range(3):
                    orch.feed_audio_event(UtteranceEnd(AUDIO, 1000, 800, False))
                    await _wait_turn_done(orch)
                await asyncio.sleep(0.1)

            await drive(orch, bus, script)
            assert asr.loads <= 1  # cooldown blocked repeats within the window

        asyncio.run(scenario())


class TestTtsRecovery:
    def test_tts_crash_skips_the_sentence_but_completes_the_turn(self) -> None:
        async def scenario() -> None:
            tts = _FlakyTTS(failures=1)
            orch, bus, _audio, _ = make_orchestrator(tts=tts)

            async def script() -> None:
                orch.feed_audio_event(UtteranceEnd(AUDIO, 1000, 800, False))
                await _wait_turn_done(orch)

            events = await drive(orch, bus, script)
            finished = [e for e in events if isinstance(e, TurnFinished)]
            # The turn itself completed WITHOUT an error — TTS degradation
            # is not a turn failure (the reply text/storage are intact).
            assert finished and finished[0].error is None
            assert tts.reloads == 1  # supervised reload was scheduled
            # The second sentence still synthesized after the first failed.
            assert tts.synthesized  # ["All good."] — post-failure sentence

        asyncio.run(scenario())
