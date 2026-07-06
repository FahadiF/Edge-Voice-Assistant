"""Turn orchestration tests with fake engines — no models, no audio hardware.

These cover the M2/M3-critical control flow: streaming order, cancellation on
barge-in, superseding utterances, stale-artifact suppression, and failure
containment.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Iterator

import numpy as np
import pytest

from eva.asr.base import ASREngine, TranscriptionResult
from eva.audio.frames import Frame
from eva.audio.segmenter import BargeIn, UtteranceEnd, UtteranceProgress
from eva.config.settings import Settings
from eva.conversation.orchestrator import Orchestrator
from eva.core.events import (
    BargeInLatencyMeasured,
    Event,
    EventBus,
    FinalTranscript,
    LlmFinished,
    LlmStarted,
    PartialTranscript,
    SpeechFinished,
    TurnCancelled,
    TurnFinished,
)
from eva.llm.base import ChatMessage, GenerationParams, LLMEngine
from eva.tts.base import TTSEngine
from tests.server_fakes import FakeMemoryStore

AUDIO = np.ones(16_000, dtype=np.int16)


class FakeASR(ASREngine):
    def __init__(self, text: str = "hello assistant", fail: bool = False) -> None:
        self.text = text
        self.fail = fail
        self.calls = 0

    def load(self) -> None: ...

    def unload(self) -> None: ...

    def transcribe(self, audio: Frame, language: str | None = None) -> TranscriptionResult:
        self.calls += 1
        if self.fail:
            raise RuntimeError("asr broken")
        return TranscriptionResult(text=self.text)


class FakeLLM(LLMEngine):
    def __init__(self, tokens: list[str] | None = None, delay_s: float = 0.0) -> None:
        self.tokens = tokens if tokens is not None else ["Hello ", "there. ", "All ", "good."]
        self.delay_s = delay_s
        self.aborted = False

    def load(self) -> None: ...

    def unload(self) -> None: ...

    def stream(
        self,
        messages: list[ChatMessage],
        params: GenerationParams,
        should_abort: Callable[[], bool],
    ) -> Iterator[str]:
        for token in self.tokens:
            if should_abort():
                self.aborted = True
                return
            if self.delay_s:
                time.sleep(self.delay_s)
            yield token


class FakeTTS(TTSEngine):
    def __init__(self) -> None:
        self.synthesized: list[str] = []

    def load(self) -> None: ...

    def unload(self) -> None: ...

    def synthesize(self, text: str, *, voice: str, speed: float = 1.0) -> Frame:
        self.synthesized.append(text)
        return np.ones(1600, dtype=np.int16)

    def voices(self) -> list[str]:
        return ["test-voice"]


class FakeAudioOut:
    def __init__(self) -> None:
        self.spoken: list[Frame] = []
        self.stops = 0

    def say(self, pcm: Frame) -> None:
        self.spoken.append(pcm)

    def finish_utterance(self) -> None:
        pass

    def stop_speaking(self) -> None:
        self.stops += 1

    @property
    def is_speaking(self) -> bool:
        return False  # playback drains instantly in tests


def make_orchestrator(
    asr: FakeASR | None = None,
    llm: FakeLLM | None = None,
    tts: FakeTTS | None = None,
) -> tuple[Orchestrator, EventBus, FakeAudioOut, FakeTTS]:
    settings = Settings()
    settings.conversation.system_prompt = "test"
    bus = EventBus()
    audio = FakeAudioOut()
    tts = tts or FakeTTS()
    orch = Orchestrator(
        settings, bus, audio, asr or FakeASR(), llm or FakeLLM(), tts, FakeMemoryStore()
    )
    return orch, bus, audio, tts


async def drive(
    orch: Orchestrator,
    bus: EventBus,
    script: Callable[[], asyncio.Future[None] | object] | None = None,
    *,
    timeout: float = 20.0,
) -> list[Event]:
    """Run the orchestrator until shutdown; collect all published events."""
    queue = bus.subscribe()
    events: list[Event] = []

    async def collector() -> None:
        while True:
            events.append(await queue.get())

    collect_task = asyncio.create_task(collector())
    run_task = asyncio.create_task(orch.run())
    await asyncio.sleep(0)  # let run() bind the loop
    if script is not None:
        result = script()
        if asyncio.iscoroutine(result) or isinstance(result, asyncio.Future):
            await result  # type: ignore[misc]
    orch.request_shutdown()
    await asyncio.wait_for(run_task, timeout)
    await asyncio.sleep(0.05)  # flush remaining events
    collect_task.cancel()
    return events


def names(events: list[Event]) -> list[str]:
    return [e.name for e in events]


async def wait_for_event(bus_events: list[Event], kind: type[Event], timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if any(isinstance(e, kind) for e in bus_events):
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"{kind.__name__} not observed within {timeout}s")


class TestNormalTurn:
    def test_complete_turn_event_order(self) -> None:
        async def scenario() -> None:
            orch, bus, audio, tts = make_orchestrator()

            async def script() -> None:
                orch.feed_audio_event(UtteranceEnd(AUDIO, 1000, 800, False))
                # wait for the turn to complete
                for _ in range(200):
                    if orch._turn_task is not None and orch._turn_task.done():
                        break
                    await asyncio.sleep(0.01)

            events = await drive(orch, bus, script)
            order = names(events)
            for expected in (
                "SpeechFinished",
                "TurnStarted",
                "FinalTranscript",
                "LlmStarted",
                "LlmToken",
                "LlmSentence",
                "TtsStarted",
                "TtsAudioReady",
                "LlmFinished",
                "TtsFinished",
                "TurnFinished",
            ):
                assert expected in order, f"missing {expected} in {order}"
            assert order.index("SpeechFinished") < order.index("TurnStarted")
            assert order.index("FinalTranscript") < order.index("LlmStarted")
            assert order.index("LlmStarted") < order.index("TtsStarted")
            # Audio actually reached the output.
            assert audio.spoken
            assert tts.synthesized == ["Hello there.", "All good."]
            finished = next(e for e in events if isinstance(e, LlmFinished))
            assert finished.text == "Hello there. All good."
            speech_finished = next(e for e in events if isinstance(e, SpeechFinished))
            assert speech_finished.duration_ms == 800  # speech_ms from UtteranceEnd

        asyncio.run(scenario())

    def test_markdown_stripped_for_tts_but_canonical_in_storage_and_events(self) -> None:
        """ADR-024: the TTS engine must never receive Markdown formatting,
        while memory and events keep the raw Markdown untouched."""

        async def scenario() -> None:
            llm = FakeLLM(tokens=["My name is ", "**Edge Voice Assistant**. ", "Use `eva run`."])
            orch, bus, _audio, tts = make_orchestrator(llm=llm)

            async def script() -> None:
                orch.feed_audio_event(UtteranceEnd(AUDIO, 1000, 800, False))
                for _ in range(200):
                    if orch._turn_task is not None and orch._turn_task.done():
                        break
                    await asyncio.sleep(0.01)

            events = await drive(orch, bus, script)

            # Spoken text: formatting characters gone, content intact.
            assert tts.synthesized == ["My name is Edge Voice Assistant.", "Use eva run."]

            # Event stream: raw Markdown (the web UI renders it).
            finished = next(e for e in events if isinstance(e, LlmFinished))
            assert finished.text == "My name is **Edge Voice Assistant**. Use `eva run`."

            # Storage: raw Markdown is canonical (ADR-024).
            stored = orch.memory.recent_turns(orch.conversation_id, 10)
            assistant_turns = [t for t in stored if t.speaker == "assistant"]
            assert assistant_turns
            assert assistant_turns[-1].text == (
                "My name is **Edge Voice Assistant**. Use `eva run`."
            )

        asyncio.run(scenario())

    def test_text_turn_runs_full_pipeline_without_asr(self) -> None:
        """M5.3 composer: submit_text() must produce a complete turn — LLM
        reply, TTS audio, memory write — with the typed text as the final
        transcript and no ASR involvement."""

        async def scenario() -> None:
            asr = FakeASR()
            orch, bus, audio, tts = make_orchestrator(asr=asr)

            async def script() -> None:
                assert orch.submit_text("Hello from the composer") is True
                for _ in range(200):
                    if orch._turn_task is not None and orch._turn_task.done():
                        break
                    await asyncio.sleep(0.01)

            events = await drive(orch, bus, script)
            final = next(e for e in events if isinstance(e, FinalTranscript))
            assert final.text == "Hello from the composer"
            assert final.asr_ms == 0  # ASR stage skipped
            assert asr.calls == 0  # and never invoked
            assert audio.spoken  # reply was synthesized and played
            stored = orch.memory.recent_turns(orch.conversation_id, 10)
            assert any(t.text == "Hello from the composer" for t in stored)

        asyncio.run(scenario())

    def test_submit_text_rejects_blank_and_before_run(self) -> None:
        orch, _bus, _audio, _tts = make_orchestrator()
        assert orch.submit_text("   ") is False  # blank
        assert orch.submit_text("hi") is False  # loop not bound yet (run() not started)

    def test_empty_transcript_skips_llm(self) -> None:
        async def scenario() -> None:
            orch, bus, audio, _ = make_orchestrator(asr=FakeASR(text=""))

            async def script() -> None:
                orch.feed_audio_event(UtteranceEnd(AUDIO, 1000, 800, False))
                await asyncio.sleep(0.2)

            events = await drive(orch, bus, script)
            assert "FinalTranscript" in names(events)
            assert "LlmStarted" not in names(events)
            assert "TurnFinished" in names(events)
            assert not audio.spoken

        asyncio.run(scenario())

    def test_history_carries_across_turns(self) -> None:
        async def scenario() -> None:
            orch, bus, _, _ = make_orchestrator()

            async def script() -> None:
                orch.feed_audio_event(UtteranceEnd(AUDIO, 1000, 800, False))
                await asyncio.sleep(0.3)
                orch.feed_audio_event(UtteranceEnd(AUDIO, 1000, 800, False))
                await asyncio.sleep(0.3)

            events = await drive(orch, bus, script)
            assert names(events).count("TurnFinished") == 2
            assert len(orch.conversation_turns) == 2

        asyncio.run(scenario())


class TestCancellation:
    def test_barge_in_cancels_generation(self) -> None:
        async def scenario() -> None:
            llm = FakeLLM(tokens=["tok "] * 100, delay_s=0.02)  # ~2 s generation
            orch, bus, audio, _ = make_orchestrator(llm=llm)
            collected: list[Event] = []
            q = bus.subscribe()

            async def pump() -> None:
                while True:
                    collected.append(await q.get())

            pump_task = asyncio.create_task(pump())

            async def script() -> None:
                orch.feed_audio_event(UtteranceEnd(AUDIO, 1000, 800, False))
                await wait_for_event(collected, LlmStarted)
                await asyncio.sleep(0.1)  # some tokens flow
                orch.feed_audio_event(BargeIn(speech_ms=200))
                await wait_for_event(collected, TurnCancelled)

            await drive(orch, bus, script)
            pump_task.cancel()
            assert llm.aborted
            assert audio.stops >= 1
            assert any(isinstance(e, TurnCancelled) and e.reason == "barge-in" for e in collected)

        asyncio.run(scenario())

    def test_new_utterance_supersedes_running_turn(self) -> None:
        async def scenario() -> None:
            llm = FakeLLM(tokens=["tok "] * 100, delay_s=0.02)
            orch, bus, _, _ = make_orchestrator(llm=llm)

            async def script() -> None:
                orch.feed_audio_event(UtteranceEnd(AUDIO, 1000, 800, False))
                await asyncio.sleep(0.15)
                orch.feed_audio_event(UtteranceEnd(AUDIO, 1000, 800, False))
                await asyncio.sleep(4.0)  # let the second turn finish (slow CI margin)

            events = await drive(orch, bus, script, timeout=10)
            assert any(isinstance(e, TurnCancelled) and e.reason == "superseded" for e in events)
            finished = [e for e in events if isinstance(e, TurnFinished)]
            assert len(finished) >= 1

        asyncio.run(scenario())

    def test_no_speech_after_cancellation(self) -> None:
        async def scenario() -> None:
            llm = FakeLLM(tokens=["Sentence one. "] + ["tok "] * 80, delay_s=0.02)
            orch, bus, audio, _ = make_orchestrator(llm=llm)

            async def script() -> None:
                orch.feed_audio_event(UtteranceEnd(AUDIO, 1000, 800, False))
                await asyncio.sleep(0.15)
                orch.feed_audio_event(BargeIn(speech_ms=200))
                await asyncio.sleep(0.3)
                spoken_at_cancel = len(audio.spoken)
                await asyncio.sleep(0.3)
                assert len(audio.spoken) == spoken_at_cancel  # nothing spoken after

            await drive(orch, bus, script)

        asyncio.run(scenario())

    def test_rapid_double_barge_in_stays_clean(self) -> None:
        """Two barge-ins fired back-to-back (e.g. a false start immediately
        followed by the real interruption) must not crash, must not leave a
        stale turn task running, and must settle to a clean listening state.
        `_dispatch` serializes events one at a time, so the second BargeIn is
        handled only once the first cancellation has fully settled — this
        confirms that sequence is safe even with nothing left to cancel the
        second time."""

        async def scenario() -> None:
            llm = FakeLLM(tokens=["tok "] * 100, delay_s=0.02)
            orch, bus, audio, _ = make_orchestrator(llm=llm)
            collected: list[Event] = []
            q = bus.subscribe()

            async def pump() -> None:
                while True:
                    collected.append(await q.get())

            pump_task = asyncio.create_task(pump())
            epoch_before = orch.current_epoch

            async def script() -> None:
                orch.feed_audio_event(UtteranceEnd(AUDIO, 1000, 800, False))
                await wait_for_event(collected, LlmStarted)
                orch.feed_audio_event(BargeIn(speech_ms=200))
                orch.feed_audio_event(BargeIn(speech_ms=200))
                await asyncio.sleep(0.2)

            await drive(orch, bus, script)
            pump_task.cancel()

            assert orch.current_epoch >= epoch_before + 2  # both barge-ins advanced the epoch
            assert audio.stops >= 2
            assert orch.state in ("listening", "idle")
            assert orch._turn_task is None or orch._turn_task.done()
            cancelled = [e for e in collected if isinstance(e, TurnCancelled)]
            assert any(e.reason == "barge-in" for e in cancelled)

        asyncio.run(scenario())

    def test_barge_in_publishes_latency_measurement(self) -> None:
        async def scenario() -> None:
            llm = FakeLLM(tokens=["tok "] * 100, delay_s=0.02)
            orch, bus, _, _ = make_orchestrator(llm=llm)
            collected: list[Event] = []
            q = bus.subscribe()

            async def pump() -> None:
                while True:
                    collected.append(await q.get())

            pump_task = asyncio.create_task(pump())

            async def script() -> None:
                orch.feed_audio_event(UtteranceEnd(AUDIO, 1000, 800, False))
                await wait_for_event(collected, LlmStarted)
                orch.feed_audio_event(BargeIn(speech_ms=200))
                await wait_for_event(collected, BargeInLatencyMeasured)

            await drive(orch, bus, script)
            pump_task.cancel()
            measured = next(e for e in collected if isinstance(e, BargeInLatencyMeasured))
            assert measured.detected_to_silent_ms >= 0
            assert orch.barge_in_count == 1
            assert orch.last_barge_in_latency_ms == measured.detected_to_silent_ms

        asyncio.run(scenario())

    def test_repeated_barge_ins_stay_clean(self) -> None:
        async def scenario() -> None:
            llm = FakeLLM(tokens=["tok "] * 50, delay_s=0.01)
            orch, bus, _, _ = make_orchestrator(llm=llm)

            async def script() -> None:
                for _ in range(5):
                    orch.feed_audio_event(UtteranceEnd(AUDIO, 1000, 800, False))
                    await asyncio.sleep(0.08)
                    orch.feed_audio_event(BargeIn(speech_ms=200))
                    await asyncio.sleep(0.05)
                # Final turn must still complete normally.
                orch.feed_audio_event(UtteranceEnd(AUDIO, 1000, 800, False))
                await asyncio.sleep(1.5)

            events = await drive(orch, bus, script, timeout=15)
            cancelled = [e for e in events if isinstance(e, TurnCancelled)]
            assert len(cancelled) >= 4
            finished = [e for e in events if isinstance(e, TurnFinished) and e.error is None]
            assert finished

        asyncio.run(scenario())


class TestFailures:
    def test_asr_failure_finishes_turn_with_error(self) -> None:
        async def scenario() -> None:
            orch, bus, _, _ = make_orchestrator(asr=FakeASR(fail=True))

            async def script() -> None:
                orch.feed_audio_event(UtteranceEnd(AUDIO, 1000, 800, False))
                await asyncio.sleep(0.3)

            events = await drive(orch, bus, script)
            finished = next(e for e in events if isinstance(e, TurnFinished))
            assert finished.error is not None

        asyncio.run(scenario())

    def test_llm_failure_does_not_hang(self) -> None:
        class BrokenLLM(FakeLLM):
            def stream(
                self,
                messages: list[ChatMessage],
                params: GenerationParams,
                should_abort: Callable[[], bool],
            ) -> Iterator[str]:
                raise RuntimeError("llm broken")
                yield ""  # pragma: no cover

        async def scenario() -> None:
            orch, bus, _, _ = make_orchestrator(llm=BrokenLLM())

            async def script() -> None:
                orch.feed_audio_event(UtteranceEnd(AUDIO, 1000, 800, False))
                await asyncio.sleep(0.5)

            events = await drive(orch, bus, script)
            assert "TurnFinished" in names(events)

        asyncio.run(scenario())


class TestPartials:
    def test_partial_transcript_published(self) -> None:
        async def scenario() -> None:
            orch, bus, _, _ = make_orchestrator(asr=FakeASR(text="partial words"))

            async def script() -> None:
                orch.feed_audio_event(UtteranceProgress(AUDIO, 1200))
                await asyncio.sleep(0.3)

            events = await drive(orch, bus, script)
            partials = [e for e in events if isinstance(e, PartialTranscript)]
            assert partials and partials[0].text == "partial words"

        asyncio.run(scenario())

    def test_partials_disabled_by_setting(self) -> None:
        async def scenario() -> None:
            orch, bus, _, _ = make_orchestrator()
            orch._settings.asr.partial_transcripts = False

            async def script() -> None:
                orch.feed_audio_event(UtteranceProgress(AUDIO, 1200))
                await asyncio.sleep(0.2)

            events = await drive(orch, bus, script)
            assert not any(isinstance(e, PartialTranscript) for e in events)

        asyncio.run(scenario())


class TestQueueBackpressure:
    def test_bounded_token_queue_delivers_every_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Shrink the token queue to force real wraparound; a slow-draining
        consumer (slow FakeTTS-driven speak_worker) must apply backpressure
        to the producer thread rather than lose or crash on tokens."""
        import eva.conversation.orchestrator as orch_mod

        monkeypatch.setattr(orch_mod, "_TOKEN_QUEUE_MAXSIZE", 2)
        monkeypatch.setattr(orch_mod, "_SENTENCE_QUEUE_MAXSIZE", 1)

        async def scenario() -> None:
            tokens = [f"word{i} " for i in range(40)]
            llm = FakeLLM(tokens=tokens)
            orch, bus, _, _ = make_orchestrator(llm=llm)

            async def script() -> None:
                orch.feed_audio_event(UtteranceEnd(AUDIO, 1000, 800, False))
                for _ in range(300):
                    if orch._turn_task is not None and orch._turn_task.done():
                        break
                    await asyncio.sleep(0.01)

            events = await drive(orch, bus, script)
            finished = next(e for e in events if isinstance(e, LlmFinished))
            assert finished.tokens == len(tokens)
            assert finished.text == "".join(tokens).strip()

        asyncio.run(scenario())

    def test_tight_bounds_and_short_timeout_never_crash_the_pipeline(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression guard: with a 1-slot queue and a near-zero backpressure
        timeout, the pipeline must still complete cleanly (no QueueFull
        crash, no hang) even in the tightest configuration."""
        import eva.conversation.orchestrator as orch_mod

        monkeypatch.setattr(orch_mod, "_TOKEN_QUEUE_MAXSIZE", 1)
        monkeypatch.setattr(orch_mod, "_QUEUE_BACKPRESSURE_TIMEOUT_S", 0.05)

        async def scenario() -> None:
            llm = FakeLLM(tokens=["a "] * 10)
            orch, bus, _, _ = make_orchestrator(llm=llm)

            async def script() -> None:
                orch.feed_audio_event(UtteranceEnd(AUDIO, 1000, 800, False))
                for _ in range(300):
                    if orch._turn_task is not None and orch._turn_task.done():
                        break
                    await asyncio.sleep(0.01)

            # Must not raise/hang even though some tokens may be dropped.
            events = await drive(orch, bus, script)
            assert "TurnFinished" in names(events)

        asyncio.run(scenario())


class TestMetrics:
    def test_metrics_recorded_for_completed_turn(self) -> None:
        async def scenario() -> None:
            orch, bus, _, _ = make_orchestrator()

            async def script() -> None:
                orch.feed_audio_event(UtteranceEnd(AUDIO, 1000, 800, False))
                await asyncio.sleep(0.5)

            await drive(orch, bus, script)
            turns = orch.metrics.turns
            assert len(turns) == 1
            assert turns[0].tokens == 4
            assert turns[0].ttfa_ms > 0
            assert not turns[0].cancelled

        asyncio.run(scenario())
