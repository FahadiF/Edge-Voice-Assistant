"""Turn orchestrator: segmenter events in, spoken responses out.

The asyncio composition of the whole pipeline (ADR-006):

  capture thread ──feed_audio_event()──► event queue ──► run() loop
      UtteranceEnd  → start a turn task (ASR → LLM tokens → sentences → TTS)
      BargeIn       → advance the epoch, stop playback, cancel the turn task
      UtteranceProgress → opportunistic partial transcript

Inside a turn, three concurrent units pipeline the response:
  producer thread   llm.stream() pushes tokens into an asyncio queue,
                    aborting via epoch staleness (per-token check);
  token consumer    publishes LlmToken events and feeds the sentence chunker;
  speak worker      synthesizes sentences sequentially and queues playback —
                    sentence N plays while N+1 synthesizes while tokens arrive.

Every hand-off point checks the turn epoch; a stale turn stops at the next
boundary and its artifacts are never spoken. The orchestrator holds no model
code: engines are ports, audio out is a three-method protocol, so the entire
control flow is unit-testable with fakes.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncGenerator, Iterator
from typing import Protocol

from eva.asr.base import ASREngine
from eva.audio.frames import Frame
from eva.audio.segmenter import (
    BargeIn,
    SegmenterEvent,
    SpeechStart,
    UtteranceDiscarded,
    UtteranceEnd,
    UtteranceProgress,
)
from eva.config.settings import Settings
from eva.conversation.chunker import SentenceChunker
from eva.conversation.context_builder import ContextBuilder
from eva.conversation.history import ConversationTurn, pair_turns
from eva.conversation.language import effective_asr_language, effective_voice, resolve_language
from eva.conversation.markdown import MarkdownSpeechFilter
from eva.core.events import (
    BargeInDetected,
    BargeInLatencyMeasured,
    EventBus,
    FinalTranscript,
    LlmFinished,
    LlmSentence,
    LlmStarted,
    LlmToken,
    PartialTranscript,
    SpeechFinished,
    SpeechStarted,
    StateChanged,
    TtsAudioReady,
    TtsFinished,
    TtsStarted,
    TurnCancelled,
    TurnFinished,
    TurnStarted,
)
from eva.core.turn import TurnController
from eva.llm.base import GenerationParams, LLMEngine
from eva.memory.base import MemoryStore
from eva.metrics.turn import MetricsCollector, TurnMetrics
from eva.tts.base import TTSEngine

logger = logging.getLogger(__name__)

_PLAYBACK_POLL_S = 0.05
_BARGE_IN_POLL_S = 0.005  # fine-grained: the target itself is < 150 ms
_BARGE_IN_LATENCY_TIMEOUT_S = 1.0  # safety cap; a hit means "unmeasured", not "slow"
_TOKEN_QUEUE_MAXSIZE = 256  # bounds memory on a pathological long reply
_SENTENCE_QUEUE_MAXSIZE = 32
_QUEUE_BACKPRESSURE_TIMEOUT_S = 5.0  # cross-thread put() safety cap; never hit in practice


class AudioOutput(Protocol):
    """The only audio surface the orchestrator needs."""

    def say(self, pcm: Frame) -> None: ...

    def finish_utterance(self) -> None: ...

    def stop_speaking(self) -> None: ...

    @property
    def is_speaking(self) -> bool: ...


def _pull_or_none(it: Iterator[Frame]) -> Frame | None:
    try:
        return next(it)
    except StopIteration:
        return None


def _close_iter(it: Iterator[Frame]) -> None:
    close = getattr(it, "close", None)
    if close is not None:
        close()


async def _drive_stream(sync_iter: Iterator[Frame]) -> AsyncGenerator[Frame, None]:
    """Advance a blocking generator one item at a time without blocking the loop.

    `next()` runs in the default executor per item, so a slow chunk (TTS
    synthesis) never stalls the event loop. The underlying generator is always
    closed on exit — normal exhaustion, an exception, or the caller stopping
    early (e.g. on barge-in) — so an engine's cleanup (ADR-018: KokoroTTS
    closes its dedicated event loop) always runs promptly.
    """
    loop = asyncio.get_running_loop()
    try:
        while True:
            chunk = await loop.run_in_executor(None, _pull_or_none, sync_iter)
            if chunk is None:
                return
            yield chunk
    finally:
        await loop.run_in_executor(None, _close_iter, sync_iter)


class Orchestrator:
    def __init__(
        self,
        settings: Settings,
        bus: EventBus,
        audio_out: AudioOutput,
        asr: ASREngine,
        llm: LLMEngine,
        tts: TTSEngine,
        memory: MemoryStore,
        *,
        context_builder: ContextBuilder | None = None,
        metrics: MetricsCollector | None = None,
    ) -> None:
        self._settings = settings
        self._bus = bus
        self._audio = audio_out
        self._asr = asr
        self._llm = llm
        self._tts = tts
        self._memory = memory
        self._metrics = metrics or MetricsCollector()
        self._controller = TurnController()
        language = resolve_language(settings)
        self._asr_language = effective_asr_language(settings, language)
        self._voice = effective_voice(settings, language)
        self._language_code = language.code
        self._context_builder = context_builder or ContextBuilder(settings, memory)
        self._conversation = memory.start_conversation(language=language.code)
        self._params = GenerationParams(
            temperature=settings.conversation.temperature,
            top_p=settings.conversation.top_p,
            max_tokens=settings.conversation.max_tokens,
            stop=tuple(settings.conversation.stop_sequences),
        )
        self._loop: asyncio.AbstractEventLoop | None = None
        self._state: str = "idle"
        self._events: asyncio.Queue[SegmenterEvent | None] = asyncio.Queue()
        self._turn_task: asyncio.Task[None] | None = None
        self._partial_task: asyncio.Task[None] | None = None
        self._partial_busy = False
        self._barge_in_count = 0
        self._last_barge_in_latency_ms: int | None = None
        self._background_tasks: set[asyncio.Task[None]] = set()
        self._current_tokens_queue: asyncio.Queue[str | None] | None = None
        self._current_sentences_queue: asyncio.Queue[str | None] | None = None

    @property
    def metrics(self) -> MetricsCollector:
        return self._metrics

    @property
    def state(self) -> str:
        """Current pipeline state: idle / listening / thinking / speaking."""
        return self._state

    @property
    def pending_audio_events(self) -> int:
        """Depth of the capture→orchestrator event queue (diagnostics)."""
        return self._events.qsize()

    @property
    def current_epoch(self) -> int:
        return self._controller.epoch

    @property
    def barge_in_count(self) -> int:
        """Number of barge-ins handled this session (diagnostics)."""
        return self._barge_in_count

    @property
    def last_barge_in_latency_ms(self) -> int | None:
        """Most recent detected-to-silent barge-in latency (diagnostics)."""
        return self._last_barge_in_latency_ms

    @property
    def token_queue_depth(self) -> int:
        """Pending LLM tokens not yet consumed by the chunker (diagnostics)."""
        return self._current_tokens_queue.qsize() if self._current_tokens_queue else 0

    @property
    def sentence_queue_depth(self) -> int:
        """Pending sentences not yet picked up by the speak worker (diagnostics)."""
        return self._current_sentences_queue.qsize() if self._current_sentences_queue else 0

    @property
    def memory(self) -> MemoryStore:
        """The active MemoryStore (diagnostics, management API)."""
        return self._memory

    @property
    def context_builder(self) -> ContextBuilder:
        """The active ContextBuilder (platform API context-preview endpoint)."""
        return self._context_builder

    @property
    def last_retrieval_ms(self) -> int | None:
        """Most recent semantic-memory retrieval latency (diagnostics)."""
        return self._context_builder.last_retrieval_ms

    @property
    def last_retrieval_score_top1(self) -> float | None:
        """Top result's score from the most recent retrieval (diagnostics)."""
        return self._context_builder.last_retrieval_top_score

    @property
    def conversation_id(self) -> str:
        """The active conversation id (M4) — memories persist under this id
        across the whole engine run; `clear_conversation()` starts a new one."""
        return self._conversation.id

    @property
    def conversation_turns(self) -> list[ConversationTurn]:
        """Paired (user, assistant) view of the active conversation — the
        pre-M4 `/conversation/history|export` API contract, unchanged in
        shape even though storage moved to `MemoryStore` (ADR-019)."""
        return pair_turns(self._memory.all_turns(self._conversation.id))

    def clear_conversation(self) -> None:
        """Start a fresh conversation. Does not delete the old one's data —
        it remains searchable/exportable; this matches the pre-M4 behavior
        of resetting the *active* session, now that conversations persist."""
        self._conversation = self._memory.start_conversation(language=self._language_code)

    def load_conversation_turns(self, turns: list[ConversationTurn]) -> None:
        """Import paired turns into the active conversation (platform API
        import)."""
        for turn in turns:
            self._memory.add_turn(
                self._conversation.id, "user", turn.user, language=self._language_code
            )
            self._memory.add_turn(
                self._conversation.id, "assistant", turn.assistant, language=self._language_code
            )

    def _set_state(self, state: str) -> None:
        self._state = state
        self._bus.publish(StateChanged(state=state))

    # ── external control (platform API — same event loop as run()) ──

    async def interrupt(self) -> bool:
        """Stop the current turn immediately (API equivalent of voice barge-in).

        Used for both the API's "interrupt" and "cancel" actions — there is
        only one way to stop a turn in this FSM; both names describe the same
        operation from the caller's perspective. Returns False if nothing was
        in flight.
        """
        if self._turn_task is None or self._turn_task.done():
            return False
        await self._cancel_turn("manual")
        return True

    # ── input bridge (called from the capture thread) ──

    def feed_audio_event(self, event: SegmenterEvent) -> None:
        if self._loop is None or self._loop.is_closed():
            return
        with contextlib.suppress(RuntimeError):
            self._loop.call_soon_threadsafe(self._events.put_nowait, event)

    def request_shutdown(self) -> None:
        if self._loop is not None and not self._loop.is_closed():
            with contextlib.suppress(RuntimeError):
                self._loop.call_soon_threadsafe(self._events.put_nowait, None)

    # ── main loop ──

    async def run(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._bus.bind_loop(self._loop)
        self._set_state("listening")
        try:
            while True:
                event = await self._events.get()
                if event is None:
                    break
                await self._dispatch(event)
        finally:
            await self._cancel_turn("shutdown")
            await self._drain_background_tasks()
            self._set_state("idle")

    async def _drain_background_tasks(self) -> None:
        """Cancel any still-running fire-and-forget tasks (e.g. a barge-in
        latency measurement in flight) so shutdown never leaves orphans."""
        tasks = list(self._background_tasks)
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def _dispatch(self, event: SegmenterEvent) -> None:
        match event:
            case SpeechStart():
                self._bus.publish(SpeechStarted(epoch=self._controller.epoch))
            case BargeIn():
                self._bus.publish(BargeInDetected(epoch=self._controller.epoch))
                await self._cancel_turn("barge-in")
            case UtteranceEnd(audio=audio, speech_ms=speech_ms):
                self._bus.publish(
                    SpeechFinished(epoch=self._controller.epoch, duration_ms=speech_ms)
                )
                # A new utterance always supersedes whatever is in flight
                # (covers half-duplex mode, where no BargeIn event exists).
                if self._turn_task is not None and not self._turn_task.done():
                    await self._cancel_turn("superseded")
                self._turn_task = asyncio.create_task(self._run_turn(audio))
            case UtteranceProgress(audio=audio):
                if self._settings.asr.partial_transcripts and not self._partial_busy:
                    self._partial_task = asyncio.create_task(self._partial_transcribe(audio))
            case UtteranceDiscarded():
                logger.debug("Utterance discarded by noise gate")

    async def _cancel_turn(self, reason: str) -> None:
        stale_epoch = self._controller.epoch
        self._controller.advance()  # all in-flight work is now stale
        cancel_started = time.perf_counter()
        self._audio.stop_speaking()
        task, self._turn_task = self._turn_task, None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            self._bus.publish(TurnCancelled(epoch=stale_epoch, reason=reason))
            self._set_state("listening")
        if reason == "barge-in":
            # Fire-and-forget: measuring silence must never delay processing
            # the next event (e.g. a second rapid barge-in). A strong
            # reference is kept until completion so the task cannot be
            # garbage-collected mid-flight.
            measure_task = asyncio.create_task(
                self._measure_barge_in_latency(stale_epoch, cancel_started)
            )
            self._background_tasks.add(measure_task)
            measure_task.add_done_callback(self._background_tasks.discard)

    async def _measure_barge_in_latency(self, epoch: int, started: float) -> None:
        deadline = started + _BARGE_IN_LATENCY_TIMEOUT_S
        while self._audio.is_speaking and time.perf_counter() < deadline:
            await asyncio.sleep(_BARGE_IN_POLL_S)
        latency_ms = int((time.perf_counter() - started) * 1000)
        self._barge_in_count += 1
        self._last_barge_in_latency_ms = latency_ms
        self._bus.publish(BargeInLatencyMeasured(epoch=epoch, detected_to_silent_ms=latency_ms))

    # ── partial transcription (best-effort, never blocks the pipeline) ──

    async def _partial_transcribe(self, audio: Frame) -> None:
        self._partial_busy = True
        try:
            epoch = self._controller.epoch
            result = await asyncio.to_thread(self._asr.transcribe, audio, self._asr_language)
            if result.text and self._controller.is_current(epoch):
                self._bus.publish(PartialTranscript(epoch=epoch, text=result.text))
        except Exception:
            logger.exception("Partial transcription failed")
        finally:
            self._partial_busy = False

    # ── the turn pipeline ──

    async def _run_turn(self, audio: Frame) -> None:
        epoch = self._controller.advance()
        t0 = time.perf_counter()
        self._bus.publish(TurnStarted(epoch=epoch))
        self._set_state("thinking")
        error: str | None = None
        metrics = TurnMetrics(epoch=epoch)
        try:
            metrics = await self._pipeline(epoch, audio, t0)
        except asyncio.CancelledError:
            self._metrics.record(metrics.model_copy(update={"cancelled": True}))
            raise
        except Exception as exc:
            logger.exception("Turn %d failed", epoch)
            error = str(exc)
        self._metrics.record(metrics)
        self._bus.publish(TurnFinished(epoch=epoch, error=error))
        if self._controller.is_current(epoch):
            self._set_state("listening")

    async def _pipeline(self, epoch: int, audio: Frame, t0: float) -> TurnMetrics:
        def elapsed_ms() -> int:
            return int((time.perf_counter() - t0) * 1000)

        self._current_tokens_queue = None
        self._current_sentences_queue = None

        # ── ASR ──
        result = await asyncio.to_thread(self._asr.transcribe, audio, self._asr_language)
        asr_ms = elapsed_ms()
        user_text = result.text.strip()
        if self._controller.is_stale(epoch):
            return TurnMetrics(epoch=epoch, asr_ms=asr_ms, cancelled=True)
        self._bus.publish(FinalTranscript(epoch=epoch, text=user_text, asr_ms=asr_ms))
        if not user_text:
            return TurnMetrics(epoch=epoch, asr_ms=asr_ms)

        # ── LLM producer thread → token queue ──
        self._bus.publish(LlmStarted(epoch=epoch))
        loop = asyncio.get_running_loop()
        tokens: asyncio.Queue[str | None] = asyncio.Queue(maxsize=_TOKEN_QUEUE_MAXSIZE)
        self._current_tokens_queue = tokens
        # Context building can embed the query + hit SQLite (ADR-021); run it
        # off the event loop like every other blocking pipeline stage.
        built_context = await asyncio.to_thread(
            self._context_builder.build, self._conversation.id, user_text
        )
        messages = built_context.messages

        def produce() -> None:
            def push(item: str | None) -> None:
                # A blocking put (not put_nowait) applies real backpressure to
                # the producer thread when the bounded queue is full, instead
                # of raising QueueFull; a timeout guards against hanging this
                # thread forever if the loop stops draining (e.g. shutdown).
                with contextlib.suppress(RuntimeError):
                    future = asyncio.run_coroutine_threadsafe(tokens.put(item), loop)
                    try:
                        future.result(timeout=_QUEUE_BACKPRESSURE_TIMEOUT_S)
                    except TimeoutError:
                        logger.warning("Token queue backpressure timeout; dropping token")

            try:
                for token in self._llm.stream(
                    messages, self._params, should_abort=lambda: self._controller.is_stale(epoch)
                ):
                    push(token)
            except Exception:
                logger.exception("LLM generation failed")
            finally:
                push(None)

        producer = asyncio.create_task(asyncio.to_thread(produce))

        # ── speak worker: sentences → TTS → playback ──
        sentences: asyncio.Queue[str | None] = asyncio.Queue(maxsize=_SENTENCE_QUEUE_MAXSIZE)
        self._current_sentences_queue = sentences
        first_audio_ms = 0
        tts_first_ms = 0

        async def speak_worker() -> None:
            nonlocal first_audio_ms, tts_first_ms
            started = False
            # One filter per turn: fence state must survive across sentence
            # segments (a code block's ``` markers arrive in different
            # segments — ADR-024). Storage/events keep the raw Markdown;
            # only what reaches the TTS engine is converted.
            speech_filter = MarkdownSpeechFilter()
            while True:
                sentence = await sentences.get()
                if sentence is None:
                    break
                if self._controller.is_stale(epoch):
                    continue  # drain without speaking
                spoken_text = speech_filter.convert(sentence)
                if not spoken_text:
                    continue  # e.g. a segment entirely inside a code fence
                if not started:
                    started = True
                    self._bus.publish(TtsStarted(epoch=epoch))
                    self._set_state("speaking")
                synth_start = time.perf_counter()
                sync_gen = self._tts.synthesize_stream(
                    spoken_text, voice=self._voice, speed=self._settings.tts.speed
                )
                try:
                    async with contextlib.aclosing(_drive_stream(sync_gen)) as chunks:
                        async for chunk in chunks:
                            if self._controller.is_stale(epoch):
                                break  # aclosing() shuts down _drive_stream, which
                                # closes sync_gen (KokoroTTS: its event loop too)
                            if first_audio_ms == 0:
                                tts_first_ms = int((time.perf_counter() - synth_start) * 1000)
                                first_audio_ms = elapsed_ms()
                                self._bus.publish(
                                    TtsAudioReady(epoch=epoch, ttfa_ms=first_audio_ms)
                                )
                            self._audio.say(chunk)
                finally:
                    self._audio.finish_utterance()

        speaker = asyncio.create_task(speak_worker())

        # ── token consumer ──
        chunker = SentenceChunker(
            min_chars=self._settings.conversation.sentence_min_chars,
            max_chars=self._settings.conversation.sentence_max_chars,
            first_chunk_min_chars=self._settings.conversation.first_sentence_min_chars,
        )
        reply_parts: list[str] = []
        token_count = 0
        ttft_ms = 0
        llm_start = time.perf_counter()
        try:
            while True:
                token = await tokens.get()
                if token is None:
                    break
                if self._controller.is_stale(epoch):
                    continue  # keep draining so the producer can finish
                if ttft_ms == 0:
                    ttft_ms = elapsed_ms()
                token_count += 1
                reply_parts.append(token)
                self._bus.publish(LlmToken(epoch=epoch, token=token))
                for sentence in chunker.feed(token):
                    self._bus.publish(LlmSentence(epoch=epoch, text=sentence))
                    await sentences.put(sentence)
            tail = chunker.flush()
            if tail is not None and self._controller.is_current(epoch):
                self._bus.publish(LlmSentence(epoch=epoch, text=tail))
                await sentences.put(tail)
        finally:
            await sentences.put(None)
            await asyncio.gather(producer, speaker)
            self._current_tokens_queue = None
            self._current_sentences_queue = None

        llm_ms = int((time.perf_counter() - llm_start) * 1000)
        reply = "".join(reply_parts).strip()
        if self._controller.is_stale(epoch):
            return TurnMetrics(
                epoch=epoch, asr_ms=asr_ms, ttft_ms=ttft_ms, tokens=token_count, cancelled=True
            )
        self._bus.publish(
            LlmFinished(
                epoch=epoch, text=reply, tokens=token_count, ttft_ms=ttft_ms, duration_ms=llm_ms
            )
        )
        if reply:
            await asyncio.to_thread(
                self._memory.add_turn,
                self._conversation.id,
                "user",
                user_text,
                language=self._language_code,
            )
            await asyncio.to_thread(
                self._memory.add_turn,
                self._conversation.id,
                "assistant",
                reply,
                language=self._language_code,
            )

        # ── wait for playback to drain (stays interruptible) ──
        while self._audio.is_speaking and self._controller.is_current(epoch):
            await asyncio.sleep(_PLAYBACK_POLL_S)
        self._bus.publish(TtsFinished(epoch=epoch))

        return TurnMetrics(
            epoch=epoch,
            asr_ms=asr_ms,
            ttft_ms=ttft_ms,
            llm_ms=llm_ms,
            tokens=token_count,
            tts_first_ms=tts_first_ms,
            ttfa_ms=first_audio_ms,
            total_ms=elapsed_ms(),
        )
