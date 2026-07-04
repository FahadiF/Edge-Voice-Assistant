"""Typed engine events and the event bus.

Events are the engine's public narration: every stage of a turn publishes what
it is doing, and consumers (CLI today; WebSocket clients, UI panels, and plugins
later) subscribe instead of polling. Events are immutable pydantic models so
they serialize to JSON for the API without a translation layer.

The bus is asyncio-native: each subscriber owns a bounded queue drained by its
own task. `publish()` never blocks — a slow subscriber loses oldest events
rather than stalling the pipeline (the same fresh-data-wins policy as the audio
rings). `publish_threadsafe()` lets worker/audio threads publish safely.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Literal

from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)

_SUBSCRIBER_QUEUE_SIZE = 256


class Event(BaseModel):
    model_config = ConfigDict(frozen=True)

    @property
    def name(self) -> str:
        return type(self).__name__


# ── Turn lifecycle ──


class TurnStarted(Event):
    epoch: int


class TurnFinished(Event):
    epoch: int
    error: str | None = None


class TurnCancelled(Event):
    epoch: int
    reason: Literal["barge-in", "superseded", "shutdown"]


# ── Capture / ASR ──


class SpeechStarted(Event):
    epoch: int


class SpeechFinished(Event):
    epoch: int
    duration_ms: int


class BargeInDetected(Event):
    epoch: int  # epoch of the *cancelled* turn


class PartialTranscript(Event):
    epoch: int
    text: str


class FinalTranscript(Event):
    epoch: int
    text: str
    asr_ms: int


# ── LLM ──


class LlmStarted(Event):
    epoch: int


class LlmToken(Event):
    epoch: int
    token: str


class LlmSentence(Event):
    epoch: int
    text: str


class LlmFinished(Event):
    epoch: int
    text: str
    tokens: int
    ttft_ms: int
    duration_ms: int


# ── TTS / playback ──


class TtsStarted(Event):
    epoch: int


class TtsAudioReady(Event):
    epoch: int
    ttfa_ms: int  # time from utterance end to first audio queued


class TtsFinished(Event):
    epoch: int


class StateChanged(Event):
    state: Literal["idle", "listening", "thinking", "speaking"]


class EventBus:
    """Fan-out pub/sub bound to one asyncio loop."""

    def __init__(self) -> None:
        self._subscribers: list[asyncio.Queue[Event]] = []
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Bind the loop used by `publish_threadsafe` (call once at startup)."""
        self._loop = loop

    def subscribe(self) -> asyncio.Queue[Event]:
        queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=_SUBSCRIBER_QUEUE_SIZE)
        self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[Event]) -> None:
        with contextlib.suppress(ValueError):
            self._subscribers.remove(queue)

    def publish(self, event: Event) -> None:
        """Publish from the event-loop thread. Never blocks."""
        for queue in self._subscribers:
            if queue.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    queue.get_nowait()  # drop oldest
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(event)

    def publish_threadsafe(self, event: Event) -> None:
        """Publish from any thread; no-op (with a log) before the loop is bound."""
        if self._loop is None or self._loop.is_closed():
            logger.debug("Event %s dropped: bus not bound to a loop", event.name)
            return
        self._loop.call_soon_threadsafe(self.publish, event)
