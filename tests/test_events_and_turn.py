from __future__ import annotations

import asyncio
import threading

from eva.core.events import EventBus, LlmToken, TurnStarted
from eva.core.turn import TurnController


class TestTurnController:
    def test_epoch_advances(self) -> None:
        c = TurnController()
        e1 = c.advance()
        e2 = c.advance()
        assert e2 == e1 + 1

    def test_staleness(self) -> None:
        c = TurnController()
        epoch = c.advance()
        assert c.is_current(epoch)
        c.advance()
        assert c.is_stale(epoch)

    def test_thread_safety_of_advance(self) -> None:
        c = TurnController()
        threads = [
            threading.Thread(target=lambda: [c.advance() for _ in range(500)]) for _ in range(8)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert c.epoch == 4000


class TestEventBus:
    def test_publish_subscribe(self) -> None:
        async def scenario() -> None:
            bus = EventBus()
            q = bus.subscribe()
            bus.publish(TurnStarted(epoch=1))
            event = await asyncio.wait_for(q.get(), 1)
            assert isinstance(event, TurnStarted)
            assert event.epoch == 1

        asyncio.run(scenario())

    def test_multiple_subscribers_all_receive(self) -> None:
        async def scenario() -> None:
            bus = EventBus()
            q1, q2 = bus.subscribe(), bus.subscribe()
            bus.publish(TurnStarted(epoch=7))
            assert (await q1.get()).epoch == 7  # type: ignore[attr-defined]
            assert (await q2.get()).epoch == 7  # type: ignore[attr-defined]

        asyncio.run(scenario())

    def test_slow_subscriber_drops_oldest_not_newest(self) -> None:
        async def scenario() -> None:
            bus = EventBus()
            q = bus.subscribe()
            for i in range(300):  # queue maxsize is 256
                bus.publish(LlmToken(epoch=1, token=str(i)))
            # Oldest were dropped; the newest must be present.
            last = None
            while not q.empty():
                last = q.get_nowait()
            assert isinstance(last, LlmToken)
            assert last.token == "299"

        asyncio.run(scenario())

    def test_threadsafe_publish_from_worker(self) -> None:
        async def scenario() -> None:
            bus = EventBus()
            bus.bind_loop(asyncio.get_running_loop())
            q = bus.subscribe()
            t = threading.Thread(target=bus.publish_threadsafe, args=(TurnStarted(epoch=3),))
            t.start()
            t.join()
            event = await asyncio.wait_for(q.get(), 1)
            assert event.epoch == 3  # type: ignore[attr-defined]

        asyncio.run(scenario())

    def test_close_wakes_subscribers_with_sentinel(self) -> None:
        """M5.7: close() must unblock a consumer waiting in get() so its
        task returns before shutdown cancellation — otherwise 'Cancel N
        running task(s)' and a timeout wait."""

        async def scenario() -> None:
            from eva.core.events import STREAM_CLOSED

            bus = EventBus()
            q = bus.subscribe()
            woke: list[object] = []

            async def consumer() -> None:
                while True:
                    event = await q.get()
                    if event is STREAM_CLOSED:
                        woke.append(event)
                        return

            task = asyncio.create_task(consumer())
            await asyncio.sleep(0.01)  # consumer is now blocked in get()
            bus.close()
            await asyncio.wait_for(task, 1)  # returns promptly, not cancelled
            assert woke == [STREAM_CLOSED]
            assert bus.closed

        asyncio.run(scenario())

    def test_subscribe_after_close_is_immediately_woken(self) -> None:
        async def scenario() -> None:
            from eva.core.events import STREAM_CLOSED

            bus = EventBus()
            bus.close()
            q = bus.subscribe()  # races shutdown
            assert await asyncio.wait_for(q.get(), 1) is STREAM_CLOSED

        asyncio.run(scenario())

    def test_threadsafe_publish_without_loop_is_noop(self) -> None:
        bus = EventBus()
        bus.publish_threadsafe(TurnStarted(epoch=1))  # must not raise

    def test_unsubscribe(self) -> None:
        async def scenario() -> None:
            bus = EventBus()
            q = bus.subscribe()
            bus.unsubscribe(q)
            bus.publish(TurnStarted(epoch=1))
            assert q.empty()

        asyncio.run(scenario())
