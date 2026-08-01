from __future__ import annotations

import asyncio
import itertools
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


class TestEventSequence:
    """`seq` is what makes a dropped event detectable: a bounded subscriber
    queue discards its oldest entry silently, so without a sequence number a
    stale UI has no way to know it missed anything (M7.3)."""

    def test_seq_increases_monotonically(self) -> None:
        async def scenario() -> None:
            bus = EventBus()
            q = bus.subscribe()
            for i in range(5):
                bus.publish(TurnStarted(epoch=i))
            seqs = [q.get_nowait().seq for _ in range(5)]
            assert seqs == sorted(seqs)
            assert len(set(seqs)) == 5
            assert all(b - a == 1 for a, b in itertools.pairwise(seqs))

        asyncio.run(scenario())

    def test_every_subscriber_sees_the_same_seq_for_one_event(self) -> None:
        """Sequence is a property of publication, not of a subscriber — two
        clients must agree on it, or a gap in one looks like a gap in both."""

        async def scenario() -> None:
            bus = EventBus()
            q1, q2 = bus.subscribe(), bus.subscribe()
            bus.publish(TurnStarted(epoch=1))
            assert q1.get_nowait().seq == q2.get_nowait().seq

        asyncio.run(scenario())

    def test_a_dropped_event_leaves_a_detectable_gap(self) -> None:
        """The whole point: overflow a subscriber, then prove the survivors'
        `seq` values are non-contiguous — which is exactly the signal the web
        client uses to decide it must resnapshot."""

        async def scenario() -> None:
            bus = EventBus()
            q = bus.subscribe()
            for i in range(300):  # queue maxsize is 256, so ~44 are dropped
                bus.publish(LlmToken(epoch=1, token=str(i)))
            received = []
            while not q.empty():
                received.append(q.get_nowait().seq)
            assert len(received) < 300, "expected the bounded queue to drop events"
            # Survivors are still ordered and contiguous among themselves...
            assert all(b - a == 1 for a, b in itertools.pairwise(received))
            # ...but the run does not start at 1, which is the observable gap.
            assert received[0] > 1

        asyncio.run(scenario())

    def test_publish_does_not_mutate_the_callers_event(self) -> None:
        """Events are frozen and `publish` stamps a copy. A future refactor
        that mutated the original instead would silently make an event object
        unsafe to publish twice or hold onto."""

        async def scenario() -> None:
            bus = EventBus()
            q = bus.subscribe()
            original = TurnStarted(epoch=1)
            bus.publish(original)
            assert original.seq == 0, "caller's instance must be untouched"
            assert q.get_nowait().seq == 1

        asyncio.run(scenario())

    def test_stream_closed_sentinel_is_not_part_of_the_sequence(self) -> None:
        """`close()` hands the singleton to subscribers directly rather than
        through `publish()`, so it keeps seq=0 and identity holds."""

        async def scenario() -> None:
            from eva.core.events import STREAM_CLOSED

            bus = EventBus()
            q = bus.subscribe()
            bus.publish(TurnStarted(epoch=1))
            q.get_nowait()
            bus.close()
            sentinel = q.get_nowait()
            assert sentinel is STREAM_CLOSED
            assert sentinel.seq == 0

        asyncio.run(scenario())
