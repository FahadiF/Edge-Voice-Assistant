"""Turn epochs — the cancellation backbone (ADR-006).

A *turn* is one user-request → assistant-response cycle. The controller holds a
monotonically increasing epoch; every artifact in the pipeline is tagged with
the epoch it belongs to. Advancing the epoch (new utterance, barge-in, shutdown)
implicitly invalidates everything in flight: producers observe staleness via
`is_stale()` between chunks and abort, consumers drop stale-tagged items.

Thread safety: read/advanced from the asyncio loop, worker threads, and the
capture thread — all operations are lock-guarded and O(1).
"""

from __future__ import annotations

import threading


class TurnController:
    def __init__(self) -> None:
        self._epoch = 0
        self._lock = threading.Lock()

    @property
    def epoch(self) -> int:
        with self._lock:
            return self._epoch

    def advance(self) -> int:
        """Start a new turn; everything belonging to older epochs is now stale."""
        with self._lock:
            self._epoch += 1
            return self._epoch

    def is_stale(self, epoch: int) -> bool:
        with self._lock:
            return epoch != self._epoch

    def is_current(self, epoch: int) -> bool:
        return not self.is_stale(epoch)
