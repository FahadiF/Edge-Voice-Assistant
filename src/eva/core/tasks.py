"""Owned background tasks (M5.5, ADR-026).

Every fire-and-forget asyncio task in the engine/server gets a named owner,
so shutdown is one deterministic sequence — cancel all, await all, done —
instead of scattered `asyncio.create_task` calls whose failures vanish and
whose lifetimes nobody tracks. Strong references are kept until completion
(a bare create_task result can be garbage-collected mid-flight), and
uncaught exceptions are logged with the task's name instead of dying
silently.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Coroutine
from typing import Any

logger = logging.getLogger(__name__)


class TaskManager:
    """Named, owned asyncio tasks with one-call teardown."""

    def __init__(self, owner: str) -> None:
        self._owner = owner
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        self._counter = 0

    def spawn(self, name: str, coro: Coroutine[Any, Any, Any]) -> asyncio.Task[Any]:
        """Start an owned task. A running task with the same name stays —
        the new one gets a unique suffix (names are for humans/diagnostics,
        not identity)."""
        self._counter += 1
        key = (
            name
            if name not in self._tasks or self._tasks[name].done()
            else (f"{name}-{self._counter}")
        )
        task = asyncio.create_task(coro, name=f"{self._owner}:{key}")
        self._tasks[key] = task

        def on_done(t: asyncio.Task[Any], key: str = key) -> None:
            self._on_done(key, t)

        task.add_done_callback(on_done)
        return task

    def _on_done(self, key: str, task: asyncio.Task[Any]) -> None:
        self._tasks.pop(key, None)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error("Background task %s:%s failed: %s", self._owner, key, exc)

    def active(self) -> dict[str, asyncio.Task[Any]]:
        """Live tasks by name (diagnostics)."""
        return {k: t for k, t in self._tasks.items() if not t.done()}

    def cancel(self, name: str) -> bool:
        task = self._tasks.get(name)
        if task is None or task.done():
            return False
        task.cancel()
        return True

    async def shutdown(self) -> None:
        """Cancel every owned task, await them all, swallow cancellations.
        Idempotent; never raises."""
        tasks = [t for t in self._tasks.values() if not t.done()]
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._tasks.clear()
