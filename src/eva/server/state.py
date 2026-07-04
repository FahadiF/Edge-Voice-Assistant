"""Server-owned runtime state: the engine lifecycle and shared services.

One `ServerState` per process, stored on `app.state.eva`. It owns the
`Assistant` (built lazily, only when the engine is started — opening audio
devices and loading models is an explicit action, never a side effect of the
server starting up) and the event bus every WebSocket client subscribes to.

This is the only place that touches the engine's lifecycle; every router is a
thin translation from HTTP/WebSocket to these methods — no router duplicates
engine-building logic (ADR-017 Part 10).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging

from eva.config.paths import AppPaths
from eva.config.settings import Settings, load_settings
from eva.core.errors import EvaError
from eva.core.events import (
    EngineStarted,
    EngineStopped,
    ErrorOccurred,
    EventBus,
    ModelDownloadCompleted,
    ModelDownloadFailed,
    ModelDownloadProgress,
)
from eva.engine import Assistant
from eva.models.manager import ModelManager
from eva.plugins.manager import PluginManager

logger = logging.getLogger(__name__)


class EngineNotRunningError(EvaError):
    """Raised when an operation needs a running engine but none is active."""


class ServerState:
    def __init__(self, paths: AppPaths) -> None:
        self.paths = paths
        self.settings: Settings = load_settings(paths.settings_file)
        self.bus = EventBus()
        self.model_manager = ModelManager(paths)
        self.plugin_manager = PluginManager()
        self.assistant: Assistant | None = None
        self._engine_task: asyncio.Task[None] | None = None
        self._downloads: dict[str, asyncio.Task[None]] = {}

    def reload_settings(self) -> Settings:
        """Re-read settings.json (call after any persisted change)."""
        self.settings = load_settings(self.paths.settings_file)
        return self.settings

    # ── engine lifecycle ──

    @property
    def engine_running(self) -> bool:
        return self.assistant is not None and self._engine_task is not None

    def require_assistant(self) -> Assistant:
        if self.assistant is None:
            raise EngineNotRunningError("the engine is not running — POST /api/v1/engine/start")
        return self.assistant

    async def start_engine(self) -> Assistant:
        if self.engine_running:
            assert self.assistant is not None
            return self.assistant
        from eva.engine import build_assistant

        self.reload_settings()
        assistant = build_assistant(self.settings, self.paths, bus=self.bus)
        await asyncio.to_thread(assistant.preload)
        await asyncio.to_thread(assistant.start_audio)
        self.assistant = assistant
        self._engine_task = asyncio.create_task(assistant.orchestrator.run())
        self.bus.publish(EngineStarted())
        logger.info("Engine started")
        return assistant

    async def stop_engine(self) -> None:
        if not self.engine_running:
            return
        assert self.assistant is not None
        assert self._engine_task is not None
        await asyncio.to_thread(self.assistant.stop)
        with contextlib.suppress(asyncio.CancelledError):
            await self._engine_task
        self.assistant = None
        self._engine_task = None
        self.bus.publish(EngineStopped())
        logger.info("Engine stopped")

    # ── model downloads (background, progress via the event bus) ──

    def download_active(self, model_id: str) -> bool:
        task = self._downloads.get(model_id)
        return task is not None and not task.done()

    def start_download(self, model_id: str) -> None:
        if self.download_active(model_id):
            return

        def progress(filename: str, done: int, total: int) -> None:
            self.bus.publish_threadsafe(
                ModelDownloadProgress(
                    model_id=model_id, filename=filename, bytes_done=done, bytes_total=total
                )
            )

        async def run() -> None:
            try:
                await asyncio.to_thread(self.model_manager.download, model_id, progress)
                self.bus.publish(ModelDownloadCompleted(model_id=model_id))
            except EvaError as exc:
                self.bus.publish(ModelDownloadFailed(model_id=model_id, error=str(exc)))
                self.bus.publish(ErrorOccurred(message=str(exc), context=f"download:{model_id}"))

        self._downloads[model_id] = asyncio.create_task(run())
