"""FastAPI application factory (ADR-017).

The API is versioned under `/api/v1` and deliberately UI-agnostic: the CLI
(`eva serve`), the future desktop shell, a future web frontend, and plugins
are all just HTTP/WebSocket clients of this one server. Nothing here renders
anything — no templates, no static files, no desktop or web UI code.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

import eva
from eva.config.paths import AppPaths, get_app_paths
from eva.server.errors import register_exception_handlers
from eva.server.routers import (
    conversation,
    diagnostics,
    engine,
    models,
    plugins,
    settings,
    system,
    websocket,
)
from eva.server.state import ServerState

API_PREFIX = "/api/v1"

# Desktop/web clients run from localhost during development; no external
# origin is ever allowed. Authentication is intentionally absent for
# localhost-only use today — see ADR-017 Part 9 for the extension point.
_LOCALHOST_ORIGIN_REGEX = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"


def create_app(paths: AppPaths | None = None) -> FastAPI:
    resolved_paths = paths or get_app_paths()
    resolved_paths.ensure_exists()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        state: ServerState = app.state.eva
        await state.stop_engine()

    app = FastAPI(
        title="Edge Voice Assistant API",
        description="Platform API for the CLI, desktop app, web UI, and plugins.",
        version=eva.__version__,
        lifespan=lifespan,
    )
    app.state.eva = ServerState(resolved_paths)

    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=_LOCALHOST_ORIGIN_REGEX,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_exception_handlers(app)

    api = APIRouter(prefix=API_PREFIX)
    for router_module in (system, settings, models, diagnostics, plugins, conversation, engine):
        api.include_router(router_module.router)
    api.include_router(websocket.router)
    app.include_router(api)

    return app
