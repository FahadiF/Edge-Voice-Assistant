"""FastAPI application factory (ADR-017, ADR-023).

The API is versioned under `/api/v1`; the CLI (`eva serve`), the desktop
shell, the web frontend, and plugins are all just HTTP/WebSocket clients of
this one server. The only rendering concern here: when a *built* web UI
exists on disk (ADR-023 — packaged assets, a dev checkout's `web/dist`, or
an `EVA_UI_DIST` override), it is served at `/` as a static SPA. With no
build present, the app is exactly the old API-only app.
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
    memory,
    models,
    personas,
    plugins,
    settings,
    system,
    users,
    voices,
    websocket,
)
from eva.server.state import ServerState
from eva.server.static import mount_ui, ui_dist_dir

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
    for router_module in (
        system,
        settings,
        models,
        diagnostics,
        plugins,
        conversation,
        engine,
        memory,
        personas,
        users,
        voices,
    ):
        api.include_router(router_module.router)
    api.include_router(websocket.router)
    app.include_router(api)

    # Serve the built web UI when one exists (ADR-023). Mounted after every
    # API route so /api/v1/*, /docs, and /openapi.json always win.
    dist = ui_dist_dir()
    if dist is not None:
        mount_ui(app, dist)

    return app
