"""Built web UI hosting (ADR-023).

The API stays UI-agnostic (ADR-017) unless a built frontend actually exists
on disk — then `mount_ui()` serves it at `/` as an SPA: real files as-is,
everything else falls back to ``index.html`` so client-side routes
deep-link. `/api/v1/*`, `/docs`, and `/openapi.json` always win because
they are registered as routes before the catch-all.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

_ENV_VAR = "EVA_UI_DIST"


def ui_dist_dir() -> Path | None:
    """The built web UI directory, or None if no build exists.

    Resolution order:
    1. ``EVA_UI_DIST`` env var — authoritative when set: use exactly that
       directory or nothing (no fall-through — an explicit override that
       silently fell back to a different build would be a debugging trap,
       and tests rely on setting it to a bogus path to disable the UI),
    2. ``src/eva/server/static/ui/`` — where packaging copies the build,
    3. ``<repo>/web/dist/`` — a developer checkout that ran `npm run build`.

    A directory only counts if it contains an ``index.html`` — a stray
    empty directory must not swallow every GET with 404s.
    """
    override = os.environ.get(_ENV_VAR)
    if override is not None:
        path = Path(override)
        return path if (path / "index.html").is_file() else None
    candidates = [Path(__file__).parent / "static" / "ui"]
    # server/static.py -> server -> eva -> src -> repo root
    candidates.append(Path(__file__).parents[3] / "web" / "dist")
    for candidate in candidates:
        if (candidate / "index.html").is_file():
            return candidate
    return None


def mount_ui(app: FastAPI, dist: Path) -> None:
    """Serve the built SPA from `dist` at `/`.

    Registered after all API routes, so the catch-all only sees paths no
    real route claimed. Static assets (hashed js/css) live under
    ``/assets`` in a Vite build and get a dedicated mount; anything else
    is index.html — the SPA router takes it from there.
    """
    assets = dist / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="ui-assets")
    index_file = dist / "index.html"

    @app.get("/{spa_path:path}", include_in_schema=False)
    def spa(spa_path: str) -> FileResponse:
        candidate = (dist / spa_path).resolve()
        # Serve real files (favicon, manifest, ...) directly; refuse path
        # escapes; everything else is the SPA entry point.
        if spa_path and candidate.is_file() and candidate.is_relative_to(dist.resolve()):
            return FileResponse(candidate)
        return FileResponse(index_file)
