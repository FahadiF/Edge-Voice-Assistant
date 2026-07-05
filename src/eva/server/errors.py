"""Uniform error → HTTP mapping (ADR-017 Part 11: consistent error responses).

Every error response has the same shape:
    {"detail": "<message>", "error_type": "<ExceptionClassName>"}
so clients can branch on `error_type` without parsing prose.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from eva.core.errors import (
    AudioError,
    ConfigError,
    EvaError,
    HardwareError,
    MemoryNotFoundError,
    MemoryStoreError,
    ModelError,
    ModelNotInstalledError,
    PluginError,
    RegistryError,
)
from eva.server.state import EngineNotRunningError

_STATUS_BY_ERROR: tuple[tuple[type[EvaError], int], ...] = (
    (EngineNotRunningError, 409),
    (ModelNotInstalledError, 404),
    (MemoryNotFoundError, 404),
    (RegistryError, 404),
    (PluginError, 404),
    (ConfigError, 400),
    (ModelError, 502),
    (AudioError, 503),
    (HardwareError, 503),
    (MemoryStoreError, 500),
)


def _status_for(exc: EvaError) -> int:
    for error_type, status in _STATUS_BY_ERROR:
        if isinstance(exc, error_type):
            return status
    return 500


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(EvaError)
    async def _eva_error_handler(_: Request, exc: EvaError) -> JSONResponse:
        return JSONResponse(
            status_code=_status_for(exc),
            content={"detail": str(exc), "error_type": type(exc).__name__},
        )

    @app.exception_handler(ValidationError)
    async def _pydantic_validation_handler(_: Request, exc: ValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "detail": "validation failed",
                "error_type": "ValidationError",
                "errors": [
                    {"loc": list(e["loc"]), "message": e["msg"], "type": e["type"]}
                    for e in exc.errors()
                ],
            },
        )
