"""Model manager API (Part 4): everything `eva models`/`eva profiles` does,
exposed for the future desktop model manager. Downloads run in the background;
progress streams over the WebSocket as `ModelDownloadProgress` events instead
of requiring the client to poll.
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Query

from eva.config.settings import save_settings
from eva.hardware.presets import CUSTOM_PROFILE_ID
from eva.server.deps import StateDep
from eva.server.schemas import DownloadStartedResponse, ModelActivateRequest

router = APIRouter(prefix="/models", tags=["models"])


@router.get("")
def list_models(
    state: StateDep,
    kind: Literal["llm", "asr", "tts", "vad"] | None = Query(None),
) -> list[dict[str, Any]]:
    manager = state.model_manager
    return [manager.describe(m.id, state.settings) for m in manager.available(kind)]


@router.get("/{model_id:path}")
def get_model(model_id: str, state: StateDep) -> dict[str, Any]:
    return state.model_manager.describe(model_id, state.settings)


@router.post("/{model_id:path}/download", response_model=DownloadStartedResponse)
def download_model(model_id: str, state: StateDep) -> DownloadStartedResponse:
    info = state.model_manager.info(model_id)
    if info.managed_by != "manager":
        return DownloadStartedResponse(model_id=model_id, status="not_applicable")
    if state.download_active(model_id):
        return DownloadStartedResponse(model_id=model_id, status="already_running")
    state.start_download(model_id)
    return DownloadStartedResponse(model_id=model_id, status="started")


@router.delete("/{model_id:path}")
def remove_model(model_id: str, state: StateDep) -> dict[str, str]:
    state.model_manager.remove(model_id)
    return {"model_id": model_id, "status": "removed"}


@router.post("/{model_id:path}/activate")
def activate_model(
    model_id: str,
    state: StateDep,
    body: ModelActivateRequest | None = None,
) -> dict[str, Any]:
    """Set a model active for its kind. Persists and switches the profile to
    'custom' — manual choices always win over presets (ADR-015)."""
    info = state.model_manager.info(model_id)
    kind = (body.kind if body else None) or info.kind
    settings = state.reload_settings()
    if kind == "llm":
        settings.llm.model = info.id
    elif kind == "asr":
        settings.asr.model = info.id
    elif kind == "tts":
        settings.tts.model = info.id
        settings.tts.engine = info.engine
    elif kind == "vad":
        settings.vad.engine = info.engine
    settings.profile = CUSTOM_PROFILE_ID
    save_settings(settings, state.paths.settings_file)
    state.settings = settings
    return state.model_manager.describe(model_id, settings)
