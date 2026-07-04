"""Plugin lifecycle API (Part 6) — backend only; no UI, no marketplace yet."""

from __future__ import annotations

from fastapi import APIRouter

from eva.plugins.manager import PluginState
from eva.server.deps import StateDep
from eva.server.schemas import PluginStatusResponse

router = APIRouter(prefix="/plugins", tags=["plugins"])


def _to_response(state: PluginState) -> PluginStatusResponse:
    return PluginStatusResponse(
        id=state.manifest.id,
        name=state.manifest.name,
        version=state.manifest.version,
        description=state.manifest.description,
        enabled=state.enabled,
        healthy=state.healthy,
        error=state.error,
        contributes=state.manifest.contributes,
        permissions=state.manifest.permissions,
    )


@router.get("", response_model=list[PluginStatusResponse])
def list_plugins(state: StateDep) -> list[PluginStatusResponse]:
    return [_to_response(p) for p in state.plugin_manager.discover()]


@router.get("/{plugin_id}", response_model=PluginStatusResponse)
def get_plugin(plugin_id: str, state: StateDep) -> PluginStatusResponse:
    state.plugin_manager.discover()
    return _to_response(state.plugin_manager.get(plugin_id))


@router.post("/{plugin_id}/enable", response_model=PluginStatusResponse)
def enable_plugin(plugin_id: str, state: StateDep) -> PluginStatusResponse:
    return _to_response(state.plugin_manager.enable(plugin_id))


@router.post("/{plugin_id}/disable", response_model=PluginStatusResponse)
def disable_plugin(plugin_id: str, state: StateDep) -> PluginStatusResponse:
    return _to_response(state.plugin_manager.disable(plugin_id))


@router.post("/{plugin_id}/reload", response_model=PluginStatusResponse)
def reload_plugin(plugin_id: str, state: StateDep) -> PluginStatusResponse:
    return _to_response(state.plugin_manager.reload(plugin_id))
