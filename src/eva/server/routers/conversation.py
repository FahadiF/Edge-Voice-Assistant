"""Conversation API (Part 7): history, current turn, interrupt/cancel,
clear, export/import. All operations need a running engine — turns persist
in the assistant's `MemoryStore` (ADR-019); this router presents the same
paired-turn shape the API contract has always used.
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter

from eva.conversation.history import ConversationTurn
from eva.server.deps import StateDep
from eva.server.schemas import (
    CONVERSATION_EXPORT_VERSION,
    ConversationExport,
    ConversationImportRequest,
    EngineStatusResponse,
    InterruptResponse,
)

router = APIRouter(prefix="/conversation", tags=["conversation"])


@router.get("/history", response_model=list[ConversationTurn])
def get_history(state: StateDep) -> list[ConversationTurn]:
    return state.require_assistant().orchestrator.conversation_turns


@router.get("/current", response_model=EngineStatusResponse)
def get_current_turn(state: StateDep) -> EngineStatusResponse:
    orchestrator = state.require_assistant().orchestrator
    return EngineStatusResponse(running=True, state=orchestrator.state)


@router.post("/interrupt", response_model=InterruptResponse)
async def interrupt(state: StateDep) -> InterruptResponse:
    happened = await state.require_assistant().orchestrator.interrupt()
    return InterruptResponse(interrupted=happened)


@router.post("/cancel", response_model=InterruptResponse)
async def cancel(state: StateDep) -> InterruptResponse:
    """Alias of /interrupt: the turn FSM has one way to stop a turn."""
    happened = await state.require_assistant().orchestrator.interrupt()
    return InterruptResponse(interrupted=happened)


@router.post("/clear")
def clear_history(state: StateDep) -> dict[str, str]:
    state.require_assistant().orchestrator.clear_conversation()
    return {"status": "cleared"}


@router.get("/export", response_model=ConversationExport)
def export_history(state: StateDep) -> ConversationExport:
    assistant = state.require_assistant()
    return ConversationExport(
        version=CONVERSATION_EXPORT_VERSION,
        exported_at=datetime.now(UTC),
        profile=assistant.settings.profile,
        language=assistant.settings.conversation.language,
        turns=assistant.orchestrator.conversation_turns,
    )


@router.post("/import")
def import_history(payload: ConversationImportRequest, state: StateDep) -> dict[str, str]:
    state.require_assistant().orchestrator.load_conversation_turns(payload.turns)
    return {"status": "imported", "turns": str(len(payload.turns))}
