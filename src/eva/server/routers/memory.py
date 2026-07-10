"""Memory management API (M4, Part 12): search, forget, pin, favorite,
archive, merge, export/import, summarize, stats, context preview. All
operations need a running engine — memory lives on the assistant's
`MemoryStore` (ADR-019).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Query

from eva.memory.models import MemorySearchResult, MemoryStats, MemorySummary
from eva.memory.summarizer import LLMSummarizer
from eva.server.deps import StateDep
from eva.server.schemas import (
    ContextPreviewResponse,
    ContextTraceResponse,
    MemorySearchRequest,
    MergeConversationsRequest,
    RenameConversationRequest,
    RetrievedMemoryTraceResponse,
)

router = APIRouter(prefix="/memory", tags=["memory"])


@router.post("/search", response_model=list[MemorySearchResult])
def search_memory(payload: MemorySearchRequest, state: StateDep) -> list[MemorySearchResult]:
    memory = state.require_assistant().memory
    return memory.search_text(
        payload.query, limit=payload.limit, conversation_id=payload.conversation_id
    )


@router.get("/stats", response_model=MemoryStats)
def memory_stats(state: StateDep) -> MemoryStats:
    return state.require_assistant().memory.stats()


@router.get("/context-preview", response_model=ContextPreviewResponse)
def preview_context(
    text: str, state: StateDep, conversation_id: str | None = Query(None)
) -> ContextPreviewResponse:
    """Show exactly what the LLM would receive for `text`, without spending
    a generation on it (ADR-021) — defaults to the active conversation."""
    orchestrator = state.require_assistant().orchestrator
    target = conversation_id or orchestrator.conversation_id
    built = orchestrator.context_builder.build(target, text)
    trace = built.trace
    return ContextPreviewResponse(
        messages=built.messages,
        trace=ContextTraceResponse(
            persona_id=trace.persona_id,
            profile_id=trace.profile_id,
            language_code=trace.language_code,
            retrieved_memories=[
                RetrievedMemoryTraceResponse(
                    turn_id=m.turn_id, score=m.score, text_preview=m.text_preview
                )
                for m in trace.retrieved_memories
            ],
            summary_included=trace.summary_included,
            summary_text_preview=trace.summary_text_preview,
            recent_turn_count=trace.recent_turn_count,
            trimmed_sections=list(trace.trimmed_sections),
        ),
    )


@router.delete("/turns/{turn_id}")
def forget_turn(turn_id: int, state: StateDep) -> dict[str, str]:
    state.require_assistant().memory.forget(turn_id)
    return {"status": "forgotten"}


@router.post("/turns/{turn_id}/pin")
def pin_turn(turn_id: int, state: StateDep, pinned: bool = Query(True)) -> dict[str, str]:
    state.require_assistant().memory.pin(turn_id, pinned=pinned)
    return {"status": "pinned" if pinned else "unpinned"}


@router.post("/turns/{turn_id}/favorite")
def favorite_turn(turn_id: int, state: StateDep, favorite: bool = Query(True)) -> dict[str, str]:
    state.require_assistant().memory.favorite(turn_id, favorite=favorite)
    return {"status": "favorited" if favorite else "unfavorited"}


@router.patch("/conversations/{conversation_id}")
def rename_conversation(
    conversation_id: str, payload: RenameConversationRequest, state: StateDep
) -> dict[str, str]:
    """Edit a conversation's title (auto-generated titles stay editable —
    M5.4 §2)."""
    state.require_assistant().memory.set_title(conversation_id, payload.title)
    return {"status": "renamed", "title": payload.title.strip()}


@router.post("/conversations/{conversation_id}/archive")
def archive_conversation(
    conversation_id: str, state: StateDep, archived: bool = Query(True)
) -> dict[str, str]:
    state.require_assistant().memory.archive_conversation(conversation_id, archived=archived)
    return {"status": "archived" if archived else "restored"}


@router.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str, state: StateDep) -> dict[str, str]:
    state.require_assistant().memory.delete_conversation(conversation_id)
    return {"status": "deleted"}


@router.post("/conversations/merge")
def merge_conversations(payload: MergeConversationsRequest, state: StateDep) -> dict[str, str]:
    state.require_assistant().memory.merge_conversations(payload.source_id, payload.target_id)
    return {"status": "merged"}


@router.post("/conversations/{conversation_id}/summarize")
def summarize_conversation(conversation_id: str, state: StateDep) -> dict[str, Any]:
    assistant = state.require_assistant()
    turns = assistant.memory.all_turns(conversation_id)
    if not turns:
        return {"status": "no_turns", "summary": None}
    summarizer = LLMSummarizer(assistant.llm)
    text = summarizer.summarize(turns)
    first_id, last_id = turns[0].id, turns[-1].id
    assert first_id is not None and last_id is not None
    saved = assistant.memory.add_summary(
        MemorySummary(
            conversation_id=conversation_id,
            turn_range_start=first_id,
            turn_range_end=last_id,
            text=text,
            created_at=datetime.now(UTC),
            model_id=assistant.settings.llm.model,
        )
    )
    return {"status": "summarized", "summary": saved.text}


@router.get("/export")
def export_memory(state: StateDep, conversation_id: str | None = Query(None)) -> dict[str, Any]:
    return state.require_assistant().memory.export_json(conversation_id)


@router.post("/import")
def import_memory(payload: dict[str, Any], state: StateDep) -> dict[str, str]:
    imported = state.require_assistant().memory.import_json(payload)
    return {"status": "imported", "turns": str(imported)}


@router.delete("")
def delete_all_memory(state: StateDep) -> dict[str, str]:
    """Privacy: wipe every conversation, turn, embedding, and summary
    (ADR-019 §10 "delete my data"). User profiles are untouched."""
    state.require_assistant().memory.delete_all()
    return {"status": "all memory deleted"}
