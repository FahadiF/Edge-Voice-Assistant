"""Privacy/retention policy application (ADR-019 Part 10).

A pure function operating through the `MemoryStore` port — not a registry
(there is exactly one retention algorithm, driven entirely by settings; see
ADR-019's rationale for consolidating `MemoryCleaner` into an operation
rather than a swappable interface). Pinned turns are always exempt from
both rules: that exemption is what "pin" means (`MemoryStore.pin`'s
docstring).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from eva.config.settings import MemorySettings
from eva.memory.base import MemoryStore


@dataclass(frozen=True)
class RetentionReport:
    turns_deleted_by_age: int
    turns_deleted_by_cap: int

    @property
    def total_deleted(self) -> int:
        return self.turns_deleted_by_age + self.turns_deleted_by_cap


def apply_retention_policy(store: MemoryStore, settings: MemorySettings) -> RetentionReport:
    """Delete turns older than `settings.retention_days`, then cap each
    conversation at `settings.max_turns_per_conversation` — each rule is a
    no-op if its setting is None. Both rules skip pinned turns."""
    deleted_by_age = _apply_age_limit(store, settings.retention_days)
    deleted_by_cap = _apply_turn_cap(store, settings.max_turns_per_conversation)
    return RetentionReport(turns_deleted_by_age=deleted_by_age, turns_deleted_by_cap=deleted_by_cap)


def _apply_age_limit(store: MemoryStore, retention_days: int | None) -> int:
    if retention_days is None:
        return 0
    cutoff = datetime.now(UTC) - timedelta(days=retention_days)
    deleted = 0
    for conversation in store.all_conversations(include_archived=True):
        for turn in store.all_turns(conversation.id):
            if turn.pinned or turn.created_at >= cutoff:
                continue
            assert turn.id is not None  # every turn read back from the store has an id
            store.forget(turn.id)
            deleted += 1
    return deleted


def _apply_turn_cap(store: MemoryStore, max_turns_per_conversation: int | None) -> int:
    if max_turns_per_conversation is None:
        return 0
    deleted = 0
    for conversation in store.all_conversations(include_archived=True):
        turns = store.all_turns(conversation.id)  # oldest first
        excess = len(turns) - max_turns_per_conversation
        if excess <= 0:
            continue
        removed_here = 0
        for turn in turns:
            if removed_here >= excess:
                break
            if turn.pinned:
                continue
            assert turn.id is not None
            store.forget(turn.id)
            removed_here += 1
        deleted += removed_here
    return deleted
