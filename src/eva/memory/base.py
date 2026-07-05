"""Memory subsystem ports (ADR-019, ADR-020, ADR-022).

`MemoryStore` is the persistence + management port: conversations, turns,
text search, and every management verb (forget/archive/pin/favorite/export/
import/merge) live here rather than on separate per-verb interfaces — see
ADR-019 for why `MemoryProvider`/`MemoryIndexer`/`MemoryCleaner`/
`MemoryExporter`/`MemoryImporter` were consolidated into one port.
`MemoryRetriever` (semantic search) and `Summarizer` are separate ports
because they plausibly have more than one real implementation.
`UserProfileStore` is a second, smaller port (ADR-022) that happens to share
a SQLite connection with `MemoryStore` in the built-in adapter, but is not
the same capability.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from eva.memory.models import (
    MemoryConversation,
    MemorySearchResult,
    MemoryStats,
    MemorySummary,
    MemoryTurn,
    Speaker,
    UserProfile,
)


class MemoryStore(ABC):
    """Persistent, searchable conversation memory."""

    # ── conversations ──

    @abstractmethod
    def start_conversation(self, *, language: str = "en", title: str = "") -> MemoryConversation:
        """Begin a new conversation and return it. Does not affect old data —
        this is what a "clear the current conversation" action maps to."""

    @abstractmethod
    def all_conversations(self, *, include_archived: bool = False) -> list[MemoryConversation]:
        """All conversations, newest first."""

    @abstractmethod
    def archive_conversation(self, conversation_id: str, *, archived: bool = True) -> None:
        """Reversible: hides from `all_conversations()` by default, keeps data."""

    @abstractmethod
    def delete_conversation(self, conversation_id: str) -> None:
        """Permanently remove a conversation and everything in it."""

    @abstractmethod
    def merge_conversations(self, source_id: str, target_id: str) -> None:
        """Move every turn/summary from `source_id` into `target_id`, then
        delete the now-empty source conversation."""

    @abstractmethod
    def delete_all(self) -> None:
        """Wipe every conversation, turn, embedding, and summary (privacy:
        "delete my data"). User profiles are untouched — a different
        capability (`UserProfileStore`) with its own lifecycle."""

    # ── turns ──

    @abstractmethod
    def add_turn(
        self,
        conversation_id: str,
        speaker: Speaker,
        text: str,
        *,
        language: str = "en",
        metadata: dict[str, Any] | None = None,
    ) -> MemoryTurn:
        """Record one speaker's utterance."""

    @abstractmethod
    def get_turn(self, turn_id: int) -> MemoryTurn:
        """Raises `MemoryNotFoundError` if `turn_id` does not exist."""

    @abstractmethod
    def get_turns(self, turn_ids: list[int]) -> list[MemoryTurn]:
        """Bulk fetch, order not guaranteed to match `turn_ids`. Missing ids
        (e.g. forgotten between an embedding scan and this call) are
        silently skipped rather than raising — this exists specifically so
        `MemoryRetriever` can score many embedding candidates without an
        N+1 query per candidate (measured, not assumed: see
        `eva.benchmark.memory`)."""

    @abstractmethod
    def recent_turns(self, conversation_id: str, limit: int) -> list[MemoryTurn]:
        """The last `limit` non-deleted turns of a conversation, oldest first."""

    @abstractmethod
    def all_turns(self, conversation_id: str) -> list[MemoryTurn]:
        """Every non-deleted turn in a conversation, oldest first — unlike
        `recent_turns`, not capped by a window. Used by retention sweeps and
        summarization, where the whole conversation is genuinely needed."""

    @abstractmethod
    def forget(self, turn_id: int) -> None:
        """Permanently delete one turn (privacy: "forget this")."""

    @abstractmethod
    def pin(self, turn_id: int, *, pinned: bool = True) -> None:
        """Pinned turns are boosted in retrieval scoring and exempt from
        retention cleanup."""

    @abstractmethod
    def favorite(self, turn_id: int, *, favorite: bool = True) -> None:
        """Favorited turns are boosted in retrieval scoring (like `pin` but
        does not exempt from retention — a UX distinction, not a storage one)."""

    # ── search ──

    @abstractmethod
    def search_text(
        self, query: str, *, limit: int = 20, conversation_id: str | None = None
    ) -> list[MemorySearchResult]:
        """Keyword search (FTS5 if available, LIKE fallback otherwise —
        transparent to callers; see `eva.memory.db`)."""

    @abstractmethod
    def store_embedding(self, turn_id: int, model_id: str, vector: bytes, dim: int) -> None:
        """`vector` is a packed float32 buffer (`numpy.ndarray.tobytes()`)."""

    @abstractmethod
    def embeddings_for(
        self, conversation_id: str | None = None, *, limit: int | None = None
    ) -> list[tuple[int, bytes, int]]:
        """`(turn_id, vector, dim)` for embedded, non-deleted turns in scope
        — what `MemoryRetriever` loads to build its search matrix.
        `conversation_id=None` searches every conversation (semantic memory
        is meant to recall *past* conversations, not just the active one —
        ADR-020). `limit`, when set, returns only the `limit` most recently
        created embeddings, bounding retrieval cost independent of how much
        history has accumulated (a personal assistant used for years must
        not get slower every year)."""

    # ── summaries ──

    @abstractmethod
    def add_summary(self, summary: MemorySummary) -> MemorySummary:
        """Store a summary. Never deletes the turns it summarizes."""

    @abstractmethod
    def latest_summary(self, conversation_id: str) -> MemorySummary | None:
        """Most recent summary for a conversation, or None."""

    # ── import/export ──

    @abstractmethod
    def export_json(self, conversation_id: str | None = None) -> dict[str, Any]:
        """A JSON-serializable snapshot: one conversation, or everything if
        `conversation_id` is None."""

    @abstractmethod
    def import_json(self, payload: dict[str, Any]) -> int:
        """Load a previously exported snapshot; returns turns imported."""

    # ── lifecycle ──

    @abstractmethod
    def stats(self) -> MemoryStats:
        """Aggregate counts and database size — feeds diagnostics."""

    @abstractmethod
    def close(self) -> None:
        """Release the underlying connection/handles. Idempotent."""


class MemoryRetriever(ABC):
    """Semantic search over embedded memories (ADR-020)."""

    @abstractmethod
    def retrieve(
        self,
        query_vector: bytes,
        *,
        top_k: int,
        conversation_id: str | None = None,
    ) -> list[MemorySearchResult]:
        """`query_vector` is a packed float32 buffer, same shape as
        `MemoryStore.store_embedding`'s `vector` argument. Scores blend
        cosine similarity, recency decay, and pinned/favorite weighting into
        one number (see the adapter for the exact formula and its settings)."""


class Summarizer(ABC):
    """Turns a run of conversation turns into a short summary (ADR-019 §9)."""

    @abstractmethod
    def summarize(self, turns: list[MemoryTurn]) -> str:
        """Never mutates or deletes `turns` — summaries are additive."""


class UserProfileStore(ABC):
    """Per-person preferences, separate from application settings (ADR-022)."""

    @abstractmethod
    def create(self, profile: UserProfile) -> UserProfile: ...

    @abstractmethod
    def get(self, profile_id: str) -> UserProfile:
        """Raises `MemoryNotFoundError` if `profile_id` does not exist."""

    @abstractmethod
    def list(self) -> list[UserProfile]: ...

    @abstractmethod
    def update(self, profile: UserProfile) -> UserProfile:
        """Replace the stored profile matching `profile.id`. Build the new
        value with `profile.model_copy(update={...})` (profiles are frozen)."""

    @abstractmethod
    def set_active(self, profile_id: str) -> None:
        """Exactly one profile is active at a time."""

    @abstractmethod
    def active(self) -> UserProfile | None:
        """The active profile, or None if none has been created yet."""

    @abstractmethod
    def delete(self, profile_id: str) -> None: ...
