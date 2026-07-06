"""SQLite adapter for `MemoryStore` and `UserProfileStore` (ADR-019, ADR-022).

Both classes share one `sqlite3.Connection` (from `eva.memory.db.connect()`)
rather than opening the database twice — they are two ports backed by one
file because user profiles and conversation memory are different concerns
that happen to want the same transactional/backup unit.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from typing import Any

from eva.core.errors import MemoryNotFoundError, MemoryStoreError
from eva.memory import db
from eva.memory.base import MemoryStore, UserProfileStore
from eva.memory.models import (
    MemoryConversation,
    MemorySearchResult,
    MemoryStats,
    MemorySummary,
    MemoryTurn,
    Speaker,
    UserProfile,
)

EXPORT_FORMAT_VERSION = 1


def _fts5_phrase(query: str) -> str:
    """Treat arbitrary user input as one literal FTS5 phrase, not a query
    expression — otherwise characters like `-`/`*`/`"` can raise
    `sqlite3.OperationalError` on a plain keyword search."""
    return '"' + query.replace('"', '""') + '"'


class SQLiteMemoryStore(MemoryStore):
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._closed = False

    # ── conversations ──

    def start_conversation(self, *, language: str = "en", title: str = "") -> MemoryConversation:
        conversation_id = str(uuid.uuid4())
        now = datetime.now(UTC)
        self._conn.execute(
            "INSERT INTO conversations (id, started_at, title, language) VALUES (?, ?, ?, ?)",
            (conversation_id, now.isoformat(), title, language),
        )
        self._conn.commit()
        return MemoryConversation(
            id=conversation_id, started_at=now, title=title, language=language
        )

    def all_conversations(self, *, include_archived: bool = False) -> list[MemoryConversation]:
        if include_archived:
            rows = self._conn.execute(
                "SELECT * FROM conversations ORDER BY started_at DESC"
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM conversations WHERE archived = 0 ORDER BY started_at DESC"
            ).fetchall()
        return [self._row_to_conversation(r) for r in rows]

    def archive_conversation(self, conversation_id: str, *, archived: bool = True) -> None:
        cur = self._conn.execute(
            "UPDATE conversations SET archived = ? WHERE id = ?",
            (int(archived), conversation_id),
        )
        self._conn.commit()
        if cur.rowcount == 0:
            raise MemoryNotFoundError(f"No conversation with id {conversation_id!r}")

    def delete_conversation(self, conversation_id: str) -> None:
        self._require_conversation(conversation_id)
        self._conn.execute(
            "DELETE FROM embeddings WHERE turn_id IN "
            "(SELECT id FROM turns WHERE conversation_id = ?)",
            (conversation_id,),
        )
        self._conn.execute("DELETE FROM summaries WHERE conversation_id = ?", (conversation_id,))
        self._conn.execute("DELETE FROM turns WHERE conversation_id = ?", (conversation_id,))
        self._conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        self._conn.commit()

    def merge_conversations(self, source_id: str, target_id: str) -> None:
        self._require_conversation(source_id)
        self._require_conversation(target_id)
        self._conn.execute(
            "UPDATE turns SET conversation_id = ? WHERE conversation_id = ?",
            (target_id, source_id),
        )
        self._conn.execute(
            "UPDATE summaries SET conversation_id = ? WHERE conversation_id = ?",
            (target_id, source_id),
        )
        self._conn.execute("DELETE FROM conversations WHERE id = ?", (source_id,))
        self._conn.commit()

    def delete_all(self) -> None:
        self._conn.execute("DELETE FROM embeddings")
        self._conn.execute("DELETE FROM summaries")
        self._conn.execute("DELETE FROM turns")
        self._conn.execute("DELETE FROM conversations")
        self._conn.commit()

    # ── turns ──

    def add_turn(
        self,
        conversation_id: str,
        speaker: Speaker,
        text: str,
        *,
        language: str = "en",
        metadata: dict[str, Any] | None = None,
    ) -> MemoryTurn:
        self._require_conversation(conversation_id)
        now = datetime.now(UTC)
        meta = metadata or {}
        cur = self._conn.execute(
            "INSERT INTO turns (conversation_id, created_at, speaker, text, language, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (conversation_id, now.isoformat(), speaker, text, language, json.dumps(meta)),
        )
        self._conn.commit()
        turn_id = cur.lastrowid
        assert turn_id is not None  # AUTOINCREMENT guarantees this after INSERT
        return MemoryTurn(
            id=turn_id,
            conversation_id=conversation_id,
            created_at=now,
            speaker=speaker,
            text=text,
            language=language,
            metadata=meta,
        )

    def get_turn(self, turn_id: int) -> MemoryTurn:
        row = self._conn.execute(
            "SELECT * FROM turns WHERE id = ? AND deleted = 0", (turn_id,)
        ).fetchone()
        if row is None:
            raise MemoryNotFoundError(f"No turn with id {turn_id}")
        return self._row_to_turn(row)

    def recent_turns(self, conversation_id: str, limit: int) -> list[MemoryTurn]:
        rows = self._conn.execute(
            "SELECT * FROM turns WHERE conversation_id = ? AND deleted = 0 "
            "ORDER BY id DESC LIMIT ?",
            (conversation_id, limit),
        ).fetchall()
        return [self._row_to_turn(r) for r in reversed(rows)]

    def get_turns(self, turn_ids: list[int]) -> list[MemoryTurn]:
        if not turn_ids:
            return []
        # Placeholder count only — turn_ids themselves are always bound as
        # parameters, never interpolated into the SQL text.
        placeholders = ",".join("?" * len(turn_ids))
        rows = self._conn.execute(
            f"SELECT * FROM turns WHERE id IN ({placeholders}) AND deleted = 0",
            turn_ids,
        ).fetchall()
        return [self._row_to_turn(r) for r in rows]

    def all_turns(self, conversation_id: str) -> list[MemoryTurn]:
        rows = self._conn.execute(
            "SELECT * FROM turns WHERE conversation_id = ? AND deleted = 0 ORDER BY id",
            (conversation_id,),
        ).fetchall()
        return [self._row_to_turn(r) for r in rows]

    def forget(self, turn_id: int) -> None:
        self._conn.execute("DELETE FROM embeddings WHERE turn_id = ?", (turn_id,))
        cur = self._conn.execute("DELETE FROM turns WHERE id = ?", (turn_id,))
        self._conn.commit()
        if cur.rowcount == 0:
            raise MemoryNotFoundError(f"No turn with id {turn_id}")

    def pin(self, turn_id: int, *, pinned: bool = True) -> None:
        cur = self._conn.execute("UPDATE turns SET pinned = ? WHERE id = ?", (int(pinned), turn_id))
        self._conn.commit()
        if cur.rowcount == 0:
            raise MemoryNotFoundError(f"No turn with id {turn_id}")

    def favorite(self, turn_id: int, *, favorite: bool = True) -> None:
        cur = self._conn.execute(
            "UPDATE turns SET favorite = ? WHERE id = ?", (int(favorite), turn_id)
        )
        self._conn.commit()
        if cur.rowcount == 0:
            raise MemoryNotFoundError(f"No turn with id {turn_id}")

    # ── search ──

    def search_text(
        self, query: str, *, limit: int = 20, conversation_id: str | None = None
    ) -> list[MemorySearchResult]:
        if db.fts5_enabled(self._conn):
            sql = (
                "SELECT turns.* FROM turns_fts "
                "JOIN turns ON turns.id = turns_fts.rowid "
                "WHERE turns_fts MATCH ? AND turns.deleted = 0"
            )
            params: list[Any] = [_fts5_phrase(query)]
            if conversation_id is not None:
                sql += " AND turns.conversation_id = ?"
                params.append(conversation_id)
            sql += " ORDER BY rank LIMIT ?"
            params.append(limit)
        else:
            sql = "SELECT * FROM turns WHERE deleted = 0 AND text LIKE ?"
            params = [f"%{query}%"]
            if conversation_id is not None:
                sql += " AND conversation_id = ?"
                params.append(conversation_id)
            sql += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
        try:
            rows = self._conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError as exc:
            raise MemoryStoreError(f"Memory search failed: {exc}") from exc
        return [
            MemorySearchResult(turn=self._row_to_turn(r), score=1.0, match_reason="keyword")
            for r in rows
        ]

    def store_embedding(self, turn_id: int, model_id: str, vector: bytes, dim: int) -> None:
        self._conn.execute(
            "INSERT INTO embeddings (turn_id, model_id, vector, dim, created_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(turn_id) DO UPDATE SET "
            "model_id = excluded.model_id, vector = excluded.vector, "
            "dim = excluded.dim, created_at = excluded.created_at",
            (turn_id, model_id, vector, dim, datetime.now(UTC).isoformat()),
        )
        self._conn.commit()

    def embeddings_for(
        self, conversation_id: str | None = None, *, limit: int | None = None
    ) -> list[tuple[int, bytes, int]]:
        sql = (
            "SELECT embeddings.turn_id, embeddings.vector, embeddings.dim "
            "FROM embeddings JOIN turns ON turns.id = embeddings.turn_id "
            "WHERE turns.deleted = 0"
        )
        params: list[Any] = []
        if conversation_id is not None:
            sql += " AND turns.conversation_id = ?"
            params.append(conversation_id)
        if limit is not None:
            # Most-recently-created first, capped — bounds retrieval cost
            # independent of total history size (ADR-020). Ordered by
            # turn_id (embeddings' own PRIMARY KEY, hence already indexed
            # and monotonically increasing with insertion order) rather than
            # the unindexed created_at column — the latter forces a full
            # sort of every matching row before LIMIT can apply, which
            # defeats the whole point of bounding the scan (measured: this
            # was the dominant cost at 20k+ turns, not the numpy math).
            sql += " ORDER BY embeddings.turn_id DESC LIMIT ?"
            params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [(int(r["turn_id"]), bytes(r["vector"]), int(r["dim"])) for r in rows]

    # ── summaries ──

    def add_summary(self, summary: MemorySummary) -> MemorySummary:
        self._require_conversation(summary.conversation_id)
        cur = self._conn.execute(
            "INSERT INTO summaries "
            "(conversation_id, turn_range_start, turn_range_end, text, created_at, model_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                summary.conversation_id,
                summary.turn_range_start,
                summary.turn_range_end,
                summary.text,
                summary.created_at.isoformat(),
                summary.model_id,
            ),
        )
        self._conn.commit()
        return summary.model_copy(update={"id": cur.lastrowid})

    def latest_summary(self, conversation_id: str) -> MemorySummary | None:
        row = self._conn.execute(
            "SELECT * FROM summaries WHERE conversation_id = ? ORDER BY id DESC LIMIT 1",
            (conversation_id,),
        ).fetchone()
        return self._row_to_summary(row) if row is not None else None

    # ── import/export ──

    def export_json(self, conversation_id: str | None = None) -> dict[str, Any]:
        if conversation_id is not None:
            self._require_conversation(conversation_id)
            conversations = self._conn.execute(
                "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchall()
        else:
            conversations = self._conn.execute("SELECT * FROM conversations").fetchall()

        entries = []
        for conv in conversations:
            turns = self._conn.execute(
                "SELECT * FROM turns WHERE conversation_id = ? AND deleted = 0 ORDER BY id",
                (conv["id"],),
            ).fetchall()
            summaries = self._conn.execute(
                "SELECT * FROM summaries WHERE conversation_id = ? ORDER BY id",
                (conv["id"],),
            ).fetchall()
            entries.append(
                {
                    "conversation": dict(conv),
                    "turns": [dict(t) for t in turns],
                    "summaries": [dict(s) for s in summaries],
                }
            )
        return {"version": EXPORT_FORMAT_VERSION, "conversations": entries}

    def import_json(self, payload: dict[str, Any]) -> int:
        if payload.get("version") != EXPORT_FORMAT_VERSION:
            raise MemoryStoreError(
                f"Unsupported memory export version: {payload.get('version')!r} "
                f"(expected {EXPORT_FORMAT_VERSION})"
            )
        imported = 0
        try:
            for entry in payload.get("conversations", []):
                conv = entry["conversation"]
                conv_id = conv["id"]
                exists = self._conn.execute(
                    "SELECT 1 FROM conversations WHERE id = ?", (conv_id,)
                ).fetchone()
                if exists is None:
                    self._conn.execute(
                        "INSERT INTO conversations "
                        "(id, started_at, title, language, archived) VALUES (?, ?, ?, ?, ?)",
                        (
                            conv_id,
                            conv["started_at"],
                            conv.get("title", ""),
                            conv.get("language", "en"),
                            conv.get("archived", 0),
                        ),
                    )
                for turn in entry.get("turns", []):
                    self._conn.execute(
                        "INSERT INTO turns "
                        "(conversation_id, created_at, speaker, text, language, metadata, "
                        "pinned, favorite) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            conv_id,
                            turn["created_at"],
                            turn["speaker"],
                            turn["text"],
                            turn.get("language", "en"),
                            turn.get("metadata", "{}"),
                            turn.get("pinned", 0),
                            turn.get("favorite", 0),
                        ),
                    )
                    imported += 1
                for summary in entry.get("summaries", []):
                    self._conn.execute(
                        "INSERT INTO summaries "
                        "(conversation_id, turn_range_start, turn_range_end, text, "
                        "created_at, model_id) VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            conv_id,
                            summary["turn_range_start"],
                            summary["turn_range_end"],
                            summary["text"],
                            summary["created_at"],
                            summary["model_id"],
                        ),
                    )
        except (KeyError, sqlite3.Error) as exc:
            self._conn.rollback()
            raise MemoryStoreError(f"Memory import failed: {exc}") from exc
        self._conn.commit()
        return imported

    # ── lifecycle ──

    def stats(self) -> MemoryStats:
        conv_count = self._conn.execute("SELECT COUNT(*) AS c FROM conversations").fetchone()["c"]
        turn_count = self._conn.execute(
            "SELECT COUNT(*) AS c FROM turns WHERE deleted = 0"
        ).fetchone()["c"]
        embedded_count = self._conn.execute("SELECT COUNT(*) AS c FROM embeddings").fetchone()["c"]
        summary_count = self._conn.execute("SELECT COUNT(*) AS c FROM summaries").fetchone()["c"]
        page_count = self._conn.execute("PRAGMA page_count").fetchone()[0]
        page_size = self._conn.execute("PRAGMA page_size").fetchone()[0]
        return MemoryStats(
            conversation_count=conv_count,
            turn_count=turn_count,
            embedded_turn_count=embedded_count,
            summary_count=summary_count,
            db_size_bytes=int(page_count) * int(page_size),
            fts_enabled=db.fts5_enabled(self._conn),
        )

    def close(self) -> None:
        if not self._closed:
            self._conn.close()
            self._closed = True

    # ── row conversion ──

    def _row_to_conversation(self, row: sqlite3.Row) -> MemoryConversation:
        return MemoryConversation(
            id=row["id"],
            started_at=datetime.fromisoformat(row["started_at"]),
            title=row["title"],
            language=row["language"],
            archived=bool(row["archived"]),
        )

    def _row_to_turn(self, row: sqlite3.Row) -> MemoryTurn:
        return MemoryTurn(
            id=row["id"],
            conversation_id=row["conversation_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            speaker=row["speaker"],
            text=row["text"],
            language=row["language"],
            metadata=json.loads(row["metadata"]),
            pinned=bool(row["pinned"]),
            favorite=bool(row["favorite"]),
            deleted=bool(row["deleted"]),
        )

    def _row_to_summary(self, row: sqlite3.Row) -> MemorySummary:
        return MemorySummary(
            id=row["id"],
            conversation_id=row["conversation_id"],
            turn_range_start=row["turn_range_start"],
            turn_range_end=row["turn_range_end"],
            text=row["text"],
            created_at=datetime.fromisoformat(row["created_at"]),
            model_id=row["model_id"],
        )

    def _require_conversation(self, conversation_id: str) -> None:
        row = self._conn.execute(
            "SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        if row is None:
            raise MemoryNotFoundError(f"No conversation with id {conversation_id!r}")


class SQLiteUserProfileStore(UserProfileStore):
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def create(self, profile: UserProfile) -> UserProfile:
        now = datetime.now(UTC)
        self._conn.execute(
            "INSERT INTO user_profiles "
            "(id, nickname, preferred_language, preferred_voice, preferred_llm_model, "
            "conversation_style, units, timezone, extra, created_at, updated_at, active) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                profile.id,
                profile.nickname,
                profile.preferred_language,
                profile.preferred_voice,
                profile.preferred_llm_model,
                profile.conversation_style,
                profile.units,
                profile.timezone,
                json.dumps(profile.extra),
                now.isoformat(),
                now.isoformat(),
                int(profile.active),
            ),
        )
        self._conn.commit()
        return profile.model_copy(update={"created_at": now, "updated_at": now})

    def get(self, profile_id: str) -> UserProfile:
        row = self._conn.execute(
            "SELECT * FROM user_profiles WHERE id = ?", (profile_id,)
        ).fetchone()
        if row is None:
            raise MemoryNotFoundError(f"No user profile with id {profile_id!r}")
        return self._row_to_profile(row)

    def list(self) -> list[UserProfile]:
        rows = self._conn.execute("SELECT * FROM user_profiles ORDER BY created_at").fetchall()
        return [self._row_to_profile(r) for r in rows]

    def update(self, profile: UserProfile) -> UserProfile:
        now = datetime.now(UTC)
        cur = self._conn.execute(
            "UPDATE user_profiles SET nickname = ?, preferred_language = ?, "
            "preferred_voice = ?, preferred_llm_model = ?, conversation_style = ?, "
            "units = ?, timezone = ?, extra = ?, updated_at = ? WHERE id = ?",
            (
                profile.nickname,
                profile.preferred_language,
                profile.preferred_voice,
                profile.preferred_llm_model,
                profile.conversation_style,
                profile.units,
                profile.timezone,
                json.dumps(profile.extra),
                now.isoformat(),
                profile.id,
            ),
        )
        self._conn.commit()
        if cur.rowcount == 0:
            raise MemoryNotFoundError(f"No user profile with id {profile.id!r}")
        return profile.model_copy(update={"updated_at": now})

    def set_active(self, profile_id: str) -> None:
        self.get(profile_id)  # raises MemoryNotFoundError if missing
        self._conn.execute("UPDATE user_profiles SET active = 0")
        self._conn.execute("UPDATE user_profiles SET active = 1 WHERE id = ?", (profile_id,))
        self._conn.commit()

    def active(self) -> UserProfile | None:
        row = self._conn.execute("SELECT * FROM user_profiles WHERE active = 1 LIMIT 1").fetchone()
        return self._row_to_profile(row) if row is not None else None

    def delete(self, profile_id: str) -> None:
        cur = self._conn.execute("DELETE FROM user_profiles WHERE id = ?", (profile_id,))
        self._conn.commit()
        if cur.rowcount == 0:
            raise MemoryNotFoundError(f"No user profile with id {profile_id!r}")

    def _row_to_profile(self, row: sqlite3.Row) -> UserProfile:
        return UserProfile(
            id=row["id"],
            nickname=row["nickname"],
            preferred_language=row["preferred_language"],
            preferred_voice=row["preferred_voice"],
            preferred_llm_model=row["preferred_llm_model"],
            conversation_style=row["conversation_style"],
            units=row["units"],
            timezone=row["timezone"],
            extra=json.loads(row["extra"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            active=bool(row["active"]),
        )
