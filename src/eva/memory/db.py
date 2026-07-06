"""Shared SQLite connection, schema, and migrations (ADR-019).

One database file (`AppPaths.conversations_dir / "memory.db"`) backs every
memory-adjacent store (`SQLiteMemoryStore`, `SQLiteUserProfileStore`) so they
share one WAL-mode connection instead of opening the file twice. Migrations
are numbered and applied in order, the same shape as
`eva.config.settings.SETTINGS_SCHEMA_VERSION` — each version's SQL runs
exactly once, tracked in `schema_migrations`.

Text search prefers SQLite FTS5, kept in sync via triggers on `turns`. FTS5
is a compile-time SQLite option that not every platform's Python ships with;
absence is detected once at connect time and degrades to a `LIKE`-based
query plan (`sqlite_store.py`) — slower on very large histories, never a
hard failure, the same graceful-degradation shape used for GPU detection.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from eva.core.errors import MemoryStoreError

logger = logging.getLogger(__name__)

DB_FILENAME = "memory.db"

# Each entry's SQL runs exactly once, in order, the first time a database
# reaches that version. Never edit an already-released migration's SQL —
# add a new numbered one instead, even to fix a mistake in an old one.
_MIGRATIONS: tuple[tuple[int, str], ...] = (
    (
        1,
        """
        CREATE TABLE conversations (
            id TEXT PRIMARY KEY,
            started_at TEXT NOT NULL,
            title TEXT NOT NULL DEFAULT '',
            language TEXT NOT NULL DEFAULT 'en',
            archived INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE turns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL REFERENCES conversations(id),
            created_at TEXT NOT NULL,
            speaker TEXT NOT NULL CHECK(speaker IN ('user', 'assistant')),
            text TEXT NOT NULL,
            language TEXT NOT NULL DEFAULT 'en',
            metadata TEXT NOT NULL DEFAULT '{}',
            pinned INTEGER NOT NULL DEFAULT 0,
            favorite INTEGER NOT NULL DEFAULT 0,
            deleted INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX idx_turns_conversation ON turns(conversation_id);
        CREATE INDEX idx_turns_created_at ON turns(created_at);

        CREATE TABLE embeddings (
            turn_id INTEGER PRIMARY KEY REFERENCES turns(id),
            model_id TEXT NOT NULL,
            vector BLOB NOT NULL,
            dim INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL REFERENCES conversations(id),
            turn_range_start INTEGER NOT NULL,
            turn_range_end INTEGER NOT NULL,
            text TEXT NOT NULL,
            created_at TEXT NOT NULL,
            model_id TEXT NOT NULL
        );
        CREATE INDEX idx_summaries_conversation ON summaries(conversation_id);

        CREATE TABLE user_profiles (
            id TEXT PRIMARY KEY,
            nickname TEXT NOT NULL DEFAULT '',
            preferred_language TEXT,
            preferred_voice TEXT,
            preferred_llm_model TEXT,
            conversation_style TEXT NOT NULL DEFAULT '',
            units TEXT NOT NULL DEFAULT 'metric',
            timezone TEXT NOT NULL DEFAULT 'UTC',
            extra TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            active INTEGER NOT NULL DEFAULT 0
        );
        """,
    ),
)

CURRENT_SCHEMA_VERSION = _MIGRATIONS[-1][0]


def connect(db_path: Path) -> sqlite3.Connection:
    """Open (creating if needed) the memory database, applying any pending
    migrations and (re)establishing the FTS5 index/triggers if available.

    A corrupted or non-SQLite file at `db_path` does not fail at
    `sqlite3.connect()` (which only opens a handle) — it fails on the first
    real read/write, here the initial PRAGMA calls. Both paths are wrapped
    so a corrupted database is always reported as `MemoryStoreError`, never
    a raw `sqlite3.DatabaseError` reaching a caller that only expects
    `EvaError` subclasses.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        _migrate(conn)
    except sqlite3.Error as exc:
        raise MemoryStoreError(f"Cannot open memory database at {db_path}: {exc}") from exc
    return conn


def schema_version(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT MAX(version) AS v FROM schema_migrations").fetchone()
    return int(row["v"]) if row and row["v"] is not None else 0


def fts5_enabled(conn: sqlite3.Connection) -> bool:
    row = conn.execute("SELECT value FROM schema_meta WHERE key = 'fts5_enabled'").fetchone()
    return bool(row and row["value"] == "1")


def _fts5_available(conn: sqlite3.Connection) -> bool:
    try:
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS _fts5_probe USING fts5(x)")
        conn.execute("DROP TABLE _fts5_probe")
        return True
    except sqlite3.OperationalError:
        return False


def _migrate(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations "
        "(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    applied = {row["version"] for row in conn.execute("SELECT version FROM schema_migrations")}
    for version, sql in _MIGRATIONS:
        if version in applied:
            continue
        try:
            conn.executescript(sql)
            conn.execute(
                "INSERT INTO schema_migrations (version, applied_at) VALUES (?, datetime('now'))",
                (version,),
            )
            conn.commit()
        except sqlite3.Error as exc:
            conn.rollback()
            raise MemoryStoreError(f"Memory database migration {version} failed: {exc}") from exc

    fts_ok = _fts5_available(conn)
    if fts_ok:
        conn.executescript(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts
                USING fts5(text, content='turns', content_rowid='id');
            CREATE TRIGGER IF NOT EXISTS turns_ai AFTER INSERT ON turns BEGIN
                INSERT INTO turns_fts(rowid, text) VALUES (new.id, new.text);
            END;
            CREATE TRIGGER IF NOT EXISTS turns_ad AFTER DELETE ON turns BEGIN
                INSERT INTO turns_fts(turns_fts, rowid, text) VALUES ('delete', old.id, old.text);
            END;
            CREATE TRIGGER IF NOT EXISTS turns_au AFTER UPDATE ON turns BEGIN
                INSERT INTO turns_fts(turns_fts, rowid, text) VALUES ('delete', old.id, old.text);
                INSERT INTO turns_fts(rowid, text) VALUES (new.id, new.text);
            END;
            """
        )
    else:
        logger.warning(
            "SQLite build lacks FTS5 — memory text search falls back to LIKE "
            "queries (functionally equivalent, slower on large histories)."
        )
    conn.execute(
        "INSERT INTO schema_meta (key, value) VALUES ('fts5_enabled', ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        ("1" if fts_ok else "0",),
    )
    conn.commit()
