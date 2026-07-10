"""SQLite memory store tests (ADR-019): schema, migrations, CRUD, search,
management verbs, export/import, corruption/concurrency safety.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from eva.core.errors import MemoryNotFoundError, MemoryStoreError
from eva.memory import db
from eva.memory.models import MemorySummary, UserProfile
from eva.memory.sqlite_store import SQLiteMemoryStore, SQLiteUserProfileStore


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "memory.db"


@pytest.fixture
def store(db_path: Path) -> Iterator[SQLiteMemoryStore]:
    conn = db.connect(db_path)
    s = SQLiteMemoryStore(conn)
    yield s
    s.close()


class TestSchemaAndMigrations:
    def test_fresh_database_reaches_current_version(self, db_path: Path) -> None:
        conn = db.connect(db_path)
        assert db.schema_version(conn) == db.CURRENT_SCHEMA_VERSION
        conn.close()

    def test_reopening_is_idempotent(self, db_path: Path) -> None:
        conn1 = db.connect(db_path)
        conn1.close()
        conn2 = db.connect(db_path)  # must not re-run migration 1 or raise
        assert db.schema_version(conn2) == db.CURRENT_SCHEMA_VERSION
        conn2.close()

    def test_fts5_detected_as_available(self, db_path: Path) -> None:
        conn = db.connect(db_path)
        assert db.fts5_enabled(conn) is True  # confirmed available in CI's sqlite3 build
        conn.close()

    def test_corrupted_file_raises_actionable_error(self, tmp_path: Path) -> None:
        bad = tmp_path / "corrupt.db"
        bad.write_bytes(b"not a sqlite database at all")
        with pytest.raises(MemoryStoreError):
            db.connect(bad)


class TestConversationsAndTurns:
    def test_start_conversation_and_add_turns(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation(language="en", title="test")
        t1 = store.add_turn(conv.id, "user", "hello there")
        t2 = store.add_turn(conv.id, "assistant", "hi, how can I help?")
        turns = store.recent_turns(conv.id, 10)
        assert [t.text for t in turns] == ["hello there", "hi, how can I help?"]
        assert turns[0].id == t1.id
        assert turns[1].id == t2.id

    def test_recent_turns_respects_limit_and_order(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation()
        for i in range(10):
            store.add_turn(conv.id, "user", f"message {i}")
        turns = store.recent_turns(conv.id, 3)
        assert [t.text for t in turns] == ["message 7", "message 8", "message 9"]

    def test_all_turns_returns_everything_oldest_first(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation()
        for i in range(5):
            store.add_turn(conv.id, "user", f"turn {i}")
        turns = store.all_turns(conv.id)
        assert [t.text for t in turns] == [f"turn {i}" for i in range(5)]

    def test_all_turns_excludes_forgotten(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation()
        keep = store.add_turn(conv.id, "user", "keep me")
        gone = store.add_turn(conv.id, "user", "forget me")
        store.forget(gone.id)
        assert [t.id for t in store.all_turns(conv.id)] == [keep.id]

    def test_add_turn_to_unknown_conversation_raises(self, store: SQLiteMemoryStore) -> None:
        with pytest.raises(MemoryNotFoundError):
            store.add_turn("does-not-exist", "user", "hi")

    def test_get_turn_unknown_raises(self, store: SQLiteMemoryStore) -> None:
        with pytest.raises(MemoryNotFoundError):
            store.get_turn(999)

    def test_get_turns_bulk_fetch(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation()
        turns = [store.add_turn(conv.id, "user", f"turn {i}") for i in range(5)]
        ids = [t.id for t in turns if t.id is not None]
        fetched = store.get_turns(ids)
        assert {t.id for t in fetched} == set(ids)

    def test_get_turns_silently_skips_missing_and_deleted(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation()
        keep = store.add_turn(conv.id, "user", "keep")
        gone = store.add_turn(conv.id, "user", "forgotten")
        store.forget(gone.id)
        fetched = store.get_turns([keep.id, gone.id, 999999])
        assert [t.id for t in fetched] == [keep.id]

    def test_get_turns_empty_list_returns_empty(self, store: SQLiteMemoryStore) -> None:
        assert store.get_turns([]) == []

    def test_metadata_round_trips_as_json(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation()
        turn = store.add_turn(conv.id, "user", "hi", metadata={"source": "voice", "n": 1})
        fetched = store.get_turn(turn.id)
        assert fetched.metadata == {"source": "voice", "n": 1}

    def test_forget_permanently_deletes(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation()
        turn = store.add_turn(conv.id, "user", "secret")
        store.forget(turn.id)
        with pytest.raises(MemoryNotFoundError):
            store.get_turn(turn.id)
        assert store.recent_turns(conv.id, 10) == []

    def test_forget_unknown_turn_raises(self, store: SQLiteMemoryStore) -> None:
        with pytest.raises(MemoryNotFoundError):
            store.forget(999)

    def test_pin_and_favorite(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation()
        turn = store.add_turn(conv.id, "user", "important")
        store.pin(turn.id)
        store.favorite(turn.id)
        fetched = store.get_turn(turn.id)
        assert fetched.pinned is True
        assert fetched.favorite is True
        store.pin(turn.id, pinned=False)
        assert store.get_turn(turn.id).pinned is False

    def test_archive_and_delete_conversation(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation()
        store.add_turn(conv.id, "user", "hi")
        store.archive_conversation(conv.id)
        assert conv.id not in [c.id for c in store.all_conversations()]
        assert conv.id in [c.id for c in store.all_conversations(include_archived=True)]

        store.delete_conversation(conv.id)
        assert conv.id not in [c.id for c in store.all_conversations(include_archived=True)]

    def test_archive_unknown_conversation_raises(self, store: SQLiteMemoryStore) -> None:
        with pytest.raises(MemoryNotFoundError):
            store.archive_conversation("nope")

    def test_set_title_persists_and_strips(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation()
        store.set_title(conv.id, "  Learning Finnish  ")
        stored = next(c for c in store.all_conversations() if c.id == conv.id)
        assert stored.title == "Learning Finnish"

    def test_set_title_unknown_conversation_raises(self, store: SQLiteMemoryStore) -> None:
        with pytest.raises(MemoryNotFoundError):
            store.set_title("nope", "Title")

    def test_title_round_trips_through_export_import(self, store: SQLiteMemoryStore) -> None:
        """M5.4 §2: titles are part of the permanent record."""
        conv = store.start_conversation()
        store.add_turn(conv.id, "user", "hi")
        store.set_title(conv.id, "Vacation Planning")
        snapshot = store.export_json(conv.id)
        store.delete_conversation(conv.id)
        store.import_json(snapshot)
        restored = next(c for c in store.all_conversations() if c.id == conv.id)
        assert restored.title == "Vacation Planning"

    def test_merge_conversations_moves_turns(self, store: SQLiteMemoryStore) -> None:
        source = store.start_conversation()
        target = store.start_conversation()
        store.add_turn(source.id, "user", "from source")
        store.add_turn(target.id, "user", "already in target")

        store.merge_conversations(source.id, target.id)

        remaining_ids = [c.id for c in store.all_conversations(include_archived=True)]
        assert source.id not in remaining_ids
        assert target.id in remaining_ids
        texts = {t.text for t in store.recent_turns(target.id, 10)}
        assert texts == {"from source", "already in target"}

    def test_delete_all_wipes_everything(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation()
        store.add_turn(conv.id, "user", "hi")
        store.delete_all()
        assert store.all_conversations(include_archived=True) == []
        stats = store.stats()
        assert stats.conversation_count == 0
        assert stats.turn_count == 0


class TestSearch:
    def test_search_finds_matching_turn(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation()
        store.add_turn(conv.id, "user", "what is the weather like today")
        store.add_turn(conv.id, "assistant", "it is sunny")
        results = store.search_text("weather")
        assert any("weather" in r.turn.text for r in results)

    def test_search_scoped_to_conversation(self, store: SQLiteMemoryStore) -> None:
        conv_a = store.start_conversation()
        conv_b = store.start_conversation()
        store.add_turn(conv_a.id, "user", "pizza toppings")
        store.add_turn(conv_b.id, "user", "pizza delivery")
        results = store.search_text("pizza", conversation_id=conv_a.id)
        assert len(results) == 1
        assert results[0].turn.conversation_id == conv_a.id

    def test_search_excludes_forgotten_turns(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation()
        turn = store.add_turn(conv.id, "user", "unique_marker_text")
        store.forget(turn.id)
        assert store.search_text("unique_marker_text") == []

    def test_search_survives_special_characters(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation()
        store.add_turn(conv.id, "user", "cost is $5.00 - a bargain!")
        # Characters like -, $, " have special meaning in FTS5 query syntax;
        # a naive MATCH would raise sqlite3.OperationalError on these.
        results = store.search_text('$5.00 - "quoted"')
        assert results == [] or isinstance(results, list)  # must not raise

    def test_like_fallback_when_fts5_disabled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(db, "_fts5_available", lambda conn: False)
        conn = db.connect(tmp_path / "no_fts5.db")
        assert db.fts5_enabled(conn) is False
        store_no_fts = SQLiteMemoryStore(conn)
        conv = store_no_fts.start_conversation()
        store_no_fts.add_turn(conv.id, "user", "searchable phrase here")
        results = store_no_fts.search_text("searchable")
        assert len(results) == 1
        store_no_fts.close()


class TestEmbeddings:
    def test_store_and_fetch_embeddings(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation()
        turn = store.add_turn(conv.id, "user", "hi")
        vector = (b"\x00\x01" * 8) * 24  # arbitrary fixed-size blob
        store.store_embedding(turn.id, "test-model", vector, dim=384)
        rows = store.embeddings_for(conv.id)
        assert len(rows) == 1
        assert rows[0][0] == turn.id
        assert rows[0][2] == 384

    def test_store_embedding_upserts(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation()
        turn = store.add_turn(conv.id, "user", "hi")
        store.store_embedding(turn.id, "model-a", b"x" * 4, dim=1)
        store.store_embedding(turn.id, "model-b", b"y" * 4, dim=1)
        rows = store.embeddings_for(conv.id)
        assert len(rows) == 1  # upsert, not a second row
        assert rows[0][1] == b"y" * 4

    def test_embeddings_for_searches_across_conversations_when_unscoped(
        self, store: SQLiteMemoryStore
    ) -> None:
        conv_a = store.start_conversation()
        conv_b = store.start_conversation()
        turn_a = store.add_turn(conv_a.id, "user", "in a")
        turn_b = store.add_turn(conv_b.id, "user", "in b")
        store.store_embedding(turn_a.id, "m", b"x" * 4, dim=1)
        store.store_embedding(turn_b.id, "m", b"y" * 4, dim=1)
        rows = store.embeddings_for(None)
        assert {r[0] for r in rows} == {turn_a.id, turn_b.id}

    def test_embeddings_for_limit_keeps_most_recent(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation()
        older = store.add_turn(conv.id, "user", "older")
        store.store_embedding(older.id, "m", b"x" * 4, dim=1)
        newer = store.add_turn(conv.id, "user", "newer")
        store.store_embedding(newer.id, "m", b"y" * 4, dim=1)
        rows = store.embeddings_for(conv.id, limit=1)
        assert len(rows) == 1
        assert rows[0][0] == newer.id

    def test_embeddings_excludes_deleted_turns(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation()
        turn = store.add_turn(conv.id, "user", "hi")
        store.store_embedding(turn.id, "m", b"x" * 4, dim=1)
        store.forget(turn.id)
        assert store.embeddings_for(conv.id) == []


class TestSummaries:
    def test_add_and_fetch_latest_summary(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation()
        store.add_turn(conv.id, "user", "a" * 10)
        summary = MemorySummary(
            conversation_id=conv.id,
            turn_range_start=1,
            turn_range_end=1,
            text="A short summary.",
            created_at=datetime.now(UTC),
            model_id="test-llm",
        )
        saved = store.add_summary(summary)
        assert saved.id is not None
        fetched = store.latest_summary(conv.id)
        assert fetched is not None
        assert fetched.text == "A short summary."

    def test_latest_summary_none_when_absent(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation()
        assert store.latest_summary(conv.id) is None


class TestExportImport:
    def test_export_then_import_round_trips(self, store: SQLiteMemoryStore, tmp_path: Path) -> None:
        conv = store.start_conversation(title="original")
        store.add_turn(conv.id, "user", "round trip me")
        store.add_turn(conv.id, "assistant", "sure thing")

        exported = store.export_json()
        assert json.dumps(exported)  # must be JSON-serializable

        # Import into a fresh, empty store.
        conn2 = db.connect(tmp_path / "import_target.db")
        store2 = SQLiteMemoryStore(conn2)
        imported_count = store2.import_json(exported)
        assert imported_count == 2
        turns = store2.recent_turns(conv.id, 10)
        assert [t.text for t in turns] == ["round trip me", "sure thing"]
        store2.close()

    def test_import_rejects_unknown_version(self, store: SQLiteMemoryStore) -> None:
        with pytest.raises(MemoryStoreError):
            store.import_json({"version": 999, "conversations": []})

    def test_import_is_transactional_on_malformed_payload(self, store: SQLiteMemoryStore) -> None:
        before = store.stats().turn_count
        with pytest.raises(MemoryStoreError):
            store.import_json(
                {
                    "version": 1,
                    "conversations": [
                        {
                            "conversation": {"id": "x", "started_at": "2026-01-01T00:00:00"},
                            "turns": [{"speaker": "user"}],  # missing required keys
                        }
                    ],
                }
            )
        assert store.stats().turn_count == before  # rolled back, not partially applied


class TestStats:
    def test_stats_reflects_content(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation()
        store.add_turn(conv.id, "user", "hi")
        stats = store.stats()
        assert stats.conversation_count == 1
        assert stats.turn_count == 1
        assert stats.db_size_bytes > 0
        assert stats.fts_enabled is True


class TestConcurrentAccess:
    def test_two_connections_can_read_and_write_wal_mode(self, db_path: Path) -> None:
        conn_a = db.connect(db_path)
        conn_b = db.connect(db_path)
        store_a = SQLiteMemoryStore(conn_a)
        store_b = SQLiteMemoryStore(conn_b)

        conv = store_a.start_conversation()
        store_a.add_turn(conv.id, "user", "written by connection a")

        # A second, independent connection sees committed writes immediately.
        turns_seen_by_b = store_b.recent_turns(conv.id, 10)
        assert len(turns_seen_by_b) == 1

        store_b.add_turn(conv.id, "assistant", "written by connection b")
        turns_seen_by_a = store_a.recent_turns(conv.id, 10)
        assert len(turns_seen_by_a) == 2

        store_a.close()
        store_b.close()

    def test_journal_mode_is_wal(self, db_path: Path) -> None:
        conn = db.connect(db_path)
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"
        conn.close()


class TestUserProfileStore:
    @pytest.fixture
    def profiles(self, db_path: Path) -> Iterator[SQLiteUserProfileStore]:
        conn = db.connect(db_path)
        yield SQLiteUserProfileStore(conn)
        conn.close()

    def _make_profile(self, profile_id: str = "u1") -> UserProfile:
        now = datetime.now(UTC)
        return UserProfile(id=profile_id, nickname="Fahad", created_at=now, updated_at=now)

    def test_create_and_get(self, profiles: SQLiteUserProfileStore) -> None:
        created = profiles.create(self._make_profile())
        fetched = profiles.get(created.id)
        assert fetched.nickname == "Fahad"

    def test_get_unknown_raises(self, profiles: SQLiteUserProfileStore) -> None:
        with pytest.raises(MemoryNotFoundError):
            profiles.get("nope")

    def test_only_one_profile_active_at_a_time(self, profiles: SQLiteUserProfileStore) -> None:
        profiles.create(self._make_profile("u1"))
        profiles.create(self._make_profile("u2"))
        profiles.set_active("u1")
        assert profiles.active() is not None
        assert profiles.active().id == "u1"  # type: ignore[union-attr]
        profiles.set_active("u2")
        active = profiles.active()
        assert active is not None
        assert active.id == "u2"

    def test_active_none_when_no_profile_created(self, profiles: SQLiteUserProfileStore) -> None:
        assert profiles.active() is None

    def test_update_replaces_fields(self, profiles: SQLiteUserProfileStore) -> None:
        created = profiles.create(self._make_profile())
        updated = profiles.update(created.model_copy(update={"nickname": "Fahian"}))
        assert updated.nickname == "Fahian"
        assert profiles.get(created.id).nickname == "Fahian"

    def test_update_unknown_raises(self, profiles: SQLiteUserProfileStore) -> None:
        with pytest.raises(MemoryNotFoundError):
            profiles.update(self._make_profile("ghost"))

    def test_delete(self, profiles: SQLiteUserProfileStore) -> None:
        created = profiles.create(self._make_profile())
        profiles.delete(created.id)
        with pytest.raises(MemoryNotFoundError):
            profiles.get(created.id)

    def test_delete_unknown_raises(self, profiles: SQLiteUserProfileStore) -> None:
        with pytest.raises(MemoryNotFoundError):
            profiles.delete("nope")


class TestLongConversations:
    def test_hundreds_of_turns_stay_fast_and_correct(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation()
        for i in range(300):
            speaker = "user" if i % 2 == 0 else "assistant"
            store.add_turn(conv.id, speaker, f"turn number {i}")

        recent = store.recent_turns(conv.id, 10)
        assert [t.text for t in recent] == [f"turn number {i}" for i in range(290, 300)]
        assert store.stats().turn_count == 300

        results = store.search_text("turn number 250")
        assert any("250" in r.turn.text for r in results)


def test_sqlite_row_factory_used(db_path: Path) -> None:
    """Sanity check the connection returns `sqlite3.Row`, not plain tuples —
    `_row_to_*` helpers index by column name."""
    conn = db.connect(db_path)
    assert conn.row_factory is sqlite3.Row
    conn.close()
