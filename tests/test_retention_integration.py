"""Retention policy wiring: automatic startup cleanup and the manual verb.

The policy itself is covered by `test_retention.py`; these tests cover when it
runs, what gates it, and that it leaves the memory index consistent.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from eva.config.settings import MemorySettings, Settings
from eva.memory import db
from eva.memory.retention import apply_retention_policy
from eva.memory.sqlite_store import SQLiteMemoryStore


@pytest.fixture
def store(tmp_path: Path) -> Iterator[SQLiteMemoryStore]:
    conn = db.connect(tmp_path / "memory.db")
    s = SQLiteMemoryStore(conn)
    yield s
    s.close()


def _turn(store: SQLiteMemoryStore, conv: str, text: str, days_ago: float) -> int:
    turn = store.add_turn(conv, "user", text)
    when = datetime.now(UTC) - timedelta(days=days_ago)
    store._conn.execute("UPDATE turns SET created_at = ? WHERE id = ?", (when.isoformat(), turn.id))
    store._conn.commit()
    store.store_embedding(int(turn.id or 0), "m", b"\x00" * (4 * 384), 384)
    return int(turn.id or 0)


def _orphaned_embeddings(store: SQLiteMemoryStore) -> int:
    row = store._conn.execute(
        "SELECT COUNT(*) FROM embeddings WHERE turn_id NOT IN (SELECT id FROM turns)"
    ).fetchone()
    return int(row[0])


class TestPolicyBehaviour:
    def test_age_limit_removes_only_turns_past_the_threshold(
        self, store: SQLiteMemoryStore
    ) -> None:
        conv = store.start_conversation().id
        old = _turn(store, conv, "old", days_ago=40)
        fresh = _turn(store, conv, "fresh", days_ago=5)
        apply_retention_policy(store, MemorySettings(retention_days=30))
        remaining = {t.id for t in store.all_turns(conv)}
        assert remaining == {fresh}
        assert old not in remaining

    def test_the_cap_keeps_the_newest_turns(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation().id
        ids = [_turn(store, conv, f"turn {i}", days_ago=20 - i) for i in range(15)]
        apply_retention_policy(store, MemorySettings(max_turns_per_conversation=10))
        remaining = [t.id for t in store.all_turns(conv)]
        assert remaining == ids[-10:], "the cap must drop the oldest, not an arbitrary set"

    def test_pinned_turns_survive_both_rules(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation().id
        pinned = _turn(store, conv, "important", days_ago=400)
        store.pin(pinned)
        for i in range(14):
            _turn(store, conv, f"disposable {i}", days_ago=399 - i)
        apply_retention_policy(
            store, MemorySettings(retention_days=30, max_turns_per_conversation=10)
        )
        assert [t.id for t in store.all_turns(conv)] == [pinned], (
            "the age limit removes everything unpinned; the pin is what exempts it"
        )

    def test_unlimited_settings_delete_nothing(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation().id
        for i in range(4):
            _turn(store, conv, f"turn {i}", days_ago=999)
        report = apply_retention_policy(
            store, MemorySettings(retention_days=None, max_turns_per_conversation=None)
        )
        assert report.total_deleted == 0
        assert len(store.all_turns(conv)) == 4

    def test_an_empty_database_is_safe(self, store: SQLiteMemoryStore) -> None:
        report = apply_retention_policy(store, MemorySettings(retention_days=1))
        assert report.total_deleted == 0

    def test_conversations_are_capped_independently(self, store: SQLiteMemoryStore) -> None:
        busy = store.start_conversation().id
        quiet = store.start_conversation().id
        for i in range(15):
            _turn(store, busy, f"busy {i}", days_ago=5 - i * 0.1)
        _turn(store, quiet, "quiet", days_ago=5)
        apply_retention_policy(store, MemorySettings(max_turns_per_conversation=10))
        assert len(store.all_turns(busy)) == 10
        assert len(store.all_turns(quiet)) == 1, "a short conversation must be untouched"

    def test_cleanup_removes_the_embeddings_of_deleted_turns(
        self, store: SQLiteMemoryStore
    ) -> None:
        """A surviving vector would keep a deleted turn semantically
        retrievable, which is the opposite of what retention is for."""
        conv = store.start_conversation().id
        _turn(store, conv, "old", days_ago=400)
        keep = _turn(store, conv, "recent", days_ago=1)
        apply_retention_policy(store, MemorySettings(retention_days=30))
        assert _orphaned_embeddings(store) == 0
        assert store.embeddings_for(conv) and store.embeddings_for(conv)[0][0] == keep

    def test_cleanup_is_idempotent(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation().id
        for i in range(4):
            _turn(store, conv, f"turn {i}", days_ago=400)
        policy = MemorySettings(retention_days=30)
        first = apply_retention_policy(store, policy)
        second = apply_retention_policy(store, policy)
        assert first.total_deleted == 4
        assert second.total_deleted == 0


class TestAutomaticCleanupGate:
    """`auto_cleanup_enabled` decides whether startup applies the policy at
    all; the policy fields decide what it removes."""

    def test_disabled_by_default(self) -> None:
        assert Settings().memory.auto_cleanup_enabled is False

    def test_startup_cleanup_runs_when_enabled(self, store: SQLiteMemoryStore) -> None:
        from eva.engine import Assistant

        conv = store.start_conversation().id
        _turn(store, conv, "ancient", days_ago=400)
        settings = Settings()
        settings.memory.auto_cleanup_enabled = True
        settings.memory.retention_days = 30

        assistant = Assistant.__new__(Assistant)
        assistant.settings = settings
        assistant.memory = store
        assistant.start_retention_cleanup()

        deadline = time.perf_counter() + 5.0
        while store.all_turns(conv) and time.perf_counter() < deadline:
            time.sleep(0.01)
        assert store.all_turns(conv) == []

    def test_startup_cleanup_is_skipped_when_disabled(self, store: SQLiteMemoryStore) -> None:
        from eva.engine import Assistant

        conv = store.start_conversation().id
        _turn(store, conv, "ancient", days_ago=400)
        settings = Settings()
        settings.memory.auto_cleanup_enabled = False
        settings.memory.retention_days = 30

        assistant = Assistant.__new__(Assistant)
        assistant.settings = settings
        assistant.memory = store
        assistant.start_retention_cleanup()

        time.sleep(0.2)
        assert len(store.all_turns(conv)) == 1

    def test_startup_cleanup_does_not_block(self, store: SQLiteMemoryStore) -> None:
        """Culling a long history costs time proportional to the turns
        removed, so it must never sit in front of the first conversation."""
        from eva.engine import Assistant

        conv = store.start_conversation().id
        for i in range(200):
            _turn(store, conv, f"turn {i}", days_ago=400)
        settings = Settings()
        settings.memory.auto_cleanup_enabled = True
        settings.memory.retention_days = 30

        assistant = Assistant.__new__(Assistant)
        assistant.settings = settings
        assistant.memory = store
        started = time.perf_counter()
        assistant.start_retention_cleanup()
        elapsed_ms = (time.perf_counter() - started) * 1000
        assert elapsed_ms < 50, f"dispatch took {elapsed_ms:.0f} ms; it must not do the work inline"

    def test_a_failing_cleanup_does_not_escape(self, store: SQLiteMemoryStore) -> None:
        """Maintenance must never take the engine down with it."""
        from eva.engine import Assistant

        settings = Settings()
        settings.memory.auto_cleanup_enabled = True
        settings.memory.retention_days = 30

        class _Broken:
            def all_conversations(self, **_: object) -> list[object]:
                raise RuntimeError("database is locked")

        assistant = Assistant.__new__(Assistant)
        assistant.settings = settings
        assistant.memory = _Broken()  # type: ignore[assignment]
        assistant.start_retention_cleanup()  # must not raise
        time.sleep(0.2)


class TestManualCleanupIsIndependent:
    def test_manual_cleanup_applies_regardless_of_the_auto_gate(
        self, store: SQLiteMemoryStore
    ) -> None:
        """`eva memory cleanup` is an explicit instruction; the automatic gate
        governs startup only."""
        conv = store.start_conversation().id
        _turn(store, conv, "ancient", days_ago=400)
        policy = MemorySettings(auto_cleanup_enabled=False, retention_days=30)
        report = apply_retention_policy(store, policy)
        assert report.total_deleted == 1
