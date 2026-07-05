"""Retention policy tests (ADR-019 Part 10) — real SQLiteMemoryStore, no
mocking: retention needs to prove it actually deletes the right rows.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from eva.config.settings import MemorySettings
from eva.memory import db
from eva.memory.retention import apply_retention_policy
from eva.memory.sqlite_store import SQLiteMemoryStore


@pytest.fixture
def store(tmp_path: Path) -> Iterator[SQLiteMemoryStore]:
    conn = db.connect(tmp_path / "memory.db")
    s = SQLiteMemoryStore(conn)
    yield s
    s.close()


def _backdate(store: SQLiteMemoryStore, turn_id: int, days_ago: int) -> None:
    timestamp = (datetime.now(UTC) - timedelta(days=days_ago)).isoformat()
    store._conn.execute("UPDATE turns SET created_at = ? WHERE id = ?", (timestamp, turn_id))
    store._conn.commit()


class TestNoOpDefaults:
    def test_no_settings_means_nothing_deleted(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation()
        store.add_turn(conv.id, "user", "hi")
        report = apply_retention_policy(store, MemorySettings())
        assert report.total_deleted == 0
        assert store.stats().turn_count == 1


class TestAgeBasedRetention:
    def test_old_turns_deleted_recent_turns_kept(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation()
        old = store.add_turn(conv.id, "user", "ancient")
        new = store.add_turn(conv.id, "user", "fresh")
        _backdate(store, old.id, days_ago=100)

        report = apply_retention_policy(
            store, MemorySettings(retention_days=30, max_turns_per_conversation=None)
        )

        assert report.turns_deleted_by_age == 1
        remaining = {t.id for t in store.all_turns(conv.id)}
        assert remaining == {new.id}

    def test_pinned_old_turns_survive(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation()
        old_pinned = store.add_turn(conv.id, "user", "old but pinned")
        _backdate(store, old_pinned.id, days_ago=100)
        store.pin(old_pinned.id)

        report = apply_retention_policy(
            store, MemorySettings(retention_days=30, max_turns_per_conversation=None)
        )

        assert report.turns_deleted_by_age == 0
        assert store.get_turn(old_pinned.id).text == "old but pinned"

    def test_sweeps_archived_conversations_too(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation()
        old = store.add_turn(conv.id, "user", "old")
        _backdate(store, old.id, days_ago=100)
        store.archive_conversation(conv.id)

        report = apply_retention_policy(
            store, MemorySettings(retention_days=30, max_turns_per_conversation=None)
        )
        assert report.turns_deleted_by_age == 1


class TestTurnCapRetention:
    def test_caps_conversation_to_max_turns_evicting_oldest_first(
        self, store: SQLiteMemoryStore
    ) -> None:
        conv = store.start_conversation()
        turns = [store.add_turn(conv.id, "user", f"turn {i}") for i in range(20)]

        report = apply_retention_policy(
            store, MemorySettings(retention_days=None, max_turns_per_conversation=10)
        )

        assert report.turns_deleted_by_cap == 10
        remaining_texts = [t.text for t in store.all_turns(conv.id)]
        assert remaining_texts == [f"turn {i}" for i in range(10, 20)]
        assert turns[0].id not in {t.id for t in store.all_turns(conv.id)}

    def test_pinned_turns_are_not_evicted_by_cap(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation()
        pinned_old = store.add_turn(conv.id, "user", "keep: pinned")
        store.pin(pinned_old.id)
        for i in range(20):
            store.add_turn(conv.id, "user", f"turn {i}")

        apply_retention_policy(
            store, MemorySettings(retention_days=None, max_turns_per_conversation=10)
        )

        remaining_ids = {t.id for t in store.all_turns(conv.id)}
        assert pinned_old.id in remaining_ids

    def test_under_cap_conversation_is_untouched(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation()
        store.add_turn(conv.id, "user", "only one turn")
        report = apply_retention_policy(
            store, MemorySettings(retention_days=None, max_turns_per_conversation=100)
        )
        assert report.turns_deleted_by_cap == 0


class TestCombinedPolicy:
    def test_age_and_cap_both_apply(self, store: SQLiteMemoryStore) -> None:
        conv = store.start_conversation()
        ancient = store.add_turn(conv.id, "user", "ancient")
        _backdate(store, ancient.id, days_ago=100)
        for i in range(15):
            store.add_turn(conv.id, "user", f"turn {i}")

        report = apply_retention_policy(
            store, MemorySettings(retention_days=30, max_turns_per_conversation=10)
        )

        assert report.turns_deleted_by_age == 1  # the ancient one
        assert report.turns_deleted_by_cap == 5  # 15 remaining -> capped to 10
        assert store.stats().turn_count == 10
