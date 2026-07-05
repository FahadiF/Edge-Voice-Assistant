"""Memory store registry: id -> factory(settings, paths) (ADR-019)."""

from __future__ import annotations

from collections.abc import Callable

from eva.config.paths import AppPaths
from eva.config.settings import Settings
from eva.core.errors import RegistryError
from eva.core.registry import Registry
from eva.memory.base import MemoryStore, UserProfileStore

MemoryStoreFactory = Callable[[Settings, AppPaths], MemoryStore]

memory_store_registry: Registry[MemoryStoreFactory] = Registry("memory-store")


def _make_sqlite(_settings: Settings, paths: AppPaths) -> MemoryStore:
    from eva.memory import db
    from eva.memory.sqlite_store import SQLiteMemoryStore

    conn = db.connect(paths.conversations_dir / db.DB_FILENAME)
    return SQLiteMemoryStore(conn)


def register_builtins() -> None:
    if "sqlite" not in memory_store_registry:
        memory_store_registry.register("sqlite", _make_sqlite)


def create_memory_store(settings: Settings, paths: AppPaths) -> MemoryStore:
    register_builtins()
    return memory_store_registry.get(settings.memory.engine)(settings, paths)


def create_stores(settings: Settings, paths: AppPaths) -> tuple[MemoryStore, UserProfileStore]:
    """Build a `MemoryStore` and a matching `UserProfileStore` sharing one
    connection where the backend supports it (ADR-022) — today, the only
    registered engine is SQLite, so both ports back onto one database file
    via one connection. A future second engine would extend this function,
    not change any caller of it."""
    register_builtins()
    if settings.memory.engine != "sqlite":
        raise RegistryError(
            f"No user-profile store paired with memory engine '{settings.memory.engine}'"
        )
    from eva.memory import db
    from eva.memory.sqlite_store import SQLiteMemoryStore, SQLiteUserProfileStore

    conn = db.connect(paths.conversations_dir / db.DB_FILENAME)
    return SQLiteMemoryStore(conn), SQLiteUserProfileStore(conn)
