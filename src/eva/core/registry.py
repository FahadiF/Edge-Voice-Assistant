"""Generic id-keyed registry — the single extension mechanism of the platform.

Every swappable concept (engines, models, personas, prompt templates, hardware
profiles, tools, plugins) is registered in a `Registry` under a string id and
resolved at runtime. Core code never names concrete implementations; the UI,
model manager, and benchmark suite enumerate the same registries users see.

Thread safety: registrations happen at startup and on plugin (un)load, lookups
happen from any thread — all operations take the internal lock, and iteration
methods return snapshots.
"""

from __future__ import annotations

import threading

from eva.core.errors import RegistryError


class Registry[T]:
    """An id → item map with explicit registration semantics."""

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._items: dict[str, T] = {}
        self._lock = threading.Lock()

    @property
    def kind(self) -> str:
        return self._kind

    def register(self, item_id: str, item: T, *, replace: bool = False) -> None:
        """Register `item` under `item_id`.

        Duplicate ids raise unless `replace=True` — silent replacement hides
        plugin conflicts, which must surface to the user instead.
        """
        if not item_id:
            raise RegistryError(f"{self._kind}: empty registry id")
        with self._lock:
            if item_id in self._items and not replace:
                raise RegistryError(f"{self._kind}: '{item_id}' is already registered")
            self._items[item_id] = item

    def unregister(self, item_id: str) -> None:
        """Remove a registration (used on plugin disable/unload)."""
        with self._lock:
            if item_id not in self._items:
                raise RegistryError(f"{self._kind}: cannot unregister unknown id '{item_id}'")
            del self._items[item_id]

    def get(self, item_id: str) -> T:
        with self._lock:
            try:
                return self._items[item_id]
            except KeyError:
                known = ", ".join(sorted(self._items)) or "<none>"
                raise RegistryError(
                    f"{self._kind}: unknown id '{item_id}' (registered: {known})"
                ) from None

    def ids(self) -> list[str]:
        with self._lock:
            return sorted(self._items)

    def snapshot(self) -> dict[str, T]:
        with self._lock:
            return dict(self._items)

    def __contains__(self, item_id: str) -> bool:
        with self._lock:
            return item_id in self._items

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)
