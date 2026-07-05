"""Persistent, searchable conversation memory (ADR-019, ADR-020).

Ports: ``MemoryStore`` (persistence + management verbs), ``MemoryRetriever``
(semantic search), ``Summarizer``. The first adapter is SQLite
(``sqlite_store.py``), one database file per ``AppPaths.conversations_dir``.
"""

from __future__ import annotations
