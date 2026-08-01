# ADR-019: Memory subsystem — ports and SQLite storage

Status: Accepted · Date: 2026-07-05

## Context

Conversation memory today (`eva.conversation.history.ConversationHistory`) is
an in-process list capped at `max_history_turns`: lost on restart, no search,
no per-turn identity beyond position in the list. `ARCHITECTURE.md`'s module
tree and ADR-010 §2 already anticipated a `memory/` subsystem with a
`MemoryStore` port and "(future) embeddings" as a registered kind — M4 is
where that gets built. The roadmap's M4 exit criterion is persistent
multi-session memory; the milestone brief additionally asks for search,
scoring, retention, export/import, and full API exposure, designed to grow
for years without a rewrite.

## Decision

1. **New subsystem package `eva/memory/`** (ADR-010 pattern): `base.py`
   (ports), `models.py` (pydantic records), `db.py` (shared connection +
   schema/migrations), `sqlite_store.py` (the first adapter), `registry.py`,
   `retriever.py`, `summarizer.py`, `retention.py`.

2. **Ports, not one facade.** `MemoryStore` (CRUD + query + management
   verbs), `MemoryRetriever` (semantic search — ADR-020), `Summarizer`
   (ADR-020/§9). Your example list also named `MemoryProvider` and
   `MemoryIndexer` — both are folded into `MemoryStore` rather than kept
   separate:
   - `MemoryProvider` would be a facade in front of `MemoryStore` with
     nothing to vary independently of it — every other capability in this
     codebase (ASR/LLM/TTS/VAD) has exactly one port per capability, not a
     port plus a facade in front of it. Adding one here would be the "hidden
     coupling for no reason" the milestone brief explicitly warns against.
   - `MemoryIndexer` is the FTS index — an implementation detail of
     `SQLiteMemoryStore`, not a concern any caller ever needs to swap
     independently of the store itself.
   - `MemoryCleaner`/`MemoryExporter`/`MemoryImporter` become methods on
     `MemoryStore` (`export_json`, `import_json`) plus a standalone
     `retention.py` function, `apply_retention_policy(store, settings)`.
     These are operations *on* one store, not alternate implementations of
     a capability — nothing in this design will ever have two different
     "exporters" for the same store the way there are two different TTS
     engines. A registry per verb would fragment one cohesive concern into
     ceremony with no payoff.

3. **One SQLite database file**, `conversations_dir/memory.db` (the
   directory already exists — `AppPaths.conversations_dir` — and is already
   gitignored). Chosen over per-table files: one file gives transactional
   consistency across turns/embeddings/summaries/profiles for free, and
   matches how comparable personal-data apps (note-taking, chat clients)
   are built at this scale. WAL journal mode for concurrent read/write
   safety (the orchestrator writes after each turn while an API request
   could read concurrently).

4. **Schema** (full DDL in `eva/memory/db.py`, numbered migrations applied
   in order at store init — same pattern as `SETTINGS_SCHEMA_VERSION`):

   ```
   schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT)
   conversations(id TEXT PRIMARY KEY, started_at TEXT, title TEXT,
                 language TEXT, archived INTEGER DEFAULT 0)
   turns(id INTEGER PRIMARY KEY AUTOINCREMENT,
         conversation_id TEXT REFERENCES conversations(id),
         created_at TEXT, speaker TEXT CHECK(speaker IN ('user','assistant')),
         text TEXT, language TEXT, metadata TEXT,  -- JSON
         pinned INTEGER DEFAULT 0, favorite INTEGER DEFAULT 0,
         deleted INTEGER DEFAULT 0)
   turns_fts  -- FTS5 virtual table mirroring turns.text, or absent (see 5)
   embeddings(turn_id INTEGER PRIMARY KEY REFERENCES turns(id),
              model_id TEXT, vector BLOB, dim INTEGER, created_at TEXT)
   summaries(id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT,
             turn_range_start INTEGER, turn_range_end INTEGER, text TEXT,
             created_at TEXT, model_id TEXT)
   user_profiles(id TEXT PRIMARY KEY, nickname TEXT, preferred_language TEXT,
                 preferred_voice TEXT, preferred_llm_model TEXT,
                 conversation_style TEXT, units TEXT, timezone TEXT,
                 extra TEXT, created_at TEXT, updated_at TEXT,
                 active INTEGER DEFAULT 0)
   ```

   `metadata` is a JSON text column on `turns` specifically so `attachments`
   (explicitly future work in your brief) needs no migration when it lands —
   it can start as a metadata key and graduate to its own table later without
   breaking anything reading the JSON blob today. `deleted`/`archived` are
   soft-delete flags: `forget()` and `delete()` are different operations
   (forget is permanent per the privacy requirement; delete/archive are not).

5. **Text search: FTS5 with a LIKE fallback.** Confirmed available in this
   environment (`sqlite3.sqlite_version` 3.43.1, `CREATE VIRTUAL TABLE ...
   USING fts5` succeeds), but FTS5 is a compile-time SQLite option that not
   every platform's Python ships. `SQLiteMemoryStore.__init__` probes for it
   once, logs a warning and falls back to a `LIKE '%...%'` query plan if
   absent — the same graceful-degradation shape already used for GPU
   detection (never a hard failure over an optional capability).

## Rationale

Ports-and-adapters keeps `MemoryStore` swappable (a future Postgres or
remote store is a new adapter, not a rewrite) while giving the orchestrator
and API exactly one thing to depend on, matching every other subsystem in
this codebase. Consolidating the finer-grained interfaces from the example
list avoids inventing distinctions that don't correspond to anything this
product will ever actually swap independently — over-fragmenting an
interface is exactly the kind of premature abstraction the engineering
guidelines warn against, even though the brief's ask for "production-quality
architecture that can grow for years" might read as a request for more
interfaces. Years-long growth is served by *correct seams* (store vs.
retriever vs. summarizer, which really are independently replaceable), not
by maximizing interface count.

## Consequences

- The orchestrator's `self._history: ConversationHistory` becomes
  `self._memory: MemoryStore` (ADR-021 covers the composition layer that
  replaces `ConversationHistory.messages()`).
- `eva.memory` depends on `eva.embedding`'s port (ADR-010 amendment) and on
  `eva.llm`'s port (for `LLMSummarizer`, ADR-020 §9) — both are genuine
  building-block relationships, documented exceptions to the "subsystems
  import only core/config" rule.
- A future encrypted-at-rest adapter (`SQLCipherMemoryStore`) is a drop-in
  `MemoryStore` implementation, not a redesign — deferred, not implemented,
  until a compiled dependency decision is explicitly wanted (see CHANGELOG).
