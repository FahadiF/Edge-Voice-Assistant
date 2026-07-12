# Changelog

Notable changes to Edge Voice Assistant. The format follows
[Keep a Changelog](https://keepachangelog.com/); versioning follows SemVer from the
first release onward.

## [Unreleased]

### 2026-07-12 — M5.7: Final UX & Windows polish

The last polish pass before M6: the microphone control does something real,
the background server behaves like a native Windows app (no flashing
consoles, no scary shutdown logs), and a fully-installed EVA no longer
touches the network at startup.

**Fixed**
- **Flashing console windows** on `eva start`: `sample_resources()` runs
  `nvidia-smi` on every diagnostics snapshot, and from a detached
  (console-less) server each call made Windows allocate — and flash — a
  console window. All external probes now go through
  `eva.core.proc.no_window_kwargs()` (`CREATE_NO_WINDOW` on Windows).
- **Faster-whisper hit Hugging Face on every startup**: even with the model
  cached, huggingface_hub makes a HEAD request per file to check for updates
  unless told the files are local. The adapter now attempts a fully-offline
  load first and only permits the network when the model isn't cached yet —
  a fully-installed EVA starts with zero network calls (verified: the load
  log gains an "offline" marker and the preceding HF request disappears).
- **`eva serve` shutdown was noisy and slow with the web UI open**: an open
  WebSocket blocks in `queue.get()`, so uvicorn's graceful pass timed out
  and logged `ERROR: Cancel N running task(s), timeout graceful shutdown
  exceeded` after a multi-second wait. `EventBus.close()` now pushes a
  `STREAM_CLOSED` sentinel to every subscriber, and a `uvicorn.Server`
  subclass calls it at the *start* of shutdown — WS handlers return
  immediately. Shutdown with 3 sockets held open went from ~7.8 s (with the
  error) to ~1.9 s, clean.
- **Abrupt WebSocket drops logged stack traces**: a closed browser tab
  surfaces as `ConnectionResetError` / WinError 10054 on the next send. That
  is a normal disconnect and is now debug-logged, never a traceback.

**Added**
- **Functional microphone button**: the Composer mic button is now a real
  mute/unmute toggle when the engine is running with microphone permission
  (🎙 ↔ 🔇). Muting drops captured-speech events at the orchestrator door —
  the assistant stops listening while typed chat and playback keep working,
  and the audio device stays open so echo cancellation is unaffected. When
  microphone permission is off the button is disabled with a tooltip
  pointing to Settings (honest, not a no-op). New `POST
  /api/v1/conversation/microphone`, `MicrophoneMuted` event, and
  `microphone_available` / `microphone_muted` fields on the runtime
  snapshot; interrupting a reply lives solely on the ⏹ Stop button now.
- `eva.core.proc` — a tiny shared home for the no-console-window subprocess
  kwargs.

**Changed**
- README and `docs/INSTALLATION.md` now split running the server into two
  clearly-labelled workflows: **Development** (`eva serve` → Ctrl+C) and
  **Background/production** (`eva start` / `stop` / `restart` / `status`).

### 2026-07-12 — M5.6: Final hardening, UX & production readiness

The last M5 milestone: everything M5 promised now behaves like a finished
product — conversations can be continued, shutdown is bounded and clean,
downloads are integrity-verified, and the remaining trust boundaries are
closed. No new capabilities; M6 (desktop) starts from here.

**Fixed**
- `eva serve` Ctrl+C no longer hangs while a web UI tab is open: uvicorn
  ran with an unbounded graceful-shutdown wait, and the UI keeps a
  WebSocket connected for its whole lifetime — shutdown now runs with
  `timeout_graceful_shutdown=5`, so exit is bounded (≤ ~5 s worst case,
  immediate when idle) and always traceback-free.
- Microphone permission OFF wedged every typed turn in the "speaking"
  state forever: audio startup was skipped entirely, so nothing drained
  the playback queue. Mic-off now opens a playback-ONLY stream (the input
  device is never touched — the permission means what it says) and typed
  conversations speak normally (`DuplexAudioStream.start(playback_only=)`).
- Non-English TTS pronunciation: the conversation language was never
  passed to Kokoro, so every reply was phonemized as US English — Spanish
  text through an English G2P is why it sounded wrong. The TTS port now
  carries `language`, and the Kokoro adapter maps it to the matching
  espeak phonemizer voice (`es`, `de`, `fi`, `sv`, …).
- Models page layout: buttons and long model ids overflowed their cards
  (grid items default to `min-width:auto`); cards now clamp and wrap
  (`min-width: 0`, `overflow-wrap`, wrapping action rows) — applied
  systemically to all `.grid-2`/`.grid-3` children.
- Stale documentation: model-catalog count corrected to 10 (was "9" in two
  places), an unedited thinking-aloud sentence removed from `HANDOFF.md`,
  milestone naming aligned (M4.5), and `ARCHITECTURE.md`'s ≤1.2 s
  first-audio target now carries the measured ~2.0 s reality and the M7
  lever (WASAPI).

**Added**
- **Continue a conversation** (ChatGPT-style): `POST
  /api/v1/conversation/resume` switches the engine back to any stored
  conversation — same id, context, summary, and title; the next message
  continues it. The Memory page's conversation list gains a primary
  "Continue" button that reopens the conversation on the Conversation
  page. New `MemoryStore.get_conversation()` port method.
- **Graceful process shutdown**: `POST /api/v1/system/shutdown` stops the
  engine then exits uvicorn via a registered hook. `eva stop` uses it
  first and only falls back to terminating the process — on Windows,
  terminate is a hard `TerminateProcess` with zero cleanup, so the API
  call is the only genuinely graceful stop for a background server.
- **Download integrity verification**: `ModelFile` carries the publisher's
  exact `size_bytes` and (where the publisher exposes one — all Hugging
  Face LFS files) `sha256`. Downloads are verified after completion; a
  checksum or size mismatch discards the file and fails loudly. The
  pre-M5.6 hole where a response without `Content-Length` skipped
  verification entirely is closed.
- **WebSocket origin policy** (`eva.server.security`): CORS middleware
  does not apply to WebSocket handshakes, so `/api/v1/ws` validated
  nothing — any website could have read live transcripts. Browser origins
  are now checked against the same localhost-only policy (foreign origins
  rejected with close code 1008; header-less non-browser clients still
  connect).
- **Time-to-first-audio cuts**: Kokoro warm-up synthesis at load (moves
  onnxruntime's first-inference kernel initialization out of the first
  reply, and it runs on the preload worker in parallel with the LLM load —
  free in wall-clock); the sentence chunker's FIRST segment may now end at
  a clause break (comma/semicolon/colon), so "Sure, let me check that."
  starts speaking after "Sure," while the rest synthesizes
  (`first_sentence_min_chars` default 6 → 4 so short openers qualify).
- SQLite thread-safety: both stores share one connection
  (`check_same_thread=False`) reached from orchestrator worker threads and
  API handlers at once — every public store method now holds a shared
  re-entrant lock, making each method one atomic critical section (WAL
  only isolates *separate* connections).
- Composer polish: the "+" menu dismisses on outside-click/Escape and
  each entry is labeled "(coming soon)" up front; attachment chips remain
  honest placeholders pending Vision support.

**Changed**
- `eva serve` uses a programmatic `uvicorn.Server` (required for both the
  bounded shutdown and the exit hook) and prints how to stop it.
- Multilingual reality documented honestly: EVA understands and answers
  in all six registered languages; a *native* voice exists only for
  English and Spanish (Kokoro has none for Finnish/Swedish/Bengali — a
  model gap, not an architecture gap). Automatic language detection and
  per-language TTS engines remain M7+ research (see ROADMAP).

### 2026-07-11 — v0.5 documentation synchronization

A maintenance pass, not a feature milestone: the repository's documentation
now matches the shipped M5.x state.

**Changed**
- Version bumped `0.1.0.dev0` → `0.5.0a1` (`pyproject.toml`,
  `eva.__version__`); development-status classifier raised to Alpha.
- `docs/ROADMAP.md` records M4 integration & validation, M5.1–M5.5, and
  this pass; M6 remains the next milestone.
- `README.md` updated with M5.1–M5.5 highlights and the current run
  commands (`eva start/stop/status/logs`).
- `docs/INSTALLATION.md` gains a "Web UI and background server" section
  and an up-to-date command reference.
- `docs/ARCHITECTURE_DIAGRAMS.md` updated to the M5.5 state: web UI and
  desktop shell no longer marked "not built"; memory/embedding subsystems,
  the M4 routers, static UI hosting, parallel preload, supervised
  recovery, and the process CLI are all reflected.

**Removed**
- Internal working notes moved out of `docs/` into an untracked `.dev/`
  folder; `.gitignore` simplified accordingly. Public docs are now
  `README`, `CHANGELOG`, and `docs/` only.

### 2026-07-06 — M5.5: Stability, lifecycle & performance (ADR-026)

The milestone that makes EVA behave like a real desktop application:
visible parallel startup, clean shutdown, fixed cancellation, owned
background tasks, supervised component recovery, and a process CLI.

**Fixed**
- TTS cancellation race: `_drive_stream` now gives each synthesis stream a
  single owner thread — a barge-in close is queued behind any in-flight
  pull (never `ValueError: generator already executing`), close runs even
  if the awaiting task is cancelled, and Kokoro's per-stream event loop is
  only ever touched from its creating thread. Kokoro cleanup hardened
  (guarded aclose/shutdown_asyncgens/close; `run_until_complete` provably
  never runs on a running loop).
- `eva serve` Ctrl+C: ordered, exception-proof engine teardown — no
  tracebacks.

**Added**
- Parallel preload with progress: LLM→ASR stay GPU-ordered (ADR-015), TTS +
  embedding load concurrently on CPU threads; new
  `ComponentLoadStarted/Finished` events drive a live startup checklist in
  the web UI (header button narrates the current component; Dashboard shows
  per-component ✓ + seconds).
- `tts.lazy_load` setting: skip TTS at startup, load on first spoken reply
  (voices API loads on demand).
- `eva.core.tasks.TaskManager`: named, owned background tasks with
  one-call cancel-all/await-all; adopted by the server (downloads) and
  orchestrator (barge-in measurements, recoveries).
- Supervised component recovery: an ASR crash costs one turn, a TTS crash
  one sentence — the engine reloads the component in the background
  (cooldown-guarded); a WebSocket disconnect never affects the engine
  (regression-tested).
- Process lifecycle CLI: `eva start` / `stop` / `restart` / `status` /
  `logs` — PID-file management over `eva serve` with graceful termination
  and stale-PID detection.
- Composer: ⏹ Stop button beside mic/send while the assistant is
  thinking/speaking (moved out of the page header); mic button verified
  (start engine when stopped, interrupt when speaking).

**Tests** — stream-ownership cancellation (close-during-pull), preload
progress/ordering/lazy/failure (4), component recovery incl. cooldown (3),
WS-disconnect resilience, service lifecycle (11), composer Stop button (2).

### 2026-07-06 — M5.4: Final integration, UX polish & production readiness

**Fixed — long-term memory finally works end-to-end (ADR-020 Amendment 2)**
- Root cause 1: nothing in the live pipeline ever embedded new turns (only
  the benchmark called `store_embedding`), so semantic retrieval always
  scanned an empty set. The orchestrator now embeds both sides of every
  exchange at write time.
- Root cause 2: without the embedding model installed, `ContextBuilder`
  returned NO memories at all. New keyword fallback (per-salient-word FTS
  search, merged) makes recall degrade gracefully instead of vanishing.
- Acceptance case pinned: "My nickname is Fahad" → new conversation →
  "What's my nickname?" → Fahad.
- Speech-filter regression fixed: a bare `***` run could leave a lone
  spoken asterisk (emphasis regex treated it as *-wrapped-*); content may
  no longer itself be a marker (CommonMark-consistent).

**Added**
- Conversation titles (M5.4 §2): auto-generated by the LLM after the first
  exchange (16-token generation masked by TTS playback), editable via
  `PATCH /memory/conversations/{id}`, `eva memory rename`, and inline ✎ in
  the Memory page; stored permanently; export/import round-trips them.
- Permissions regrouped (ADR-025 amendment): General / Files / Devices /
  Tools / Privacy with clearer toggles, three of them now genuinely
  enforced — system-info prompt gating, `devices.microphone` (off =
  typed-chat-only assistant), `privacy.remember_conversations` (off =
  nothing stored; replaces the dead `conversation.memory_enabled` flag).
  Settings schema v2 with in-memory migration of v1 documents. SchemaForm
  renders nested groups (one-level $ref resolution).
- Memory page UX: per-conversation Export, Import button, titles column
  with inline rename, "Search memories" + clearer placeholder, Context
  Inspector moved above the list with a real explanation.
- Conversation UX: sticky always-visible composer (transcript scrolls
  independently), animated thinking indicator before the first token,
  functional mic button (starts engine / interrupts), "Vision support
  coming soon" phrasing for attachments, "Online (coming soon)" mode label.

**Removed (backend review)**
- Dead `conversation.memory_enabled` setting (migrated) and the
  never-called `effective_system_prompt()` helper.

### 2026-07-06 — M5.3: Final UX & capability polish

**Fixed**
- Markdown-to-TTS hardened (the "asterisk asterisk Generate" leak): the
  speech filter now decodes HTML entities, unwraps nested emphasis to a
  fixpoint, handles intraword underscores per CommonMark
  (`file_name_here` survives), and — the actual bug — scrubs *unpaired*
  markers left when the sentence chunker splits inside an emphasis span.
  Contract: formatting characters are never spoken, paired or not.
  (+16 adversarial tests.)

**Added**
- `PermissionsSettings` (ADR-025): 15 toggles (date/time, timezone, locale,
  CPU, GPU, RAM, OS, internet, local files, camera, clipboard, browser,
  shell, Python, plugins). Read-only facts default on; acting capabilities
  default off and are the consent contract for future providers. Renders
  automatically in the Settings UI (schema-driven, ADR-009).
- System Information provider (`eva.conversation.system_info`): permission-
  gated local facts (fresh date/time each turn; cached hardware detection)
  injected into the prompt — "what time is it?" / "what GPU do I have?"
  now get real answers; a denied permission is attributed to the user's
  settings, never to permanent inability.
- Typed conversation: `POST /conversation/say` → `Orchestrator.submit_text()`
  — same event queue and turn pipeline as speech, minus ASR; replies stream
  and speak normally. `engine/start` now yields once so the orchestrator
  loop is bound before "started" is reported.
- ChatGPT-style composer on the Conversation page: Enter/Shift+Enter,
  + menu (image/document/screenshot placeholders — "not available in this
  build"), drag-and-drop and paste producing removable placeholder chips,
  live mic-state indicator, disabled-with-guidance when the engine is
  stopped.
- Offline/Online mode selector beside the engine controls (Online is a
  disabled placeholder for future providers).
- Empty-state guidance: Memory page explains how conversations get there;
  export/delete-all disabled (with tooltip) when there is nothing to act on;
  Conversation empty state adapts to engine state.

**Tests** — orchestrator text-turn (full pipeline, no ASR, supersede
semantics), `/conversation/say` (accept/409/422), system-info gating +
prompt integration (11), Composer (8), markdown hardening (16).

### 2026-07-06 — M5.2: Conversational intelligence & prompt engineering

Real-conversation testing showed the pipeline worked but the *conversation*
didn't: fragments ("with rows and columns.") were treated as new requests,
the assistant said "I cannot process images" (permanently) and "I am not a
spreadsheet" (unhelpfully), personas sounded identical, and the name got
repeated. Root cause was prompt engineering, not context selection — the
20-turn history window already contained everything needed (ADR-021
Amendment 3).

**Changed**
- System-prompt hierarchy rebuilt (`context_builder.py`): one-sentence
  identity (name used only when asked) → shared conversation guidance
  (fragments/pronouns continue the topic; user's goal over
  self-description; anything expressible in text can be produced;
  ambiguity → helpful assumption or one short question) → capability
  honesty ("not enabled in this build", never "impossible") → persona
  style → language/profile. Conversation summary now precedes retrieved
  memories; technical backend facts moved to the last (least salient)
  section.
- Memory block reframed: "You remember these things … use them naturally,
  don't announce that you are recalling them" (was recital-inducing
  "Potentially relevant earlier context:").
- All six built-in persona prompts rewritten from one-liners into
  substantial, mutually distinct style instructions; new **teacher**
  persona (analogies, step-by-step, checks understanding).

**Validated live against the real model** — all previously-failing
scenarios now pass: fragment extends the table; "how tall is it?" resolves
the pronoun; image question gets a build-scoped answer; "act as a
spreadsheet" computes the sum; ordinary replies never name-drop; minimal
vs teacher personas are unmistakably different.

**Tests** — new `tests/test_conversation_quality.py` (16): prompt
hierarchy/order, identity-appears-once, continuity/helpfulness/capability
guidance, fragment+pronoun antecedents in the message list, 20-turn
window, memory-block ordering + phrasing, persona pairwise distinctness.

**Docs** — ADR-021 Amendment 3; MANUAL_TESTING §15 (conversational
evaluation: continuity, pronouns, helpfulness, capability messaging,
identity, personas, memory naturalness, long conversations, ambiguity).

### 2026-07-05 — M5.1: Markdown presentation layer + review fixes

A senior review pass over M5, plus the fix for a UX bug found in manual
testing: the UI showed raw Markdown and the TTS spoke formatting characters.

**Added — Markdown presentation layer (ADR-024)**
- `eva/conversation/markdown.py`: `MarkdownSpeechFilter` converts Markdown
  to speakable text at the *only* LLM→TTS boundary (orchestrator speak
  worker). Stateful: code-fence suppression carries across sentence
  segments (a fence's ``` markers arrive in different segments under
  streaming). Formatting markers removed, links→text, tables→comma-joined
  cells, fenced code content skipped. Storage, events, API, export, memory,
  and summaries keep raw Markdown canonical — verified by a new orchestrator
  test asserting the stored/emitted reply keeps `**`/`` ` `` while the
  spoken text does not.
- Web UI renders assistant messages with `react-markdown` + `remark-gfm`
  (bold/italic/headings/inline+fenced code/blockquotes/lists/tables/links);
  raw HTML disabled; fenced blocks get a Copy button. User messages stay
  plain text.
- Tests: `tests/test_markdown_speech.py` (32 cases incl. malformed input),
  `web/src/components/Markdown.test.tsx` (12 cases incl. HTML-injection
  guard).

**Fixed (review findings)**
- WebSocket resilience: `EngineStarted`/`EngineStopped` and post-reconnect
  snapshots now invalidate REST caches (were only picked up on a 5 s poll);
  `stopWebSocket()` detaches handlers before closing so a deliberate stop
  can't schedule a zombie reconnect (StrictMode-safe).
- Conversation import no longer races the history refetch (refetch → reseed,
  in order); transcript keys can't collide.
- User-profile import validates the file shape and reports per-item
  failures instead of silently continuing.
- `postBinary` parses the standard `{detail, error_type}` error shape like
  every other call (shared `throwApiError`).
- Settings section-switch with unsaved changes uses `ConfirmDialog`, not
  `window.confirm`.
- Shared `downloadJson()` replaces three copies of blob-download code.
- Accessibility: `aria-busy` on voice-preview buttons, `aria-readonly` +
  explanation on the disabled persona-id field.

### 2026-07-05 — M5: Web UI & Desktop Shell

The first full consumer of the platform API (ADR-017): a production-quality
React + TypeScript web UI, plus a minimal `pywebview` desktop shell landing
a milestone early (ADR-023).

**Added**
- `web/`: React 18 + TypeScript (strict) + Vite frontend. TanStack Query for
  REST, one zustand store for the WebSocket event stream, react-router,
  hand-rolled accessible components (native `<dialog>`, ARIA), CSS-custom-
  property theming (dark/light/system, driven by `settings.ui.theme`).
- Ten pages: Dashboard, Conversation, Memory, Personas, Users, Models,
  Voices, Settings, Diagnostics, Plugins — one per M5 part, each a pure
  client of the existing REST/WebSocket API (no backend logic duplicated).
- `Settings` page is fully schema-driven (ADR-009): a `SchemaForm` component
  renders every section/field/bound/description from `GET /settings/schema`
  — nothing hardcoded.
- `src/eva/server/static.py` (ADR-023): serves the built UI as an SPA at `/`
  when a build exists (env override → packaged dir → `web/dist`); the API
  is byte-for-byte unchanged when no build is present.
- `src/eva/desktop.py` (`eva-desktop`, optional `[desktop]` extra): starts
  the same FastAPI app on a background thread and opens one native
  `pywebview` window at it — no tray/hotkey/supervision/installer (M6).
- `eva serve --open`: opens the built UI in the default browser.
- ADR-023 (web UI architecture and hosting).
- `docs/MANUAL_TESTING.md` §14: step-by-step validation for every page.
- CI: a Node job (`npm ci && lint && build && test`) alongside the
  existing Python job.

**Testing**
- Frontend: 26 vitest tests (WebSocket store reducers including epoch-
  discipline drops, `SchemaForm` against a real captured schema fixture,
  API client error handling), ESLint clean, `tsc -b` clean, production
  build verified.
- Backend: new `tests/test_server_static.py` (SPA mount/fallback/path-
  escape safety) and `tests/test_desktop.py` (free-port allocation, health
  polling, window launch — `pywebview` mocked since it's an optional
  extra). Full existing suite stays green.
- Manual: built UI served by the real backend against the real installed
  models (`qwen3.5-4b-instruct-q4_k_m`, Kokoro) — engine start, live
  dashboard, settings round-trip, voice preview decode, model catalog,
  memory browsing, and context-preview all verified end-to-end.

### 2026-07-05 — Critical fix: multiple system messages crashed real conversations

Real-hardware testing (after the integration pass below) found `eva run`
failing on the first turn with `ValueError: System message must be at the
beginning.` from llama.cpp's Qwen chat template. Root cause: `ContextBuilder`
emitted identity, technical facts, retrieved memories, and the conversation
summary as up to four separate `system`-role messages — every chat template
(Qwen, Llama, Mistral) requires exactly one, first. Every M4/integration-pass
unit test used mocked messages and never caught this because none exercised
a real chat-template engine.

**Fixed**
- `ContextBuilder` now merges identity, persona, language, profile
  preferences, technical facts, retrieved memories, and the summary into
  **one** system message, always `messages[0]`; no other message may be
  `system` (ADR-021 Amendment 2).
- Added `eva.llm.base.validate_chat_messages()` — a model-agnostic guard
  (no Qwen-specific logic) enforcing "one system message, first, then
  strict user/assistant alternation," called on every `ContextBuilder.build()`.
- Added `ContextBuilder._normalize_alternation()` to merge any adjacent
  same-speaker turns from storage (e.g. a malformed import, a dangling
  unanswered turn) before validation, so malformed history degrades
  gracefully instead of crashing the turn.
- New `tests/test_llm_chat_validation.py`; `tests/test_context_builder.py`
  gained alternation-normalization and single-system-message tests covering
  the exact memory+summary+history combination that triggered the failure.

### 2026-07-05 — M4 Integration & Validation Pass

Manual testing after M4 shipped found that its subsystems, while fully
built and tested, weren't actually reachable through the runtime: the
assistant introduced itself using the underlying LLM's identity, personas
and user profiles had no CLI, and the active persona/profile/voice weren't
visible anywhere at runtime. This pass closes those gaps without changing
any of M4's underlying design.

**Fixed**
- Assistant no longer leaks the underlying LLM's identity — a fixed
  identity preamble in `ContextBuilder` establishes "Edge Voice Assistant"
  regardless of persona; a separate technical-facts system message lets it
  answer honestly *only* when explicitly asked a technical question (ADR-021
  amendment).
- `settings.conversation.active_profile_id` is now actually written when a
  profile is activated (API and CLI) — previously dead, always-stale data.

**Added**
- CLI parity for every M4 capability: `eva personas` (list/show/create/
  delete/use), `eva users` (list/show/create/edit/activate/delete), `eva
  voices` (list/preview/use), `eva memory` (stats/list/show/search/forget/
  pin/favorite/archive/delete-conversation/merge/export/import/delete-all/
  summarize), `eva profile` (active-user-profile shortcut), and `eva run
  --persona` for one-session overrides.
- Startup banner (`eva run`) and `eva serve` now print the active persona,
  user profile, voice, and memory stats. `RuntimeSnapshot` gained
  `active_persona_id`, `active_profile_id`, `active_voice`.
- [`docs/MANUAL_TESTING.md`](docs/MANUAL_TESTING.md): step-by-step
  end-to-end validation guide covering every M4 acceptance item.

### 2026-07-05 — M4: Memory, Personalization & Intelligence

A large new subsystem: persistent conversation memory, semantic search,
deterministic context composition, personas, user profiles, and voice
metadata — the shift from "conversational assistant" to "personal
assistant" the milestone brief asked for. Four new ADRs document the
architecture decisions (docs-first, per instruction) before any of this was
implemented; one of them (ADR-020) was amended mid-milestone after a real
benchmark run exposed a design assumption that didn't hold up.

**Added**
- `eva/memory/` subsystem (ADR-019): `MemoryStore` + `UserProfileStore`
  ports, `SQLiteMemoryStore`/`SQLiteUserProfileStore` adapter sharing one
  connection, one database file (`conversations_dir/memory.db`, WAL mode,
  numbered migrations, FTS5 full-text search with a `LIKE` fallback when
  FTS5 isn't compiled into the platform's SQLite). Schema: `conversations`,
  `turns` (speaker-granular, JSON metadata column reserved for future
  attachments), `embeddings`, `summaries`, `user_profiles`.
- `eva/embedding/` subsystem (ADR-020, and an ADR-010 amendment permitting
  this one capability-on-capability dependency): `EmbeddingProvider` port,
  `OnnxEmbeddingProvider` running `all-MiniLM-L6-v2` via `onnxruntime` +
  `tokenizers` (no PyTorch — ADR-012 stays intact). New `kind="embedding"`
  model-catalog entry, downloaded/verified through the existing
  `ModelManager` — no new install mechanism.
- Semantic retrieval: `NumpyMemoryRetriever` — brute-force cosine similarity
  (no vector-database dependency), blended with recency decay and
  pinned/favorite importance boosting. Searches across every conversation,
  not just the active one (recalling *past* sessions is the point of
  persistent memory). Bounded by a new `retrieval_scan_limit` setting
  (default 2000 candidates) so latency stays flat regardless of how much
  history has accumulated — confirmed by real measurement, see Benchmarks.
- `ContextBuilder` (ADR-021): deterministic prompt composition — persona +
  language + user-profile preferences → relevant memories → latest
  conversation summary → recent-turn window → current utterance. Every
  build returns a `ContextTrace` (what was retrieved, scores, what was
  trimmed for budget) for inspection without spending a generation on it.
  Replaces the old in-process `ConversationHistory` entirely.
- Personas (ADR-022): `eva/conversation/personas.py`, mirroring the existing
  language-profile registry pattern. Six built-ins (Default, Professional,
  Friendly, Technical, Minimal, Creative); custom personas persist in
  `settings.json` (configuration, not conversation data) and register
  alongside the built-ins at startup.
- User profiles (ADR-022): nickname, preferred language/voice/model,
  conversation style, units, timezone — SQLite-backed (not settings-based),
  designed to extend to multiple users without redesign. Named "user
  profile" throughout, deliberately distinct from the pre-existing
  hardware/model "profile" concept (`eva profiles`, `Settings.profile`).
- Voices (ADR-022): `eva/tts/voices.py` — a registry over each TTS engine's
  existing `voices()` capability discovery, enriched with best-effort
  metadata (Kokoro's `{lang}{gender}_{name}` id convention parsed for
  display name/language/gender; unrecognized ids fall back to the bare id).
  Preview reuses the already-loaded engine's `synthesize()` — no new
  synthesis path.
- `LLMSummarizer` (ADR-019 §9): reuses the existing `LLMEngine` port to
  summarize a conversation's turn range — no new ML dependency. Summaries
  are additive; originals are never deleted.
- Retention policy (`eva/memory/retention.py`): age-based and
  per-conversation turn-count caps, both settings-driven, both skip pinned
  turns.
- `RuntimeSnapshot` gains `memory_enabled`, `memory_turn_count`,
  `memory_db_size_bytes`, `memory_embedding_count`, `last_retrieval_ms`,
  `last_retrieval_score_top1` — additive, same pattern as M3's diagnostics
  fields.
- Four new FastAPI routers, all ADR-017-compliant and additive to the
  existing API: `/api/v1/memory` (search/forget/pin/favorite/archive/
  merge/export/import/summarize/stats/context-preview), `/api/v1/personas`,
  `/api/v1/users`, `/api/v1/voices`. New `MemoryStoreError`/
  `MemoryNotFoundError` in the error hierarchy, mapped to HTTP 500/404.
- New base dependencies: `onnxruntime`, `tokenizers` (both universal
  wheels, no compiler required — ADR-013's preferred pattern).

**Fixed (found by measurement, not inspection)**
- An N+1 query pattern in `NumpyMemoryRetriever`: `MemoryStore.get_turn()`
  was called once per *candidate* embedding, not once per *result* —
  `MemoryStore.get_turns()` (bulk fetch) replaced it.
- `ContextBuilder` originally scoped semantic retrieval to the active
  conversation only, which doesn't actually recall *past* conversations —
  the stated point of persistent memory (ADR-020 amendment). Now searches
  globally by default.
- `MemoryStore.embeddings_for()`'s `limit` parameter originally ordered by
  `created_at` (unindexed), which forced SQLite to fully sort every
  candidate before the limit could apply. Reordered to `embeddings.turn_id`
  (that table's own primary key, already indexed, monotonic with insertion
  order) — SQLite's planner can now stop early instead of sorting.

**Benchmarks** (`eva.benchmark.memory.run_memory_benchmark`, real
measurements, many-conversation realistic distribution, RTX 3060 Laptop
reference machine)

| Total turns | Keyword search (FTS) | Semantic retrieval | Context composition |
|---|---|---|---|
| 100 | 0.67 ms | 2.08 ms | 2.54 ms |
| 1,000 | 2.77 ms | 28.24 ms | 28.95 ms |
| 5,000 | 10.64 ms | 59.16 ms | 60.83 ms |
| 20,000 | 39.67 ms | 59.02 ms | 60.91 ms |
| 50,000 | 101.09 ms | 60.54 ms | 60.84 ms |

Semantic retrieval and context composition plateau once total history
exceeds the 2000-candidate scan limit — flat latency regardless of years of
accumulated history. Keyword (FTS) search still scales with total turns;
acceptable since it's a deliberate, user-initiated search action, not
something in the live conversation turn's critical path.

**Not implemented (documented, not silently dropped)**
- CLI parity (`eva memory ...`, `eva user ...`) — the milestone asked for
  API exposure ("No UI yet"), not CLI commands; `cli.py` is already flagged
  as oversized technical debt. Candidate for M5 or an early fast-follow.
- Real encryption-at-rest — `MemorySettings.encrypt_at_rest` exists as a
  documented, honestly-inert flag; the `MemoryStore` port means a future
  `SQLCipherMemoryStore` adapter is a drop-in swap when a compiled
  dependency for it is explicitly wanted.

**Tests**
- +171 tests (462 total): SQLite schema/migrations/CRUD/search/management-
  verbs/corruption-recovery/concurrent-access/long-conversation, ONNX
  embedding provider (mocked tokenizer/session), semantic retrieval
  accuracy on synthetic vectors with known nearest neighbors, retention
  policy, context builder composition/trace/budget-trimming, personas,
  voice-id parsing, summarizer, and all four new API routers via FastAPI's
  `TestClient`.

### 2026-07-04 — M3: Natural Voice Conversation

A latency/interruption-quality milestone, not a feature milestone. Pipeline
inspection (see `docs/adr/ADR-018-tts-streaming-synthesis.md`) found the
~0.9-2.0 s time-to-first-audio was dominated by Kokoro synthesizing an entire
sentence before any audio reached the speaker — the same call was also the
largest gap in barge-in responsiveness, since it gave the pipeline no
cancellation checkpoint mid-synthesis.

**Added**
- ADR-018 + `TTSEngine.synthesize_stream()`: an additive, non-abstract port
  method (default: one chunk via `synthesize()`, so every existing adapter is
  unaffected). `KokoroTTS` overrides it via kokoro-onnx's native
  `create_stream()`, bridging its async generator to a dedicated event loop
  per call. The orchestrator's `speak_worker` now plays audio chunk-by-chunk,
  checking turn-epoch staleness between chunks — closing the single largest
  gap in "interruption while TTS is generating."
- `conversation.first_sentence_min_chars` setting (default 6, vs. 12 for
  later sentences): `SentenceChunker` gains an optional `first_chunk_min_chars`
  override so the first sentence of a turn is spoken sooner.
- `BargeInLatencyMeasured` event: detected-to-silent latency for every
  barge-in, measured fire-and-forget so the measurement itself never delays
  handling the next event (e.g. a second rapid interruption).
- `SpeechFinished` event (defined since M2, never published) now emitted when
  an utterance ends.
- Bounded, backpressured token/sentence queues in the orchestrator (real
  blocking `put()`/`run_coroutine_threadsafe` backpressure, not `put_nowait` +
  `QueueFull`) — no unnecessary buffering, no crash under a pathological
  long reply or a stalled consumer.
- `RuntimeSnapshot` gains `token_queue_depth`, `sentence_queue_depth`,
  `playback_queued_seconds`, `barge_in_count`, `last_barge_in_latency_ms` —
  additive fields, no new API endpoints (ADR-017), consumable by the future
  desktop/web UI's diagnostics page today.
- `eva bench` reports a real TTFA breakdown (`asr_ms`/`ttft_ms`/
  `first_chunk_ms`/`ttfa_ms`) measured through the same `synthesize_stream()`
  path the live pipeline uses, not a full-sentence-blocking estimate.
- Ctrl+C now exits cleanly at every stage of `eva run` — model loading, audio
  startup, and an active conversation all funnel through one try/finally that
  always calls `assistant.stop()`; every other CLI command gets a top-level
  backstop in `main()` (exit code 130, no traceback).

**Tests**
- +27 tests (291 total): TTS streaming (ABC default fallback, Kokoro
  multi-chunk streaming, early-stop cleanup simulating barge-in, error
  wrapping), first-chunk chunking threshold, playback chunk-boundary gap
  regression (with a contrast test proving the old per-chunk-flush behavior
  *would* have gapped), bounded-queue backpressure (including a tight-bounds/
  short-timeout crash guard), rapid double-barge-in, a 20-consecutive-
  interruption stress test plus a zero-delay-burst variant, benchmark TTFA
  breakdown, diagnostics field extensions, and CLI/voice-loop Ctrl+C handling
  at every stage.

**Deferred (documented, not implemented)**
- Speculative LLM generation on unconfirmed partial transcripts: would cut
  TTFA further but adds a second speculative-cancellation path at the same
  time this milestone hardens the existing one — worse risk/reward during a
  hardening pass. Candidate for M4+.
- ASR remains fully blocking per utterance (CTranslate2 has no per-token
  abort hook) — an accepted, bounded limitation (typically 200-400 ms),
  unchanged by this milestone.

**Not yet exit-tested**
- The literal "<150 ms audible stop" and "20 consecutive real-mic
  interruptions" targets need a real microphone/speaker and a stopwatch or
  audio-level probe on the reference machine (RTX 3060 Laptop, Ryzen 9
  5900HX) — not reproducible in this development environment. The automated
  stress tests validate the cancellation *mechanism* (epoch correctness, no
  leaked tasks, no crashes) under adversarial timing with fake engines, not
  physical audio latency.

### 2026-07-04 — CI fix: import order + pre-commit hooks

**Fixed (release blocker)**
- GitHub Actions failed lint with `I001` on `tests/test_language.py` and
  `tests/test_server_engine_and_conversation.py`. Root cause: adding
  `tests/__init__.py` (the earlier `ModuleNotFoundError` fix) changed how
  ruff's import sorter classifies `tests.*` imports as first-party, and the
  existing import order in those two files no longer matched. Fixed with
  `ruff check --fix` — pure import reordering, no functional change.
  Verified with a completely fresh `.ruff_cache` and a clean virtual
  environment: `ruff check`, `ruff format --check`, `mypy` (both platform
  targets), and the full 264-test suite all pass with the exact CI commands.

**Added**
- Pre-commit hooks (`.pre-commit-config.yaml`): trailing-whitespace,
  end-of-file-fixer, mixed-line-ending, check-yaml/toml, merge-conflict and
  large-file guards, and `ruff check --fix` + `ruff format` on every commit
  (ruff pinned to the exact version CI uses); `mypy` (strict) on every push.
  Documented in `docs/DEVELOPMENT.md`, including the one real caveat: the
  mypy hook runs against whatever's on `PATH`, so it requires the project's
  virtual environment to be active.

### 2026-07-04 — M2.6: Platform API & UI backend

**Added**
- FastAPI platform API (ADR-017): versioned REST under `/api/v1` plus one
  WebSocket event stream. `eva serve` runs it; the CLI is now one client of
  the same engine services the server exposes (Part 10 — no duplicated logic).
- `eva/server/`: app factory (localhost-only CORS, uniform `EvaError` → HTTP
  status mapping, OpenAPI/Swagger UI generated automatically), `ServerState`
  (the single engine-lifecycle owner — explicit `POST /engine/start`, never
  an implicit side effect of the server booting), and one router per concern:
  settings, models, diagnostics, plugins, conversation, engine, system.
- WebSocket (`/api/v1/ws`): forwards every existing engine event
  (transcripts, LLM tokens/sentences, TTS/playback, state transitions, turn
  lifecycle) plus new `ModelDownloadProgress/Completed/Failed` and
  `EngineStarted/Stopped` events; sends an initial `snapshot` so clients never
  need to poll before their first live event. `EventBus` now keeps bounded
  history for reconnects/diagnostics.
- Settings API: GET/PUT/PATCH/validate/reset + JSON Schema, all backed by a
  new shared `eva.config.service` module (also used by the new `eva config
  show|schema|reset` CLI group).
- Model manager API: list/info/download (background, progress via WebSocket)/
  remove/activate — the full `describe()` model card exposed over HTTP.
- Diagnostics API: `RuntimeSnapshot` with and without a running engine
  (`snapshot_idle` for the "server up, engine not started" state).
- Plugin API (`eva/plugins/`, ADR-011 backend): manifest schema + a genuinely
  functional `PluginManager` using standard entry points (group
  `eva.plugins`) — discover/enable/disable/reload, empty by default until a
  plugin package exists.
- Conversation API: history, current turn, interrupt/cancel (new
  `Orchestrator.interrupt()` — barge-in reachable without a microphone; new
  `TurnCancelled` reason `"manual"`), clear, export/import (new
  `ConversationHistory.turns`/`load_turns`).
- `docs/API.md` (endpoint map + WebSocket protocol) and ADR-017.
- 69 new tests (264 total): every router, the WebSocket stream (including
  disconnect/unsubscribe and multi-client fan-out), the plugin manager against
  fake entry points, the settings service, and full engine start/stop/interrupt/
  export/import cycles against a fake engine — plus a real end-to-end check
  against the installed Qwen3.5/faster-whisper/Kokoro models on reference
  hardware (LLM/ASR on CUDA, TTS on CPU, matching the M2.5 startup banner).
- Verified in a clean virtual environment (standing release gate): base
  install includes FastAPI/uvicorn/websockets with no compiler required;
  `eva serve` runs as a real subprocess answering HTTP and OpenAPI requests.

### 2026-07-04 — M2.5: Production hardening

**Fixed (release blockers)**
- **CI failed on every run because `src/eva/models/` was never committed**: the
  unanchored `.gitignore` pattern `models/` (meant for downloaded weights) also
  matched the source package. Runtime-artifact ignores are now anchored to the
  repo root, and a package-integrity test imports every `eva` module so a
  missing package can never pass CI again. GitHub Actions bumped to
  checkout@v5 / setup-python@v6 (Node deprecation warnings).
- **Inconsistent behavior across restarts** (different model selected, changed
  barge-in feel) — root-caused to unpersisted configuration, silent
  order-dependent device placement, and zero startup visibility; fixed
  architecturally (ADR-015), not by tuning thresholds.

**Added**
- Persistent configuration: first run resolves the active preset against the
  detected hardware tier and writes `settings.json`; model selection is stable
  across restarts and releases, with a pinning regression test.
- Model presets (ADR-015): Balanced / Fast / High Accuracy / Low Memory /
  Developer as registry data per hardware tier; `eva profiles list|set`;
  manual `eva models use <id>` persists and flips the profile to `custom`.
- Startup banner: profile, hardware tier, all four active models (LLM/ASR/
  TTS/VAD), the device each engine actually landed on, and the language.
- Deterministic engine load order (LLM → ASR → TTS): the LLM owns the GPU;
  engine ports expose the `device` actually used.
- Multilingual foundation (ADR-016): language registry with per-language ASR
  hints, prompt notes, and voice preferences; wired through the orchestrator;
  English, Finnish, Swedish, Bengali (tested) plus German and Spanish;
  graceful voice fallback when TTS lacks a native voice.
- Model manager as UI backend: `describe()` full model card (name, version,
  provider, license, languages, context length, VRAM/RAM, quantization, disk
  size, install state, update placeholder, active flag, hardware
  compatibility); `eva models info <id>`; provider/version metadata on every
  catalog entry.
- Developer diagnostics API (`eva.metrics.diagnostics`): JSON-serializable
  runtime snapshot — active models and devices, pipeline state, turn epoch,
  playback/VAD levels, queue depths, dropped frames, CPU/RAM/GPU/VRAM usage,
  last-turn latency metrics (TTFT/TTFA/tokens-per-s), and recent events (the
  bus now keeps a bounded history).
- Configuration system audit: every settings field now carries a description
  (schema-enforced by test); previously hidden defaults promoted to settings
  (`tts.model`, `audio.fade_out_ms`, sentence-chunker bounds,
  `conversation.language`).

**Changed**
- `run_probe` made public (shared by hardware detection and diagnostics);
  the hidden TTS-model mapping in the engine assembly was replaced by the
  `tts.model` setting.

**Tests**
- +37 tests (195 total): package integrity, presets (including preset↔catalog
  consistency), configuration persistence and stability pinning, language
  resolution for en/fi/sv/bn, diagnostics snapshots, model cards and
  compatibility flags, settings-documentation enforcement.

### 2026-07-04 — Guided first-run onboarding

**Added**
- Interactive setup wizard (`eva/onboarding.py`): on `eva run`, if the system
  is not fully set up, EVA explains what will happen (detected hardware,
  recommended runtime, required models with sizes and a time estimate), asks for
  one confirmation, then installs the runtime, downloads models, verifies, and
  starts the assistant — with step-by-step progress. No documentation required
  (ADR-014).
- `eva first-run` command: runs the wizard directly; `--setup-only` finishes
  setup without starting; `--yes` auto-confirms.
- `eva run --yes` for non-interactive/automated first runs.
- Persisted `SetupState` (`config/setup_state.json`) for first-time-vs-repair
  messaging and future config migration; the authoritative readiness gate
  remains the real installed artifacts.
- `download_mb_hint` on catalog entries so the wizard shows honest sizes for
  engine-managed models (e.g. Faster Whisper).

**Changed**
- `eva doctor` and the `run`/`bench` preflight now share one `check_readiness`
  implementation with the wizard (no duplicated readiness logic).
- `eva run` no longer just prints commands when setup is incomplete — it guides
  the user through it. Failures are reported in friendly language; tracebacks
  are never shown to end users.

**Preserved**
- `eva setup`, `eva doctor`, `eva models`, `eva diagnose` remain first-class
  developer tools; the wizard reuses them rather than duplicating logic.

**Tests**
- +16 tests (158 total): onboarding readiness, plan + estimates,
  confirm/decline, non-interactive blocking, full-run step execution, friendly
  failure, and state persistence — all hermetic (no network, models, or audio).

### 2026-07-04 — M2 packaging fix: installable from a clean checkout

**Fixed (release blocker)**
- Declared the ML runtimes that were used but missing from `pyproject.toml`.
  `faster-whisper` and `kokoro-onnx` (both ship universal PyPI wheels) are now
  base dependencies, so `pip install -e "."` yields a runnable ASR + TTS + audio
  application with no compiler. Previously a clean checkout failed at runtime
  with `No module named 'faster_whisper'` / `'llama_cpp'`.

**Added**
- `eva setup`: detects hardware and installs the `llama-cpp-python` build
  (CPU or CUDA) from the llama.cpp wheel index — the LLM runtime has no PyPI
  wheels, so it cannot be a plain dependency (ADR-013). Supports `--cpu`,
  `--cuda`, `--dry-run`, `--force`.
- `eva doctor`: readiness report listing every runtime and model as
  `ok`/`MISSING` with the exact remedy command.
- `[cpu]` and `[cuda]` optional-dependency extras for manual/reproducible
  installs; the `[cuda]` extra also pulls the NVIDIA cudart/cuBLAS wheels.
- `eva.runtime` module: runtime probing and install-command construction (pure,
  unit-tested).
- Preflight in `eva run` and `eva bench`: both now report missing runtimes and
  models with actionable guidance instead of raising `ModuleNotFoundError`.
- `docs/INSTALLATION.md` (Windows + Linux) and ADR-013.
- 17 new tests (142 total): runtime probing, variant selection, install-command
  construction, CLI `doctor`/`setup`/graceful-preflight behavior, and download
  truncation/resume.
- Established the clean-environment smoke test as a per-milestone release gate.

### 2026-07-04 — M2: Streaming conversational pipeline

**Added**
- Event system (`eva.core.events`): typed, JSON-serializable engine events
  (turn lifecycle, transcripts, LLM tokens/sentences, TTS, state changes) on an
  asyncio event bus with bounded per-subscriber queues and thread-safe publish.
- Turn management (`eva.core.turn`): monotonic turn epochs as the cancellation
  backbone (ADR-006); every pipeline artifact is epoch-tagged and stale work
  aborts at the next boundary.
- Engine ports + registries: `ASREngine`, `LLMEngine` (streaming + per-token
  abort contract), `TTSEngine` (sentence-granular synthesis, voice discovery),
  mirroring the M1 VAD registry (ADR-010).
- Adapters: faster-whisper (CTranslate2, CUDA→CPU fallback, greedy decode
  tuned for short utterances), llama.cpp (GGUF, streaming chat completion,
  abort per token, Windows CUDA DLL resolution), Kokoro via kokoro-onnx
  (torch-free, 24→16 kHz resampling at the adapter boundary) — ADR-012.
- Turn orchestrator (`eva.conversation.orchestrator`): asyncio pipeline —
  LLM producer thread → token consumer → sentence chunker → speak worker;
  barge-in/supersede/shutdown cancellation; partial transcripts from
  segmenter `UtteranceProgress` snapshots; per-turn metrics.
- Punctuation-aware sentence chunker (abbreviation/decimal guards, clause-
  boundary force-split for run-on generations).
- Conversation history with turn windowing (persistence lands in M4).
- Model manager backend: catalog as data (ids, licenses, sizes, VRAM/RAM
  needs, verified download URLs), atomic downloads with progress, install/
  remove/resolve, disk usage; consistency tests keep settings/profiles/catalog
  aligned.
- Default LLM updated to **Qwen3.5-4B** Q4_K_M (ADR-002 amendment).
- CLI: `eva run` (interactive voice loop with live token streaming),
  `eva models list|download|remove`, `eva bench` (reproducible end-to-end
  pipeline benchmark using TTS-generated speech — no microphone needed).
- Per-turn metrics collection (ASR, TTFT, tokens/s, first-sentence TTS, TTFA,
  total) with session summary.
- Dependencies: faster-whisper, kokoro-onnx, llama-cpp-python (CUDA wheel) +
  nvidia CUDA runtime wheels.
- 56 new unit tests (127 total): event bus, turn controller, chunker,
  history, resampler, model manager (truncation detection, resume, failure
  atomicity), and full orchestrator coverage with fake engines (streaming
  order, barge-in cancellation, superseding, repeated interruptions, failure
  paths, partial transcripts, metrics).

**Fixed**
- Model downloads now verify received bytes against Content-Length and resume
  via HTTP Range on retry — a dropped connection previously produced a
  silently truncated model file that failed at load time.

**Benchmarks** (RTX 3060 Laptop 6 GB, Ryzen 9 5900HX; `eva bench`, warm run)
- ASR (faster-whisper small int8, CUDA): 490 ms for 2.9 s of speech
- Time to first token (ASR + LLM prefill): 535 ms
- LLM (Qwen3.5-4B Q4_K_M, full GPU offload): 65 tok/s
- First reply sentence ready: 140 ms after generation start
- First-sentence TTS (Kokoro, CPU): ~1.3 s (RTF ≈ 0.6)
- Estimated time to first audio: ~2.0 s — dominated by first-sentence TTS;
  identified M3/M7 lever: chunked/streamed synthesis of the first segment
  (kokoro-onnx supports incremental synthesis) and/or shorter first segment.
- Model load time (all three engines): ~16 s cold.

### 2026-07-03 — M1: Full-duplex audio core

**Added**
- Canonical audio format: 16 kHz mono int16 in 10 ms frames (`eva.audio.frames`),
  with level metering and float/int conversion helpers.
- `FrameRing`: bounded, drop-oldest frame queue between the audio callback and
  consumer threads, with overflow diagnostics.
- `PlaybackQueue`: frame-granular playback with click-free fade-out (40 ms) —
  the mechanism barge-in uses to silence the assistant instantly.
- `DuplexAudioStream`: one PortAudio stream for capture + playback (single
  clock), real-time-safe callback, measured loop delay reported to the echo
  canceller, per-callback error containment.
- WebRTC APM integration (`WebRtcAudioProcessor`): echo cancellation, noise
  suppression, AGC, high-pass filter; graceful fallback to passthrough when the
  native module is unavailable.
- VAD subsystem (`eva.vad`): `VADEngine` port, Silero adapter (ONNX, no torch),
  and the platform's first component registry (`eva.core.registry.Registry`).
- `SpeechSegmenter`: pure-logic endpointing state machine — 300 ms pre-roll,
  noise gate, mid-utterance pause tolerance, max-utterance safety stop, and
  single-shot barge-in confirmation that keeps the triggering speech for ASR.
- `CapturePipeline` consumer thread (frames → VAD chunks → segmenter events)
  and `AudioSystem` composition root.
- CLI: `eva devices`, `eva listen` (live VAD monitor), `eva echo-test`
  (speaker/microphone echo-suppression measurement with pass/fail verdict).
- Dependencies: numpy, sounddevice, livekit (WebRTC APM), pysilero-vad (<3).
- 48 new unit tests (71 total), including a device-free APM test proving
  >10 dB attenuation of a synthetic echo.

**Verified**
- Full quality gate green (ruff, mypy strict, pytest) on Windows.
- Live duplex run on reference hardware: WebRTC APM active, 0 callback errors,
  no VAD self-triggers during tone playback.

### 2026-07-03 — Architecture review & project identity

- Renamed the project to **Edge Voice Assistant** across the repository
  (folder `edge-voice-assistant`, docs, architecture, roadmap); release
  versioning now targets 1.0.0.
- ADR-010: subsystem packages (`vad/`, `asr/`, `llm/`, `tts/`, `memory/`,
  `tools/`, …) each owning port + registry + adapters, replacing the
  `ports/`/`adapters/` layering; single registry primitive in `core`;
  dependency-direction rule documented.
- ADR-011: plugin SDK — manifest + entry points, narrow `eva.sdk` facade,
  marketplace-ready lifecycle (install/update/enable/disable/remove).
- Hardware profiles redesigned as two layers: detected capability tier →
  goal-oriented presets (Balanced / Fast / High Accuracy / Low Memory /
  Developer / Custom, user-editable).
- Settings surface expanded to the full section list (General, per-subsystem
  model managers, Conversation, Memory, Prompt Templates, Personalities, Audio,
  Hardware, Performance, Plugins, Developer, Diagnostics, Appearance,
  Accessibility, Privacy, Updates).
- Added `docs/DEVELOPMENT.md` (setup, quality gate, architecture rules, coding
  standards, release checklist).

### 2026-07-03 — M0: Project foundation

**Added**
- Installable `eva` package (src layout, `pyproject.toml`, MIT license, typed).
- Settings system: strict pydantic schema (audio, VAD, ASR, LLM, TTS, conversation,
  server, UI, developer sections) with validation bounds, atomic JSON persistence,
  and partial-file merge. VAD defaults carry over the values tuned in the thesis
  prototype.
- Application paths via platformdirs, with `EVA_HOME` override for portable
  installs and test isolation.
- Hardware detection (psutil + `nvidia-smi`/`rocm-smi` probes; degrades to
  CPU-only, never raises) and hardware-profile recommendation
  (`cpu-only` / `gpu-6gb` / `gpu-12gb`).
- Logging: console + rotating file handler, optional JSON line format.
- CLI: `eva diagnose` (system/hardware/profile/configuration/paths report) and
  `eva version`; UTF-8 output enforced on Windows consoles.
- Tooling: ruff (lint + format), mypy strict with pydantic plugin, pytest
  (23 unit tests), CI workflow for Windows + Linux.

**Verified**
- Full quality gate green (lint, format, types, tests).
- `eva diagnose` on reference hardware (RTX 3060 Laptop, 6 GB VRAM) detects the
  GPU and recommends the `gpu-6gb` profile.

### 2026-07-03 — Project inception

- Analyzed the thesis prototype; findings recorded in internal notes.
- Evaluated the current open-weight model landscape (ASR, LLM, TTS, VAD, AEC).
- Defined the system architecture (`docs/ARCHITECTURE.md`), roadmap
  (`docs/ROADMAP.md`), and ADR-001 … ADR-009.
