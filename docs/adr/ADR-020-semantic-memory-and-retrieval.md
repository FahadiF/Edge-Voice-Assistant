# ADR-020: Semantic memory — embedding model and retrieval strategy

Status: Accepted · Date: 2026-07-05

## Context

M4 requires semantic search over conversation history: relevant memories,
similar past conversations, and context retrieval scored by relevance,
recency, and importance. This needs (a) a way to turn text into vectors and
(b) a way to find nearby vectors quickly. ADR-012 forbids PyTorch in the
product; ADR-013 prefers dependencies with universal wheels over ones
needing a compiler; ADR-010 (as amended) permits `eva.memory` to depend on
one new capability subsystem.

## Decision

1. **New subsystem `eva/embedding/`** (sibling to `vad`/`asr`/`llm`/`tts`,
   same port+registry+adapter shape): `base.py` (`EmbeddingProvider` port),
   `registry.py`, `onnx.py` (`OnnxEmbeddingProvider`). ADR-010 §2 already
   named "(future) embeddings" as a top-level registered kind before M4
   existed to need it — this keeps that plan rather than burying embeddings
   inside `eva/memory/`, since nothing about turning text into a vector is
   memory-specific (a future RAG-over-documents or semantic-tool-matching
   feature would want the same capability without importing memory).

2. **Model: `all-MiniLM-L6-v2` exported to ONNX** (384-dim output, ~90 MB,
   Apache-2.0). It's the standard offline sentence-embedding baseline —
   small enough to run on CPU in single-digit milliseconds per sentence,
   good enough quality for personal-conversation-scale semantic search, and
   avoids competing with the LLM for VRAM (matches the existing pattern
   where TTS/VAD stay CPU-resident so the LLM keeps the GPU — ADR-015 §5).
   Runs via **`onnxruntime`** (already installed transitively through
   `kokoro-onnx`/`pysilero-vad`; promoted to a direct base dependency since
   the embedding adapter imports it directly) for inference and
   **`tokenizers`** (HuggingFace's Rust-backed tokenizer bindings — no
   PyTorch, no TensorFlow) for tokenization. Both publish universal wheels
   for every target platform (confirmed: `pip index versions` shows current
   releases with prebuilt wheels), so this follows ADR-013's "prefer
   universal-wheel base dependencies" pattern rather than ADR-013's
   exception path (only `llama-cpp-python` needs that). Pooling
   (mean-pool over token embeddings, masked by attention, then
   L2-normalize) is ~40 lines of numpy — no `sentence-transformers`
   package needed, which would pull in a heavier dependency surface for a
   single well-documented operation.

3. **New model catalog entry**, `kind="embedding"` (extends the `ModelKind`
   Literal in `eva.models.catalog` from `["llm","asr","tts","vad"]`),
   `id="all-minilm-l6-v2-onnx"`, downloaded and integrity-verified through
   the existing `ModelManager` — no new install mechanism, no change to the
   download/resume/verification code that already exists and must not
   regress (HANDOFF's standing rule on the download-integrity check).

4. **Retrieval: brute-force cosine similarity in numpy, not a vector
   database.** `MemoryRetriever.retrieve()` loads the embedding matrix for
   the relevant scope from SQLite into numpy each call (no cache — see
   Amendment below for why one wouldn't help), computes `query · matrix.T`
   (a single BLAS call), blends in a recency-decay term (exponential
   half-life, configurable) and an importance boost (pinned/favorite turns
   score higher), and returns the top-K. No `faiss`/`chromadb`/`sqlite-vec`
   dependency: at personal-assistant scale (thousands, plausibly tens of
   thousands of turns over years of use — not millions), a single
   384-dimension matrix-vector product is faster than the LLM/ASR/TTS
   stages it feeds, and every candidate vector library either needs a
   compiled binary with platform-specific wheels (`faiss-cpu`) or is a
   newer, less battle-tested SQLite extension (`sqlite-vec`) — added
   packaging risk for a performance problem this scale doesn't have.
   `MemoryRetriever` stays a real port specifically so this can change: a
   `FaissMemoryRetriever` adapter is a drop-in replacement if a future
   scale (or a plugin wanting a different scoring policy) ever needs it,
   with zero changes to `MemoryStore`, the Context Builder, or the API.

5. **Measured, not estimated.** `eva/benchmark/memory.py` generates
   synthetic embedding sets at realistic-to-generous sizes and reports real
   retrieval latency, memory footprint, and (with the Context Builder,
   ADR-021) composition time — the milestone brief asks for measurement
   specifically where practical, and retrieval latency is exactly the kind
   of number that's easy to guess wrong.

## Rationale

Every existing speech-model choice in this codebase (Silero, faster-whisper,
Kokoro) already follows "smallest well-established open model that clears
the bar, run via ONNX/CTranslate2, downloaded through the same manager" —
the embedding model choice is the same policy applied to a new capability,
not a new policy. Brute-force retrieval is the "measure before optimizing"
principle applied to a problem that, at this product's actual scale, doesn't
need the machinery a cloud-scale RAG system would.

## Consequences

- `eva.memory`'s dependency on `eva.embedding` is the ADR-010 amendment's
  first (and, for this milestone, only) instance.
- If retrieval latency measurements from `eva/benchmark/memory.py` ever show
  brute-force numpy becoming a bottleneck at real usage scale, the fix is a
  new `MemoryRetriever` adapter — not a redesign of the memory subsystem.
- The embedding model adds one more model the onboarding wizard can
  offer/download; it is not required for M4's core persistence/search-by-
  text functionality to work (FTS5/LIKE text search functions without any
  embedding model installed — semantic search degrades gracefully to
  keyword search if the user hasn't downloaded it yet).

## Amendment (M4, 2026-07-05): measured findings changed two things

Running `eva/benchmark/memory.py` for real (not estimating) surfaced two
decisions this ADR's first version got wrong or left unbounded:

1. **`ContextBuilder` retrieval is unscoped (`conversation_id=None`),
   searching every conversation, not just the active one.** The original
   `MemoryRetriever.retrieve()` implementation and `ContextBuilder` wiring
   both defaulted to scoping by the active conversation — which technically
   works, but contradicts the actual point of persistent semantic memory
   (Part 3: "related conversations", "similar memories" — recalling *past*
   sessions, not just the current one, which `recent_turns` already covers
   via the recent-turn window). Fixed by having `ContextBuilder` always call
   `retrieve(..., conversation_id=None)`.

2. **`MemoryStore.embeddings_for()` gained a `limit` parameter** (most
   recent N by `embeddings.turn_id`, which is that table's own primary key
   and therefore already indexed and monotonic with insertion order) and
   `NumpyMemoryRetriever` a `scan_limit` (new setting:
   `MemorySettings.retrieval_scan_limit`, default 2000) — bounding how many
   candidate embeddings are ever scored per query, independent of how much
   history has accumulated. Also fixed an N+1 query pattern in the original
   retriever (`get_turn()` called once per *candidate*, not just per
   *result* — `MemoryStore.get_turns()` bulk fetch replaced it).

**A caching layer was considered and rejected** (despite the first version
of this ADR claiming one): the orchestrator's real access pattern is one
retrieval *before* each new turn is written, so the embedding set has
already changed by the next retrieval — a naive cache invalidates every
single turn in normal use, providing no benefit while adding complexity.

**A real limitation, documented rather than fully solved:** the
`limit`+`ORDER BY turn_id DESC` bound is fast when a query spans many
conversations (SQLite's planner scans `embeddings` in primary-key order and
stops early — measured at ~12 ms scoring 2000 of 20,000 candidates). It is
*not* fast if a single conversation alone accumulates many thousands of
turns without ever being archived or rotated: SQLite's planner drives that
query from `turns` (filtered by `conversation_id`) instead, and sorts the
full per-conversation result before the limit applies (measured: ~900 ms at
20,000 turns in one conversation). This only matters for scoped
(`conversation_id=<id>`) queries — the default unscoped path `ContextBuilder`
actually uses is unaffected. Mitigated today by `MemorySettings.
max_turns_per_conversation` (retention policy, ADR-019 §10); a proper fix
(a covering index, or restructuring the query to always drive from
`embeddings`) is a candidate for M5 if a single-conversation-scoped search
endpoint sees real use at that scale.
