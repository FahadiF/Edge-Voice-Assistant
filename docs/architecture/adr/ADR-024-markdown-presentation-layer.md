# ADR-024: Markdown presentation layer

Status: Accepted · Date: 2026-07-05

## Context

LLMs emit Markdown. Manual M5 testing surfaced two presentation failures:
the web UI displayed raw Markdown source (`**bold**`, ` ``` ` fences) as
text, and the TTS pipeline spoke the formatting characters aloud
("asterisk asterisk"). Both are presentation bugs, not data bugs — the
text itself is fine.

## Decision

**Markdown is the canonical representation everywhere except the two
presentation boundaries.** SQLite storage, the REST API, WebSocket events,
export/import, memory search, and summaries all carry the raw text exactly
as the LLM produced it — zero transformation (verified: `orchestrator.py`'s
`add_turn` calls and every `Llm*` event publish raw text). Only two places
transform, at render time:

### 1. Web UI: `react-markdown` + `remark-gfm`

Assistant bubbles (live-streaming and history) render through
`react-markdown` with the GFM plugin — bold, italic, headings, inline code,
fenced code blocks, blockquotes, ordered/unordered lists, tables, and
links. Raw HTML in model output is **not** rendered (react-markdown's
default; no `rehype-raw`) — LLM output is untrusted input and the CSP
argument from ADR-023 applies doubly to injected markup. Code blocks get a
copy button; syntax highlighting is deliberately deferred (a highlighter
grammar bundle is a large dependency for an offline app — revisit if users
ask). This does not contradict ADR-023's "no component framework" stance:
react-markdown is a focused renderer for exactly one well-bounded problem,
bundled into the same offline build.

User bubbles stay plain text — users speak, they don't dictate Markdown,
and rendering their words as markup would misrepresent the transcript.

### 2. TTS: `eva.conversation.markdown` — a *stateful* speech filter

`MarkdownSpeechFilter.convert(segment)` turns Markdown into clean speakable
text: formatting markers removed (content kept), `[text](url)` → text,
`![alt](…)` → alt, heading/blockquote/bullet markers dropped, table pipes
and separator rows dropped, horizontal rules dropped, **fenced code block
content suppressed entirely** (speaking source code aloud is worse than
silence; the screen has it).

It is a class, not a pure function, for one measured reason: the speak
worker synthesizes *per sentence segment* (ADR-018 streaming), and a code
fence routinely spans segments — the opening ` ``` ` arrives in one
segment, the closing in a later one. Only per-turn state can know "we are
inside a fence" at segment N. The orchestrator creates one filter per turn
and feeds each segment through it before `synthesize_stream()`
(`orchestrator.py` speak worker — the single point where LLM text reaches
TTS, confirmed by grep). Segments that convert to empty/whitespace are
skipped, not synthesized.

A stateless `markdown_to_speech(text)` wrapper exists for whole-text
callers and tests.

### What deliberately does NOT change

- `LlmToken`/`LlmSentence`/`LlmFinished` events carry raw Markdown — the
  web UI needs it to render; a client that wants plain text can convert.
- The `SentenceChunker` stays Markdown-unaware. Its existing
  digit-abbreviation rule already prevents `1.` list items from splitting
  prematurely; making it parse Markdown would couple sentence segmentation
  to a syntax it doesn't need to understand.
- Voice preview and the benchmark synthesize user/fixture-provided phrases,
  not LLM output — no filter there.

## Consequences

- Two new frontend dependencies (`react-markdown`, `remark-gfm`), bundled
  offline like everything else.
- The speech filter is heuristic (regex-based, not a full CommonMark
  parser) — good enough for LLM-typical Markdown; a pathological document
  might slip a marker through. It must never *crash* on weird input:
  tests cover unclosed fences, nested markers, and empty results.
- If a future engine wants SSML instead of plain text, the filter is the
  seam: swap the converter, not the orchestrator.
