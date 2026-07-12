"""Punctuation-aware sentence chunking of the LLM token stream.

Feeds TTS with speakable segments before generation finishes — the single
biggest lever on time-to-first-audio. Emission rules:

- A segment ends at sentence punctuation (. ! ? … and CJK equivalents)
  followed by whitespace/end, once at least `min_chars` have accumulated
  (avoids speaking fragments like "Hi." separately when more follows fast).
- Common abbreviations and decimal points do not end a segment.
- If a segment reaches `max_chars` without punctuation (run-on generation),
  it is emitted at the last clause break (comma/semicolon/colon) or word
  boundary — TTS quality degrades gracefully instead of stalling.
- The very first segment of a turn may use a lower `first_chunk_min_chars`
  threshold (M3/ADR-018): the first sentence is the one the user is waiting
  on, so it is worth trading a touch of "naturalness" (occasionally speaking
  a short fragment on its own) for a materially earlier first sound. Every
  segment after the first uses `min_chars` as before.
- The first segment additionally splits at the first CLAUSE break — a comma,
  semicolon, or colon followed by whitespace (M5.6). Synthesis cost scales
  with text length, so "Sure, let me check that." starts speaking after
  "Sure," instead of after the whole sentence — the single biggest
  time-to-first-audio lever left after streaming synthesis (ADR-018). Only
  the first segment: later sentences are synthesized while earlier ones
  play, so they gain nothing from splitting and keep natural prosody.
"""

from __future__ import annotations

import re

_SENTENCE_END = re.compile(r"[.!?…。！？](?=\s|$)")  # noqa: RUF001 — CJK punctuation intended
_CLAUSE_BREAK = re.compile(r"[,;:](?=\s)")  # decimal commas ("1,5") don't match
_ABBREVIATIONS = frozenset(
    {"dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e", "approx"}
)


class SentenceChunker:
    def __init__(
        self,
        min_chars: int = 12,
        max_chars: int = 350,
        first_chunk_min_chars: int | None = None,
    ) -> None:
        self._min_chars = min_chars
        self._max_chars = max_chars
        self._first_chunk_min_chars = first_chunk_min_chars
        self._buffer = ""
        self._emitted_any = False

    def feed(self, token: str) -> list[str]:
        """Add a token; return zero or more complete speakable segments."""
        self._buffer += token
        segments: list[str] = []
        while True:
            segment = self._extract_segment()
            if segment is None:
                break
            segments.append(segment)
        return segments

    def flush(self) -> str | None:
        """Return whatever remains (call when generation ends)."""
        rest = self._buffer.strip()
        self._buffer = ""
        return rest or None

    def reset(self) -> None:
        self._buffer = ""
        self._emitted_any = False

    def _active_min_chars(self) -> int:
        if not self._emitted_any and self._first_chunk_min_chars is not None:
            return self._first_chunk_min_chars
        return self._min_chars

    def _extract_segment(self) -> str | None:
        min_chars = self._active_min_chars()
        for match in _SENTENCE_END.finditer(self._buffer):
            end = match.end()
            if end < min_chars:
                continue
            if self._is_abbreviation(self._buffer[:end]):
                continue
            segment = self._buffer[:end].strip()
            self._buffer = self._buffer[end:].lstrip()
            self._emitted_any = True
            return segment
        # First segment only (M5.6): a clause break is good enough to start
        # audio — the user is waiting in silence; see the module docstring.
        if not self._emitted_any and self._first_chunk_min_chars is not None:
            for match in _CLAUSE_BREAK.finditer(self._buffer):
                end = match.end()
                if end < min_chars:
                    continue
                segment = self._buffer[:end].strip()
                self._buffer = self._buffer[end:].lstrip()
                self._emitted_any = True
                return segment
        if len(self._buffer) >= self._max_chars:
            return self._force_split()
        return None

    def _is_abbreviation(self, text: str) -> bool:
        if not text.endswith("."):
            return False
        last_word = text[:-1].rsplit(None, 1)[-1].lower() if text[:-1].split() else ""
        # Single letters ("J."), known abbreviations, and decimals don't end sentences.
        return len(last_word) == 1 or last_word in _ABBREVIATIONS or last_word.isdigit()

    def _force_split(self) -> str:
        self._emitted_any = True
        window = self._buffer[: self._max_chars]
        for pattern in (r"[,;:](?=\s)", r"\s"):
            matches = list(re.finditer(pattern, window))
            if matches:
                end = matches[-1].end()
                segment = self._buffer[:end].strip()
                self._buffer = self._buffer[end:].lstrip()
                return segment
        segment, self._buffer = window, self._buffer[self._max_chars :]
        return segment.strip()
