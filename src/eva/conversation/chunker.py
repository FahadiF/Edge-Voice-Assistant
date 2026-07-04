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
"""

from __future__ import annotations

import re

_SENTENCE_END = re.compile(r"[.!?…。！？](?=\s|$)")  # noqa: RUF001 — CJK punctuation intended
_ABBREVIATIONS = frozenset(
    {"dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e", "approx"}
)


class SentenceChunker:
    def __init__(self, min_chars: int = 12, max_chars: int = 350) -> None:
        self._min_chars = min_chars
        self._max_chars = max_chars
        self._buffer = ""

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

    def _extract_segment(self) -> str | None:
        for match in _SENTENCE_END.finditer(self._buffer):
            end = match.end()
            if end < self._min_chars:
                continue
            if self._is_abbreviation(self._buffer[:end]):
                continue
            segment = self._buffer[:end].strip()
            self._buffer = self._buffer[end:].lstrip()
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
