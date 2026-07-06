"""Markdown → speakable plain text (ADR-024, hardened in M5.3).

The TTS boundary is the ONLY place this runs — storage, events, API, and
export all keep raw Markdown canonical. `MarkdownSpeechFilter` is stateful
because the speak worker synthesizes per sentence *segment* (ADR-018): a
fenced code block's opening ``` and closing ``` routinely arrive in
different segments, and only per-turn state can suppress the lines in
between.

Why not a CommonMark parser: the filter's input is streaming *fragments*,
not documents. The sentence chunker can split inside an emphasis span
("**Generate a summary. **Then…"), leaving markers unpaired in each
fragment — CommonMark says unpaired markers render literally, so a
compliant parser would still speak "asterisk asterisk" (the exact M5.3
bug). The contract here is stronger than CommonMark: formatting characters
are NEVER spoken, paired or not. Hence: paired transforms run to a
fixpoint (nesting), then a final scrub removes any residual marker runs.
Legitimate uses survive — "3 * 4" keeps its asterisk (spaces both sides);
"file_name_here" keeps its underscores (see `_scrub_residual_markers`).
"""

from __future__ import annotations

import html
import re

# ── inline transforms (paired constructs, applied to a fixpoint) ──
_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_AUTOLINK = re.compile(r"<(https?://[^>\s]+)>")
_BARE_URL = re.compile(r"https?://\S+")
# Asterisk emphasis can be intraword (CommonMark); underscore emphasis
# cannot — `file_name_here` is literal, `_word_` is italic.
_ASTERISK_EMPHASIS = re.compile(r"(\*{1,3})(?=\S)(.+?)(?<=\S)\1")
_UNDERSCORE_EMPHASIS = re.compile(r"(?<!\w)(_{1,3})(?=\S)(.+?)(?<=\S)\1(?!\w)")
_STRIKETHROUGH = re.compile(r"~~(?=\S)(.+?)(?<=\S)~~")
_INLINE_CODE = re.compile(r"`+([^`]*)`+")
_HTML_TAG = re.compile(r"</?[a-zA-Z][^>]*>")

# ── block-level line shapes ──
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+")
_BLOCKQUOTE = re.compile(r"^\s{0,3}(?:>\s?)+")
_BULLET = re.compile(r"^\s*[-*+]\s+")
_TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{2,}[\s|:-]*$")
_HORIZONTAL_RULE = re.compile(r"^\s{0,3}(?:[-*_]\s*){3,}$")
_FENCE = re.compile(r"^\s{0,3}(```|~~~)")

# ── residual scrub (unpaired/malformed markers that survived the paired
#    transforms — the streaming-fragment case) ──
_RESIDUAL_RUNS = re.compile(r"(\*{2,}|_{2,}|~{2,}|`+)")
_RESIDUAL_EDGE = re.compile(r"(?:(?<=\s)|^)[*_](?=\S)|(?<=\S)[*_](?=\s|$|[.,;:!?])")
_MULTISPACE = re.compile(r"[ \t]{2,}")

_MAX_FIXPOINT_PASSES = 5  # nesting deeper than this doesn't occur in practice


def _convert_inline(line: str) -> str:
    line = html.unescape(line)
    line = _IMAGE.sub(r"\1", line)
    line = _LINK.sub(r"\1", line)
    line = _AUTOLINK.sub(r"\1", line)
    line = _INLINE_CODE.sub(r"\1", line)
    # Paired emphasis to a fixpoint: ***bold italic***, **bold *nested***,
    # and similar unwrap one layer per pass.
    for _ in range(_MAX_FIXPOINT_PASSES):
        stripped = _ASTERISK_EMPHASIS.sub(r"\2", line)
        stripped = _UNDERSCORE_EMPHASIS.sub(r"\2", stripped)
        stripped = _STRIKETHROUGH.sub(r"\1", stripped)
        if stripped == line:
            break
        line = stripped
    line = _HTML_TAG.sub("", line)
    return _scrub_residual_markers(line)


def _scrub_residual_markers(line: str) -> str:
    """Remove formatting-marker characters the paired transforms could not
    resolve (unpaired `**` from a mid-span chunker split, stray backticks,
    malformed emphasis). Deliberately conservative for single characters:
    only a `*`/`_` hugging a word on exactly one side is treated as a
    marker — "3 * 4" (spaces both sides) and "snake_case_name" (letters
    both sides) are real content and survive."""
    line = _RESIDUAL_RUNS.sub("", line)
    return _RESIDUAL_EDGE.sub("", line)


def _convert_line(line: str) -> str | None:
    """One non-fence line → speakable text, or None to drop the line."""
    if _HORIZONTAL_RULE.match(line) or _TABLE_SEPARATOR.match(line):
        return None
    line = _HEADING.sub("", line)
    line = _BLOCKQUOTE.sub("", line)
    line = _BULLET.sub("", line)
    if "|" in line:
        # Table row: cells become comma-separated speech.
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        line = ", ".join(cell for cell in cells if cell)
    line = _convert_inline(line)
    line = _MULTISPACE.sub(" ", line).strip()
    return line if line else None


class MarkdownSpeechFilter:
    """Converts Markdown segments to speech text, one turn's worth at a
    time. Create one per assistant turn; feed each sentence segment through
    `convert()` in order — fence state carries across calls."""

    def __init__(self) -> None:
        self._in_fence = False

    def convert(self, segment: str) -> str:
        spoken_lines: list[str] = []
        for line in segment.splitlines():
            if _FENCE.match(line):
                self._in_fence = not self._in_fence
                continue  # the fence marker itself is never spoken
            if self._in_fence:
                continue  # code is for the screen, not the voice (ADR-024)
            converted = _convert_line(line)
            if converted is not None:
                spoken_lines.append(converted)
        return " ".join(spoken_lines).strip()


def markdown_to_speech(text: str) -> str:
    """Whole-text convenience wrapper (single-shot callers and tests)."""
    return MarkdownSpeechFilter().convert(text)
