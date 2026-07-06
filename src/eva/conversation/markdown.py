"""Markdown → speakable plain text (ADR-024).

The TTS boundary is the ONLY place this runs — storage, events, API, and
export all keep raw Markdown canonical. `MarkdownSpeechFilter` is stateful
because the speak worker synthesizes per sentence *segment* (ADR-018): a
fenced code block's opening ``` and closing ``` routinely arrive in
different segments, and only per-turn state can suppress the lines in
between. Heuristic by design (LLM-typical Markdown, not full CommonMark);
the contract is "never speak formatting characters, never crash".
"""

from __future__ import annotations

import re

# Inline transforms, applied in order. Fence handling happens line-by-line
# in the filter (stateful) before these run.
_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_BOLD_ITALIC = re.compile(r"(\*{1,3}|_{1,3})(?=\S)(.+?)(?<=\S)\1")
_STRIKETHROUGH = re.compile(r"~~(?=\S)(.+?)(?<=\S)~~")
_INLINE_CODE = re.compile(r"`([^`]*)`")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+")
_BLOCKQUOTE = re.compile(r"^\s{0,3}(?:>\s?)+")
_BULLET = re.compile(r"^\s*[-*+]\s+")
_TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{2,}[\s|:-]*$")
_HORIZONTAL_RULE = re.compile(r"^\s{0,3}(?:[-*_]\s*){3,}$")
_HTML_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
_FENCE = re.compile(r"^\s{0,3}(```|~~~)")
_MULTISPACE = re.compile(r"[ \t]{2,}")


def _convert_inline(line: str) -> str:
    line = _IMAGE.sub(r"\1", line)
    line = _LINK.sub(r"\1", line)
    line = _INLINE_CODE.sub(r"\1", line)
    # Run emphasis twice: ***x*** unwraps to *x* on the first pass.
    line = _BOLD_ITALIC.sub(r"\2", line)
    line = _BOLD_ITALIC.sub(r"\2", line)
    line = _STRIKETHROUGH.sub(r"\1", line)
    return _HTML_TAG.sub("", line)


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
