"""Markdown → speech conversion tests (ADR-024).

The contract: the TTS engine must never receive formatting characters —
asterisks, underscores, backticks, table pipes, link syntax — and must
never crash on malformed Markdown. Raw Markdown stays canonical everywhere
else (storage/events are covered by the orchestrator test asserting the
stored reply keeps its Markdown while the spoken text doesn't).
"""

from __future__ import annotations

import pytest

from eva.conversation.markdown import MarkdownSpeechFilter, markdown_to_speech


class TestInlineFormatting:
    def test_bold(self) -> None:
        assert (
            markdown_to_speech("My name is **Edge Voice Assistant**.")
            == "My name is Edge Voice Assistant."
        )

    def test_italic_asterisk_and_underscore(self) -> None:
        assert markdown_to_speech("This is *important* and _subtle_.") == (
            "This is important and subtle."
        )

    def test_bold_italic_triple_markers(self) -> None:
        assert markdown_to_speech("***very strong***") == "very strong"

    def test_double_underscore_bold(self) -> None:
        assert markdown_to_speech("__bold__ text") == "bold text"

    def test_strikethrough(self) -> None:
        assert markdown_to_speech("~~wrong~~ right") == "wrong right"

    def test_inline_code(self) -> None:
        assert markdown_to_speech("Run `eva doctor` to check.") == "Run eva doctor to check."

    def test_multiplication_asterisk_is_not_emphasis(self) -> None:
        # "3 * 4" has spaces around the asterisk — not an emphasis pair.
        assert markdown_to_speech("3 * 4 equals 12") == "3 * 4 equals 12"


class TestBlockElements:
    def test_headings(self) -> None:
        assert markdown_to_speech("## Setup steps") == "Setup steps"
        assert markdown_to_speech("# Title\nBody text.") == "Title Body text."

    def test_blockquote(self) -> None:
        assert markdown_to_speech("> quoted wisdom") == "quoted wisdom"
        assert markdown_to_speech(">> nested quote") == "nested quote"

    def test_bullet_lists(self) -> None:
        assert markdown_to_speech("- first\n- second\n* third") == "first second third"

    def test_numbered_lists_keep_numbers(self) -> None:
        # "1." is natural speech for an enumeration; only bullets are dropped.
        assert markdown_to_speech("1. first\n2. second") == "1. first 2. second"

    def test_horizontal_rule_dropped(self) -> None:
        assert markdown_to_speech("above\n---\nbelow") == "above below"

    def test_html_tags_stripped(self) -> None:
        assert markdown_to_speech("a <br> b <span>c</span>") == "a b c"


class TestLinksAndImages:
    def test_link_speaks_text_not_url(self) -> None:
        assert (
            markdown_to_speech("See [the docs](https://example.com/x) for more.")
            == "See the docs for more."
        )

    def test_image_speaks_alt_text(self) -> None:
        assert markdown_to_speech("![a chart](chart.png) shows growth") == "a chart shows growth"


class TestTables:
    def test_table_rows_become_comma_separated(self) -> None:
        table = "| Name | Size |\n|------|------|\n| Qwen | 4B |"
        assert markdown_to_speech(table) == "Name, Size Qwen, 4B"

    def test_separator_row_never_spoken(self) -> None:
        assert "-" not in markdown_to_speech("|---|---|")


class TestCodeFences:
    def test_fenced_code_content_suppressed(self) -> None:
        text = "Here is the fix:\n```python\nx = 1\nprint(x)\n```\nDone."
        assert markdown_to_speech(text) == "Here is the fix: Done."

    def test_fence_state_carries_across_segments(self) -> None:
        """The real streaming case (ADR-024): the opening and closing fences
        arrive in different sentence segments — per-turn state must keep
        suppressing between them."""
        f = MarkdownSpeechFilter()
        assert f.convert("Look at this:\n```python") == "Look at this:"
        assert f.convert("secret = 'do not speak me'") == ""
        assert f.convert("```\nAnd that's the code.") == "And that's the code."

    def test_unclosed_fence_does_not_crash_and_stays_silent(self) -> None:
        f = MarkdownSpeechFilter()
        assert f.convert("```\ncode forever") == ""
        assert f.convert("still inside") == ""

    def test_tilde_fences(self) -> None:
        assert markdown_to_speech("~~~\nhidden\n~~~\nspoken") == "spoken"


class TestRobustness:
    @pytest.mark.parametrize(
        "weird",
        ["", "   ", "***", "``", "[", "![](", "| | |", "**unclosed bold", "`unclosed code"],
    )
    def test_never_crashes_on_malformed_input(self, weird: str) -> None:
        markdown_to_speech(weird)  # must not raise

    def test_plain_text_passes_through_unchanged(self) -> None:
        assert markdown_to_speech("Hello, how can I help you today?") == (
            "Hello, how can I help you today?"
        )

    def test_no_formatting_characters_survive_typical_reply(self) -> None:
        reply = (
            "# Answer\n\nUse **bold** and `code`:\n\n"
            "```js\nconsole.log('*');\n```\n\n"
            "1. [click here](http://x.dev)\n- done\n\n"
            "| a | b |\n|---|---|\n| 1 | 2 |"
        )
        spoken = markdown_to_speech(reply)
        for char in ("*", "`", "#", "|", "](", "~~"):
            assert char not in spoken, f"{char!r} leaked into speech: {spoken!r}"
