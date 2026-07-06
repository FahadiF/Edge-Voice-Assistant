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


class TestUnpairedMarkers:
    """The M5.3 reported bug: emphasis spanning chunker segments leaves
    unpaired `**` in a fragment — CommonMark renders those literally, but
    the speech contract is stronger: formatting characters are NEVER
    spoken."""

    def test_unpaired_bold_at_segment_start_is_scrubbed(self) -> None:
        # Chunker split inside a bold span: this fragment has an opening
        # ** with no closing pair.
        assert markdown_to_speech("**Generate a summary of this.") == (
            "Generate a summary of this."
        )

    def test_unpaired_bold_at_segment_end_is_scrubbed(self) -> None:
        assert markdown_to_speech("and that concludes it.**") == "and that concludes it."

    def test_bold_spanning_two_segments(self) -> None:
        f = MarkdownSpeechFilter()
        assert f.convert("**Generate a summary.") == "Generate a summary."
        assert f.convert("Then continue normally.**") == "Then continue normally."

    def test_unpaired_single_asterisk_hugging_word(self) -> None:
        assert markdown_to_speech("*Generate a report") == "Generate a report"
        assert markdown_to_speech("something odd* happened") == "something odd happened"

    def test_stray_backticks_scrubbed(self) -> None:
        assert markdown_to_speech("run `eva doctor now") == "run eva doctor now"

    def test_multiplication_and_snake_case_survive_the_scrub(self) -> None:
        assert markdown_to_speech("3 * 4 equals 12") == "3 * 4 equals 12"
        assert markdown_to_speech("open file_name_here please") == "open file_name_here please"


class TestNestedFormatting:
    def test_bold_containing_italic(self) -> None:
        assert markdown_to_speech("**bold with *nested* inside**") == "bold with nested inside"

    def test_italic_containing_code(self) -> None:
        assert markdown_to_speech("*emphasis with `code` inside*") == "emphasis with code inside"

    def test_bold_inside_list_item(self) -> None:
        assert markdown_to_speech("1. **First** item") == "1. First item"

    def test_link_text_with_emphasis(self) -> None:
        assert markdown_to_speech("[**bold link**](https://x.dev)") == "bold link"


class TestHtmlEntities:
    def test_common_entities_decoded(self) -> None:
        assert markdown_to_speech("Tom &amp; Jerry") == "Tom & Jerry"
        assert markdown_to_speech("5 &lt; 10 &gt; 2") == "5 < 10 > 2"

    def test_nbsp_becomes_space(self) -> None:
        # &nbsp; decodes to U+00A0; it must not be spoken as words.
        spoken = markdown_to_speech("a&nbsp;b")
        assert "nbsp" not in spoken

    def test_numeric_entities(self) -> None:
        assert markdown_to_speech("caf&#233;") == "café"


class TestAutolinksAndUrls:
    def test_autolink_keeps_url_text(self) -> None:
        assert markdown_to_speech("<https://example.com>") == "https://example.com"

    def test_link_with_title_attribute(self) -> None:
        assert markdown_to_speech('[docs](https://example.com "the title")') == "docs"


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
