from __future__ import annotations

from eva.conversation.chunker import SentenceChunker


def feed_all(chunker: SentenceChunker, text: str, piece: int = 3) -> list[str]:
    """Feed text in small token-like pieces; collect emitted segments."""
    out: list[str] = []
    for i in range(0, len(text), piece):
        out.extend(chunker.feed(text[i : i + piece]))
    return out


def test_simple_sentences() -> None:
    c = SentenceChunker(min_chars=5)
    segments = feed_all(c, "Hello there. How are you today? I am fine!")
    assert segments == ["Hello there.", "How are you today?", "I am fine!"]
    assert c.flush() is None


def test_flush_returns_tail() -> None:
    c = SentenceChunker()
    feed_all(c, "This sentence never ends")
    assert c.flush() == "This sentence never ends"


def test_min_chars_defers_tiny_fragments() -> None:
    c = SentenceChunker(min_chars=12)
    segments = feed_all(c, "Hi. It works fine now.")
    # "Hi." alone is below min_chars, so it merges with the next sentence.
    assert segments == ["Hi. It works fine now."]


def test_abbreviations_do_not_split() -> None:
    c = SentenceChunker(min_chars=5)
    segments = feed_all(c, "Dr. Smith visited Mr. Jones yesterday. They talked.")
    assert segments == ["Dr. Smith visited Mr. Jones yesterday.", "They talked."]


def test_decimals_do_not_split() -> None:
    c = SentenceChunker(min_chars=5)
    segments = feed_all(c, "The value of pi is approximately 3. 14159 is wrong. Done here.")
    # "…approximately 3." is protected (digit before the period)
    assert segments[0] == "The value of pi is approximately 3. 14159 is wrong."


def test_single_initials_do_not_split() -> None:
    c = SentenceChunker(min_chars=5)
    segments = feed_all(c, "John F. Kennedy was president. Yes he was.")
    assert segments == ["John F. Kennedy was president.", "Yes he was."]


def test_question_and_exclamation() -> None:
    c = SentenceChunker(min_chars=3)
    assert feed_all(c, "Really? Absolutely! Good.") == ["Really?", "Absolutely!", "Good."]


def test_force_split_on_runaway_text() -> None:
    c = SentenceChunker(max_chars=50)
    long_text = "word " * 30  # 150 chars, no sentence punctuation
    segments = feed_all(c, long_text)
    assert segments  # must not stall waiting forever
    assert all(len(s) <= 60 for s in segments)


def test_force_split_prefers_clause_boundary() -> None:
    c = SentenceChunker(max_chars=40)
    segments = feed_all(c, "alpha beta gamma delta, epsilon zeta eta theta iota kappa")
    assert segments[0].endswith(",")


def test_reset_clears_buffer() -> None:
    c = SentenceChunker()
    c.feed("partial text")
    c.reset()
    assert c.flush() is None


def test_multibyte_and_cjk_punctuation() -> None:
    c = SentenceChunker(min_chars=2)
    segments = feed_all(c, "你好。今天怎么样？")  # noqa: RUF001
    assert len(segments) >= 1


def test_first_chunk_min_chars_applies_only_to_first_segment() -> None:
    # min_chars=12 would defer "Hi." into the next sentence; a lower
    # first_chunk_min_chars lets the first segment fire on its own so the
    # first sound reaches the speaker sooner (M3/ADR-018).
    c = SentenceChunker(min_chars=12, first_chunk_min_chars=2)
    segments = feed_all(c, "Hi. It works fine now. Another sentence follows.")
    assert segments[0] == "Hi."
    # Second sentence is short too, but first_chunk_min_chars no longer
    # applies — it falls back to the normal min_chars threshold.
    assert segments[1] == "It works fine now."


def test_first_chunk_min_chars_none_behaves_like_before() -> None:
    c = SentenceChunker(min_chars=12, first_chunk_min_chars=None)
    segments = feed_all(c, "Hi. It works fine now.")
    assert segments == ["Hi. It works fine now."]


def test_reset_restores_first_chunk_threshold() -> None:
    c = SentenceChunker(min_chars=12, first_chunk_min_chars=2)
    feed_all(c, "Hi. ")
    c.reset()
    segments = feed_all(c, "Hi. It works fine now.")
    # After reset, the next turn's first segment is eligible again.
    assert segments[0] == "Hi."
