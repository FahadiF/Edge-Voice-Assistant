"""Tests for conversational name capture (`identity.extract_stated_name`).

The point of this feature is consistency, so the tests weigh false-positive
rejection as heavily as extraction — a wrong capture would make the assistant
call the user by the wrong name for the whole session.
"""

from __future__ import annotations

import pytest

from eva.conversation.identity import extract_stated_name


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hi, my name is Fahad.", "Fahad"),
        ("my name is fahad", "Fahad"),  # normalized to title case
        ("My name's Sam", "Sam"),
        ("You can call me Alex please", "Alex"),
        ("please call me Sam", "Sam"),
        ("call me Riya", "Riya"),
        ("I go by Max", "Max"),
        ("my name is actually John", "John"),  # filler skipped
        ("my name's really Priya", "Priya"),
        ("My name is Anne-Marie", "Anne-Marie"),  # hyphenated
        ("my name is O'Brien", "O'Brien"),  # apostrophe kept
    ],
)
def test_extracts_explicit_self_naming(text: str, expected: str) -> None:
    assert extract_stated_name(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "my name is not important",
        "my name is a secret",
        "my name is hard to pronounce",
        "my name is difficult to spell",
        "what is my name?",
        "do you remember my name",
        "tell me about the weather",
        "the name of the file is report",  # not first-person self-naming
        "her name is Sarah",  # about someone else
        "the product name is EVA",
        "",
    ],
)
def test_rejects_non_names_and_false_positives(text: str) -> None:
    assert extract_stated_name(text) is None


def test_returns_first_match_when_multiple() -> None:
    assert extract_stated_name("my name is Fahad, call me F") == "Fahad"
