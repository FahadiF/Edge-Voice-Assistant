"""Extract a name the user states about themselves, for session identity.

Why this exists: without a stored profile nickname, the user's name reaches the
prompt only via the recent-turn window (bounded by `max_history_turns`) or
query-dependent memory retrieval — so once the naming turn scrolls out of the
window, the assistant "knows" the name for name-related questions but not for
unrelated ones, and contradicts itself within a session. Capturing an
explicitly-stated name into a session fact (injected into the system prompt
every turn thereafter) makes it deterministic and consistent.

Deliberately conservative: only clear first-person self-naming phrases, and the
captured token is validated as name-like, with a small block-list for the
common false positives ("my name is not …", "my name is hard to say"). Better
to miss an unusual phrasing than to start calling the user the wrong thing.
"""

from __future__ import annotations

import re

# One capture group: the token immediately following an explicit self-naming
# phrase. An optional filler ("actually"/"really"/"now") is consumed so
# "my name is actually John" still yields "John". The token allows internal
# apostrophes/hyphens (O'Brien, Anne-Marie); the char class is built via a
# named escape so a literal smart-quote never appears in source.
_APOS = "'" + "\N{RIGHT SINGLE QUOTATION MARK}"
_NAME_TOKEN = rf"([A-Za-z][A-Za-z{_APOS}-]{{1,29}})"
_FILLER = r"(?:actually |really |now |just )?"
_PATTERNS = [
    re.compile(rf"\bmy name is {_FILLER}{_NAME_TOKEN}", re.IGNORECASE),
    re.compile(rf"\bmy name[{_APOS}]?s {_FILLER}{_NAME_TOKEN}", re.IGNORECASE),
    re.compile(rf"\byou can call me {_NAME_TOKEN}", re.IGNORECASE),
    re.compile(rf"\bplease call me {_NAME_TOKEN}", re.IGNORECASE),
    re.compile(rf"\bcall me {_NAME_TOKEN}", re.IGNORECASE),
    re.compile(rf"\bi go by {_NAME_TOKEN}", re.IGNORECASE),
]

# Words that follow a naming phrase but are not names — the common false
# positives. Kept lowercase; comparison is case-insensitive.
_NOT_A_NAME = frozenset(
    {
        "not",
        "a",
        "an",
        "the",
        "so",
        "kind",
        "sort",
        "hard",
        "easy",
        "difficult",
        "unusual",
        "common",
        "long",
        "short",
        "spelled",
        "spelt",
        "pronounced",
        "written",
        "actually",
        "really",
        "just",
        "now",
        "gonna",
        "going",
        "kinda",
        "sorta",
        "none",
        "nobody",
        "no",
        "yes",
        "here",
        "there",
        "important",
        "secret",
        "private",
        "confidential",
        "irrelevant",
        "unknown",
    }
)


def _normalize(token: str) -> str:
    """Title-case a captured token conservatively: capitalize a first letter for
    an all-lowercase token, otherwise keep the user's own casing (McDonald)."""
    return token.capitalize() if token.islower() else token


def extract_stated_name(text: str) -> str | None:
    """Return a name the user explicitly stated about themselves, or None.

    Conservative by design (see module docstring) — only clear self-naming
    phrases, validated against a false-positive block-list. Returns the first
    match found, normalized (e.g. "fahad" → "Fahad").
    """
    for pattern in _PATTERNS:
        match = pattern.search(text)
        if match is None:
            continue
        token = match.group(1)
        if token.lower() in _NOT_A_NAME:
            continue
        return _normalize(token)
    return None
