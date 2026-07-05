"""`ConversationTurn`: the paired (user, assistant) shape the pre-M4
`/conversation/history|export|import` API contract uses.

Prompt composition and persistence moved to `MemoryStore` + `ContextBuilder`
(ADR-019, ADR-021) in M4 — `ConversationHistory` (an in-process list, lost on
restart) is gone. `pair_turns()` bridges the new speaker-granular
`MemoryTurn` storage back to the paired shape that API contract already
promises, so that surface does not change shape under M4.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from eva.memory.models import MemoryTurn


class ConversationTurn(BaseModel):
    model_config = ConfigDict(frozen=True)

    user: str
    assistant: str


def pair_turns(turns: list[MemoryTurn]) -> list[ConversationTurn]:
    """Pair consecutive (user, assistant) `MemoryTurn`s, oldest first, into
    `ConversationTurn`s. An unpaired turn (e.g. the last one, awaiting a
    reply) is skipped rather than guessed at."""
    pairs: list[ConversationTurn] = []
    i = 0
    while i < len(turns) - 1:
        if turns[i].speaker == "user" and turns[i + 1].speaker == "assistant":
            pairs.append(ConversationTurn(user=turns[i].text, assistant=turns[i + 1].text))
            i += 2
        else:
            i += 1
    return pairs
