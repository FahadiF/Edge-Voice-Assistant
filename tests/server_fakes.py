"""Fake engine components shared by server tests needing a "running" engine
without real models or audio hardware — same fakes used in test_orchestrator.py,
combined into an Assistant-shaped object for `ServerState.start_engine()`.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from typing import Any

import numpy as np

from eva.asr.base import ASREngine, TranscriptionResult
from eva.audio.frames import Frame
from eva.config.paths import AppPaths
from eva.config.settings import Settings
from eva.conversation.orchestrator import Orchestrator
from eva.core.errors import MemoryNotFoundError
from eva.core.events import EventBus
from eva.engine import Assistant
from eva.llm.base import ChatMessage, GenerationParams, LLMEngine
from eva.memory.base import MemoryStore, UserProfileStore
from eva.memory.models import (
    MemoryConversation,
    MemorySearchResult,
    MemoryStats,
    MemorySummary,
    MemoryTurn,
    Speaker,
    UserProfile,
)
from eva.tts.base import TTSEngine


class FakeASR(ASREngine):
    device = "cpu"

    def __init__(self, text: str = "hello") -> None:
        self.text = text

    def load(self) -> None: ...
    def unload(self) -> None: ...

    def transcribe(self, audio: Frame, language: str | None = None) -> TranscriptionResult:
        return TranscriptionResult(text=self.text)


class FakeLLM(LLMEngine):
    device = "cpu"

    def load(self) -> None: ...
    def unload(self) -> None: ...

    def stream(
        self,
        messages: list[ChatMessage],
        params: GenerationParams,
        should_abort: Callable[[], bool],
    ) -> Iterator[str]:
        for token in ["Hello", " there."]:
            if should_abort():
                return
            yield token


class FakeTTS(TTSEngine):
    device = "cpu"

    def load(self) -> None: ...
    def unload(self) -> None: ...

    def synthesize(self, text: str, *, voice: str, speed: float = 1.0) -> Frame:
        return np.ones(1600, dtype=np.int16)

    def voices(self) -> list[str]:
        return ["test-voice"]


class _FakePipeline:
    level_dbfs = -40.0


class _FakeRing:
    dropped = 0

    def __len__(self) -> int:
        return 0


class _FakePlayback:
    def queued_seconds(self) -> float:
        return 0.0


class FakeAudioSystem:
    """Satisfies both the orchestrator's AudioOutput protocol and the
    lifecycle/diagnostics surface `Assistant.audio` needs."""

    def __init__(self) -> None:
        self.spoken: list[Frame] = []
        self.pipeline = _FakePipeline()
        self.capture_ring = _FakeRing()
        self.playback = _FakePlayback()
        self._speaking = False

    # AudioOutput protocol (used by the orchestrator)
    def say(self, pcm: Frame) -> None:
        self.spoken.append(pcm)

    def finish_utterance(self) -> None:
        pass

    def stop_speaking(self) -> None:
        self._speaking = False

    @property
    def is_speaking(self) -> bool:
        return self._speaking

    # Lifecycle (used by Assistant.start_audio/stop)
    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


class FakeMemoryStore(MemoryStore):
    """Pure in-process `MemoryStore` — no SQLite, no file I/O. Implements
    the full port so it's a drop-in for `Orchestrator`/`ContextBuilder`
    tests that don't care about persistence itself (that's what
    `test_memory_sqlite.py` covers against the real adapter)."""

    def __init__(self) -> None:
        self._conversations: dict[str, MemoryConversation] = {}
        self._turns: dict[int, MemoryTurn] = {}
        self._summaries: dict[str, MemorySummary] = {}
        self._embeddings: dict[int, tuple[bytes, int]] = {}
        self._next_id = 1

    def start_conversation(self, *, language: str = "en", title: str = "") -> MemoryConversation:
        conversation_id = str(uuid.uuid4())
        conv = MemoryConversation(
            id=conversation_id, started_at=datetime.now(UTC), title=title, language=language
        )
        self._conversations[conversation_id] = conv
        return conv

    def all_conversations(self, *, include_archived: bool = False) -> list[MemoryConversation]:
        return [c for c in self._conversations.values() if include_archived or not c.archived]

    def archive_conversation(self, conversation_id: str, *, archived: bool = True) -> None:
        conv = self._conversations[conversation_id]
        self._conversations[conversation_id] = conv.model_copy(update={"archived": archived})

    def delete_conversation(self, conversation_id: str) -> None:
        self._conversations.pop(conversation_id, None)
        self._turns = {
            tid: t for tid, t in self._turns.items() if t.conversation_id != conversation_id
        }

    def merge_conversations(self, source_id: str, target_id: str) -> None:
        for turn_id, turn in list(self._turns.items()):
            if turn.conversation_id == source_id:
                self._turns[turn_id] = turn.model_copy(update={"conversation_id": target_id})
        self._conversations.pop(source_id, None)

    def delete_all(self) -> None:
        self._conversations.clear()
        self._turns.clear()
        self._summaries.clear()

    def add_turn(
        self,
        conversation_id: str,
        speaker: Speaker,
        text: str,
        *,
        language: str = "en",
        metadata: dict[str, Any] | None = None,
    ) -> MemoryTurn:
        turn_id = self._next_id
        self._next_id += 1
        turn = MemoryTurn(
            id=turn_id,
            conversation_id=conversation_id,
            created_at=datetime.now(UTC),
            speaker=speaker,
            text=text,
            language=language,
            metadata=metadata or {},
        )
        self._turns[turn_id] = turn
        return turn

    def get_turn(self, turn_id: int) -> MemoryTurn:
        if turn_id not in self._turns:
            raise MemoryNotFoundError(f"No turn with id {turn_id}")
        return self._turns[turn_id]

    def get_turns(self, turn_ids: list[int]) -> list[MemoryTurn]:
        return [
            self._turns[tid]
            for tid in turn_ids
            if tid in self._turns and not self._turns[tid].deleted
        ]

    def _conversation_turns(self, conversation_id: str) -> list[MemoryTurn]:
        matching = [
            t
            for t in self._turns.values()
            if t.conversation_id == conversation_id and not t.deleted
        ]
        matching.sort(key=lambda t: t.id or 0)
        return matching

    def recent_turns(self, conversation_id: str, limit: int) -> list[MemoryTurn]:
        return self._conversation_turns(conversation_id)[-limit:]

    def all_turns(self, conversation_id: str) -> list[MemoryTurn]:
        return self._conversation_turns(conversation_id)

    def forget(self, turn_id: int) -> None:
        if turn_id not in self._turns:
            raise MemoryNotFoundError(f"No turn with id {turn_id}")
        del self._turns[turn_id]

    def pin(self, turn_id: int, *, pinned: bool = True) -> None:
        self._turns[turn_id] = self._turns[turn_id].model_copy(update={"pinned": pinned})

    def favorite(self, turn_id: int, *, favorite: bool = True) -> None:
        self._turns[turn_id] = self._turns[turn_id].model_copy(update={"favorite": favorite})

    def search_text(
        self, query: str, *, limit: int = 20, conversation_id: str | None = None
    ) -> list[MemorySearchResult]:
        matches = [
            t
            for t in self._turns.values()
            if query.lower() in t.text.lower()
            and not t.deleted
            and (conversation_id is None or t.conversation_id == conversation_id)
        ]
        return [
            MemorySearchResult(turn=t, score=1.0, match_reason="keyword") for t in matches[:limit]
        ]

    def set_title(self, conversation_id: str, title: str) -> None:
        conv = self._conversations.get(conversation_id)
        if conv is None:
            from eva.core.errors import MemoryNotFoundError

            raise MemoryNotFoundError(f"No conversation with id {conversation_id!r}")
        self._conversations[conversation_id] = conv.model_copy(update={"title": title.strip()})

    def store_embedding(self, turn_id: int, model_id: str, vector: bytes, dim: int) -> None:
        self._embeddings[turn_id] = (vector, dim)

    def embeddings_for(
        self, conversation_id: str | None = None, *, limit: int | None = None
    ) -> list[tuple[int, bytes, int]]:
        rows = [
            (turn_id, vector, dim)
            for turn_id, (vector, dim) in self._embeddings.items()
            if conversation_id is None
            or (
                self._turns.get(turn_id, None) is not None
                and self._turns[turn_id].conversation_id == conversation_id
            )
        ]
        return rows[-limit:] if limit is not None else rows

    def add_summary(self, summary: MemorySummary) -> MemorySummary:
        saved = summary.model_copy(update={"id": len(self._summaries) + 1})
        self._summaries[summary.conversation_id] = saved
        return saved

    def latest_summary(self, conversation_id: str) -> MemorySummary | None:
        return self._summaries.get(conversation_id)

    def export_json(self, conversation_id: str | None = None) -> dict[str, Any]:
        if conversation_id is not None:
            conversations = (
                [self._conversations[conversation_id]]
                if conversation_id in self._conversations
                else []
            )
        else:
            conversations = list(self._conversations.values())
        entries = [
            {
                "conversation": conv.model_dump(mode="json"),
                "turns": [t.model_dump(mode="json") for t in self._conversation_turns(conv.id)],
                "summaries": [
                    s.model_dump(mode="json")
                    for s in self._summaries.values()
                    if s.conversation_id == conv.id
                ],
            }
            for conv in conversations
        ]
        return {"version": 1, "conversations": entries}

    def import_json(self, payload: dict[str, Any]) -> int:
        imported = 0
        for entry in payload.get("conversations", []):
            conv_data = entry["conversation"]
            conv_id = conv_data["id"]
            if conv_id not in self._conversations:
                self._conversations[conv_id] = MemoryConversation(
                    id=conv_id,
                    started_at=conv_data["started_at"],
                    title=conv_data.get("title", ""),
                    language=conv_data.get("language", "en"),
                    archived=bool(conv_data.get("archived", False)),
                )
            for turn_data in entry.get("turns", []):
                self.add_turn(
                    conv_id,
                    turn_data["speaker"],
                    turn_data["text"],
                    language=turn_data.get("language", "en"),
                    metadata=turn_data.get("metadata") or {},
                )
                imported += 1
        return imported

    def stats(self) -> MemoryStats:
        return MemoryStats(
            conversation_count=len(self._conversations),
            turn_count=len(self._turns),
            embedded_turn_count=0,
            summary_count=len(self._summaries),
            db_size_bytes=0,
            fts_enabled=False,
        )

    def close(self) -> None:
        pass


class FakeUserProfileStore(UserProfileStore):
    def __init__(self) -> None:
        self._profiles: dict[str, UserProfile] = {}
        self._active_id: str | None = None

    def create(self, profile: UserProfile) -> UserProfile:
        self._profiles[profile.id] = profile
        return profile

    def get(self, profile_id: str) -> UserProfile:
        if profile_id not in self._profiles:
            raise MemoryNotFoundError(f"No user profile with id {profile_id!r}")
        return self._profiles[profile_id]

    def list(self) -> list[UserProfile]:
        return list(self._profiles.values())

    def update(self, profile: UserProfile) -> UserProfile:
        if profile.id not in self._profiles:
            raise MemoryNotFoundError(f"No user profile with id {profile.id!r}")
        self._profiles[profile.id] = profile
        return profile

    def set_active(self, profile_id: str) -> None:
        self.get(profile_id)
        self._active_id = profile_id

    def active(self) -> UserProfile | None:
        return self._profiles.get(self._active_id) if self._active_id else None

    def delete(self, profile_id: str) -> None:
        if profile_id not in self._profiles:
            raise MemoryNotFoundError(f"No user profile with id {profile_id!r}")
        del self._profiles[profile_id]


def build_fake_assistant(
    settings: Settings, _paths: AppPaths, bus: EventBus | None = None
) -> Assistant:
    """Drop-in replacement for `eva.engine.build_assistant` in server tests."""
    bus = bus or EventBus()
    audio = FakeAudioSystem()
    asr, llm, tts = FakeASR(), FakeLLM(), FakeTTS()
    memory = FakeMemoryStore()
    orchestrator = Orchestrator(settings, bus, audio, asr, llm, tts, memory)
    return Assistant(
        settings=settings,
        bus=bus,
        audio=audio,
        orchestrator=orchestrator,
        asr=asr,
        llm=llm,
        tts=tts,
        memory=memory,
        profiles=FakeUserProfileStore(),
        embedding=None,
    )
