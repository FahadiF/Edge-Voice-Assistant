"""Per-turn latency metrics.

The reference timeline starts at utterance end (the moment the user stops
speaking) because that is what perceived responsiveness is measured against:

  utterance end ── asr_ms ── ttft_ms ── … ── ttfa_ms (first audio queued)
"""

from __future__ import annotations

import statistics

from pydantic import BaseModel, ConfigDict


class TurnMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    epoch: int
    asr_ms: int = 0
    ttft_ms: int = 0  # utterance end → first LLM token
    llm_ms: int = 0
    tokens: int = 0
    tts_first_ms: int = 0  # first sentence synthesis time
    ttfa_ms: int = 0  # utterance end → first audio queued
    total_ms: int = 0  # utterance end → playback drained
    cancelled: bool = False

    @property
    def tokens_per_s(self) -> float:
        return self.tokens / (self.llm_ms / 1000) if self.llm_ms > 0 else 0.0


class MetricsCollector:
    def __init__(self) -> None:
        self._turns: list[TurnMetrics] = []

    def record(self, metrics: TurnMetrics) -> None:
        self._turns.append(metrics)

    @property
    def turns(self) -> list[TurnMetrics]:
        return list(self._turns)

    def summary(self) -> str:
        completed = [t for t in self._turns if not t.cancelled and t.ttfa_ms > 0]
        if not completed:
            return "No completed turns."

        def med(values: list[int]) -> int:
            return int(statistics.median(values))

        lines = [
            f"Turns: {len(self._turns)} ({len(completed)} completed)",
            f"ASR (median):               {med([t.asr_ms for t in completed])} ms",
            f"Time to first token:        {med([t.ttft_ms for t in completed])} ms",
            f"First-sentence TTS:         {med([t.tts_first_ms for t in completed])} ms",
            f"Time to first audio:        {med([t.ttfa_ms for t in completed])} ms",
            f"LLM speed (median):         "
            f"{statistics.median([t.tokens_per_s for t in completed]):.1f} tokens/s",
        ]
        return "\n".join(lines)
