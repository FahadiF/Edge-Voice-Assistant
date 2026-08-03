"""OpenAI-compatible chat-completions adapter (Batch 8, M7.4).

One adapter reaches Ollama, LM Studio, vLLM, and any other server exposing an
OpenAI-shaped `/chat/completions` streaming endpoint — the milestone's stated
deliverable, not a review finding.

**Local-only this milestone (decision 8.3, Final Execution Roadmap, frozen).**
The constructor rejects a non-loopback `base_url` outright. Batch 6 made
`eva.core.net` the sole egress point and pinned `urllib.request` to four named
files via an import-direction test (`tests/test_offline_invariant.py`);
Ollama/LM Studio/vLLM run on `127.0.0.1`, which is loopback IPC, not egress —
so this module is added to that allowlist as a fifth loopback caller rather
than routed through `core.net`. An actual remote endpoint is explicitly
deferred to M7.5 (Online Mode), which brings its own connection-mode and
privacy controls that this batch does not build.

Does not implement `LocalWeights`: no local file, no `device`, no load/unload
step. `is_local()` returns False for this adapter; `engine_device()` reports
"remote" for diagnostics and the runtime-awareness prompt.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from collections.abc import Callable, Generator

from eva.core.errors import ModelError
from eva.core.events import FinishReason
from eva.core.net import is_loopback_url
from eva.core.secrets import resolve_secret
from eva.core.tools import ToolDefinition
from eva.llm.base import ChatMessage, GenerationOutcome, GenerationParams, LLMEngine

logger = logging.getLogger(__name__)


class OpenAICompatibleLLM(LLMEngine):
    """Streams from a `/chat/completions`-compatible endpoint over SSE."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key_ref: str | None = None,
        timeout_s: float = 120.0,
    ) -> None:
        if not is_loopback_url(base_url):
            raise ModelError(
                f"OpenAI-compatible provider base_url must be a local address "
                f"(127.0.0.1/localhost) this milestone; got {base_url!r}. "
                "Remote endpoints are M7.5 (Online Mode) territory."
            )
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key_ref = api_key_ref
        self._timeout_s = timeout_s

    def stream(
        self,
        messages: list[ChatMessage],
        params: GenerationParams,
        should_abort: Callable[[], bool],
        *,
        tools: tuple[ToolDefinition, ...] = (),
    ) -> Generator[str, None, GenerationOutcome]:
        if tools:
            # Same honesty rule as the llama.cpp adapter: say so rather than
            # generate a tool-less answer that looks like a declined offer.
            logger.warning(
                "This adapter cannot offer tools to the model yet; generating "
                "without the %d offered (%s)",
                len(tools),
                ", ".join(t.name for t in tools),
            )
        headers = {"Content-Type": "application/json"}
        if self._api_key_ref:
            # Resolved immediately before the call, never stored — see
            # eva.core.secrets. The reference travels through settings; the
            # value never does.
            headers["Authorization"] = f"Bearer {resolve_secret(self._api_key_ref)}"
        payload = {
            "model": self._model,
            "messages": [m.model_dump(exclude_none=True) for m in messages],
            "temperature": params.temperature,
            "top_p": params.top_p,
            "max_tokens": params.max_tokens,
            "stop": list(params.stop) or None,
            "stream": True,
        }
        request = urllib.request.Request(
            f"{self._base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        # Server reports the reason on the last streamed chunk only; default
        # to "stop" so a server that never reports one reads as complete —
        # mirrors the llama.cpp adapter's same default for the same reason.
        reason: FinishReason = "stop"
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
                for raw_line in response:
                    if should_abort():
                        logger.debug("LLM generation aborted")
                        return GenerationOutcome(reason="abort")
                    line = raw_line.decode("utf-8").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choice = (chunk.get("choices") or [{}])[0]
                    reported = choice.get("finish_reason")
                    if reported in ("stop", "length"):
                        reason = reported
                    token = (choice.get("delta") or {}).get("content")
                    if token:
                        yield token
        except (urllib.error.URLError, OSError) as exc:
            raise ModelError(f"OpenAI-compatible request failed: {exc}") from exc
        if reason == "length":
            logger.info(
                "Generation hit the %d-token ceiling; reply is truncated", params.max_tokens
            )
        return GenerationOutcome(reason=reason)
