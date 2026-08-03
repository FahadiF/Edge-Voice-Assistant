"""Both LLM adapters must expose the same failure contract (Batch 8.1).

`LlamaCppLLM.stream()` and `OpenAICompatibleLLM.stream()` fail for entirely
different reasons (a native runtime exception vs. a network error), but a
caller of the transport-neutral `LLMEngine` port should see one exception
type either way. These construct each adapter with its real lifecycle intact
and force a failure mid-generation — not at `load()`/construction, which both
adapters already handled correctly before this batch.
"""

from __future__ import annotations

import urllib.error
from pathlib import Path
from typing import Any

import pytest

from eva.core.errors import ModelError
from eva.llm.base import ChatMessage, GenerationParams
from eva.llm.llamacpp import LlamaCppLLM
from eva.llm.openai_compat import OpenAICompatibleLLM

_MESSAGES = [ChatMessage(role="user", content="hi")]
_PARAMS = GenerationParams()


class _CrashingCompletion:
    """Stands in for whatever `llama_cpp.Llama.create_chat_completion`
    returns: an iterable that fails partway through, exactly like a native
    crash or an internal llama.cpp error would."""

    def __iter__(self) -> _CrashingCompletion:
        return self

    def __next__(self) -> dict[str, Any]:
        raise RuntimeError("native generation failure")

    def close(self) -> None:
        pass


class _AbortableCompletion:
    """A completion that would keep yielding chunks forever — proves abort is
    still checked and returned as `GenerationOutcome(reason="abort")` rather
    than being caught by the new `except Exception` and misreported as a
    `ModelError`, since `should_abort` returning True is not an exception at
    all."""

    def __iter__(self) -> _AbortableCompletion:
        return self

    def __next__(self) -> dict[str, Any]:
        return {"choices": [{"delta": {"content": "x"}, "finish_reason": None}]}

    def close(self) -> None:
        pass


class TestLlamaCppStreamErrorContract:
    def test_generation_failure_raises_model_error(self) -> None:
        llm = LlamaCppLLM(Path("model.gguf"))
        llm._llama = type(
            "FakeLlama", (), {"create_chat_completion": lambda self, **kw: _CrashingCompletion()}
        )()

        with pytest.raises(ModelError, match="llama\\.cpp generation failed"):
            list(llm.stream(_MESSAGES, _PARAMS, should_abort=lambda: False))

    def test_abort_is_unaffected_by_the_error_wrapping(self) -> None:
        llm = LlamaCppLLM(Path("model.gguf"))
        llm._llama = type(
            "FakeLlama", (), {"create_chat_completion": lambda self, **kw: _AbortableCompletion()}
        )()

        gen = llm.stream(_MESSAGES, _PARAMS, should_abort=lambda: True)
        with pytest.raises(StopIteration) as exc_info:
            next(gen)
        assert exc_info.value.value.reason == "abort"


class TestOpenAICompatibleStreamErrorContract:
    def test_generation_failure_raises_model_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _raise_urlopen(*args: object, **kwargs: object) -> None:
            raise urllib.error.URLError("connection refused")

        monkeypatch.setattr("urllib.request.urlopen", _raise_urlopen)
        llm = OpenAICompatibleLLM(base_url="http://127.0.0.1:11434/v1", model="llama3")

        with pytest.raises(ModelError, match="OpenAI-compatible request failed"):
            list(llm.stream(_MESSAGES, _PARAMS, should_abort=lambda: False))


class TestBothAdaptersShareOneFailureType:
    """The point of this batch: not *that* each adapter raises something, but
    that both raise the exact same type for a mid-generation failure."""

    def test_both_adapters_raise_model_error_on_generation_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        llamacpp = LlamaCppLLM(Path("model.gguf"))
        llamacpp._llama = type(
            "FakeLlama", (), {"create_chat_completion": lambda self, **kw: _CrashingCompletion()}
        )()

        def _raise_urlopen(*args: object, **kwargs: object) -> None:
            raise OSError("network unreachable")

        monkeypatch.setattr("urllib.request.urlopen", _raise_urlopen)
        openai_compat = OpenAICompatibleLLM(base_url="http://127.0.0.1:11434/v1", model="llama3")

        for engine in (llamacpp, openai_compat):
            with pytest.raises(ModelError):
                list(engine.stream(_MESSAGES, _PARAMS, should_abort=lambda: False))
