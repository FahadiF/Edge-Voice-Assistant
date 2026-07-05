"""ONNX embedding provider tests (ADR-020) — fake tokenizer/session, no real
model files needed. Bypasses `load()`'s deferred imports the same way M3's
Kokoro streaming tests bypass `kokoro_onnx.Kokoro` (assign fakes directly to
the private attributes `load()` would otherwise populate).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from eva.core.errors import ModelError
from eva.embedding.onnx import OnnxEmbeddingProvider


class _FakeEncoding:
    def __init__(self, ids: list[int], attention_mask: list[int]) -> None:
        self.ids = ids
        self.attention_mask = attention_mask


class _FakeTokenizer:
    def __init__(self, ids: list[int] | None = None) -> None:
        self._ids = ids or [101, 2054, 2003, 102]  # arbitrary 4 "tokens"
        self.truncation_max_length: int | None = None

    def enable_truncation(self, max_length: int) -> None:
        self.truncation_max_length = max_length

    def encode(self, text: str) -> _FakeEncoding:
        return _FakeEncoding(ids=list(self._ids), attention_mask=[1] * len(self._ids))


class _FakeSession:
    def __init__(
        self,
        hidden: np.ndarray,
        input_names: tuple[str, ...] = ("input_ids", "attention_mask"),
        fail: bool = False,
    ) -> None:
        self._hidden = hidden
        self._input_names = input_names
        self._fail = fail
        self.last_feed: dict[str, np.ndarray] | None = None

    def get_inputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name=n) for n in self._input_names]

    def get_outputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(shape=["batch", "sequence", self._hidden.shape[-1]])]

    def run(self, _output_names: object, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        if self._fail:
            raise RuntimeError("onnxruntime exploded")
        self.last_feed = feed
        seq_len = feed["input_ids"].shape[1]
        return [self._hidden[None, :seq_len, :]]


def _make_provider(
    tmp_path: Path, tokenizer: _FakeTokenizer, session: _FakeSession
) -> OnnxEmbeddingProvider:
    provider = OnnxEmbeddingProvider(tmp_path / "model.onnx", tmp_path / "tokenizer.json")
    provider._session = session  # bypass load()'s deferred onnxruntime/tokenizers imports
    provider._tokenizer = tokenizer
    provider._input_names = {i.name for i in session.get_inputs()}
    provider.dim = session._hidden.shape[-1]
    provider.device = "cpu"
    return provider


class TestLoad:
    def test_missing_model_file_raises(self, tmp_path: Path) -> None:
        (tmp_path / "tokenizer.json").write_text("{}")
        provider = OnnxEmbeddingProvider(tmp_path / "model.onnx", tmp_path / "tokenizer.json")
        with pytest.raises(ModelError, match="not found"):
            provider.load()

    def test_missing_tokenizer_file_raises(self, tmp_path: Path) -> None:
        (tmp_path / "model.onnx").write_bytes(b"not a real onnx file")
        provider = OnnxEmbeddingProvider(tmp_path / "model.onnx", tmp_path / "tokenizer.json")
        with pytest.raises(ModelError, match="not found"):
            provider.load()

    def test_unload_clears_session(self, tmp_path: Path) -> None:
        hidden = np.random.default_rng(0).random((4, 384)).astype(np.float32)
        provider = _make_provider(tmp_path, _FakeTokenizer(), _FakeSession(hidden))
        provider.unload()
        assert provider._session is None
        assert provider._tokenizer is None


class TestEmbed:
    def test_empty_text_returns_zero_vector_of_correct_dim(self, tmp_path: Path) -> None:
        hidden = np.zeros((4, 384), dtype=np.float32)
        provider = _make_provider(tmp_path, _FakeTokenizer(), _FakeSession(hidden))
        vector = provider.embed("   ")
        assert vector.shape == (384,)
        assert np.all(vector == 0)

    def test_embed_returns_l2_normalized_vector(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(42)
        hidden = rng.random((4, 384)).astype(np.float32) * 10  # not pre-normalized
        provider = _make_provider(tmp_path, _FakeTokenizer(), _FakeSession(hidden))
        vector = provider.embed("hello world")
        assert vector.dtype == np.float32
        assert vector.shape == (384,)
        assert np.isclose(np.linalg.norm(vector), 1.0, atol=1e-5)

    def test_padding_tokens_excluded_from_mean_pool(self, tmp_path: Path) -> None:
        # Token 0 and 1 are "real"; token 2 is padding (attention_mask=0) with an
        # extreme value that must NOT influence the pooled result.
        hidden = np.array(
            [[1.0] * 4, [3.0] * 4, [1000.0] * 4, [1000.0] * 4], dtype=np.float32
        )

        class _PaddedTokenizer(_FakeTokenizer):
            def encode(self, text: str) -> _FakeEncoding:
                return _FakeEncoding(ids=[10, 20, 0, 0], attention_mask=[1, 1, 0, 0])

        session = _FakeSession(hidden, input_names=("input_ids", "attention_mask"))
        provider = _make_provider(tmp_path, _PaddedTokenizer(), session)
        vector = provider.embed("real tokens only")
        # Mean of [1,1,1,1] and [3,3,3,3] (masked positions excluded) = [2,2,2,2],
        # then L2-normalized -> every component equal.
        assert np.allclose(vector, vector[0])

    def test_token_type_ids_included_when_model_expects_them(self, tmp_path: Path) -> None:
        hidden = np.ones((4, 384), dtype=np.float32)
        session = _FakeSession(
            hidden, input_names=("input_ids", "attention_mask", "token_type_ids")
        )
        provider = _make_provider(tmp_path, _FakeTokenizer(), session)
        provider.embed("hello")
        assert session.last_feed is not None
        assert "token_type_ids" in session.last_feed
        assert np.all(session.last_feed["token_type_ids"] == 0)

    def test_token_type_ids_omitted_when_model_does_not_expect_them(
        self, tmp_path: Path
    ) -> None:
        hidden = np.ones((4, 384), dtype=np.float32)
        session = _FakeSession(hidden, input_names=("input_ids", "attention_mask"))
        provider = _make_provider(tmp_path, _FakeTokenizer(), session)
        provider.embed("hello")
        assert session.last_feed is not None
        assert "token_type_ids" not in session.last_feed

    def test_inference_error_wrapped_as_model_error(self, tmp_path: Path) -> None:
        hidden = np.ones((4, 384), dtype=np.float32)
        session = _FakeSession(hidden, fail=True)
        provider = _make_provider(tmp_path, _FakeTokenizer(), session)
        with pytest.raises(ModelError, match="inference failed"):
            provider.embed("hello")

    def test_same_text_yields_identical_vector(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(7)
        hidden = rng.random((4, 384)).astype(np.float32)
        provider = _make_provider(tmp_path, _FakeTokenizer(), _FakeSession(hidden))
        v1 = provider.embed("deterministic")
        v2 = provider.embed("deterministic")
        assert np.array_equal(v1, v2)
