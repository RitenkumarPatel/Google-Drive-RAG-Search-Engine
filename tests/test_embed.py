"""Tests for gdrive_rag.embed — SentenceTransformer model is faked (no download)."""

import math

import numpy as np
import pytest

from gdrive_rag import embed


class _Settings:
    local_embed_model = "BAAI/bge-base-en-v1.5"
    embed_delay = 0.0


class _FakeModel:
    """Fake SentenceTransformer: returns a deterministic non-zero 3-dim vector per text."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def encode(self, texts, *, normalize_embeddings, show_progress_bar):
        self.calls.append(list(texts))
        return np.array([[float(i % 5 + 1), 0.0, 0.0] for i in range(len(texts))])


@pytest.fixture(autouse=True)
def clear_model_cache():
    """Ensure model cache is empty before and after each test."""
    embed._MODEL_CACHE.clear()
    yield
    embed._MODEL_CACHE.clear()


def test_l2_normalize_unit():
    assert embed._l2_normalize([3.0, 4.0]) == [0.6, 0.8]


def test_l2_normalize_zero_vector():
    assert embed._l2_normalize([0.0, 0.0]) == [0.0, 0.0]


def test_embed_texts_empty():
    assert embed.embed_texts(_Settings(), []) == []


def test_embed_texts_batches_and_normalizes(monkeypatch):
    fake = _FakeModel()
    monkeypatch.setattr(embed, "get_model", lambda s: fake)

    out = embed.embed_texts(_Settings(), [f"t{i}" for i in range(150)], batch_size=64)

    assert len(out) == 150
    assert len(fake.calls) == 3          # 64 + 64 + 22
    assert len(fake.calls[0]) == 64
    assert len(fake.calls[1]) == 64
    assert len(fake.calls[2]) == 22
    for v in out:                        # every vector is unit-norm or the zero vector
        n = math.sqrt(sum(x * x for x in v))
        assert n < 1e-6 or abs(n - 1.0) < 1e-6
    assert any(math.sqrt(sum(x * x for x in v)) > 0.5 for v in out)  # some non-trivial


def test_bge_query_prefix_applied(monkeypatch):
    """RETRIEVAL_QUERY with a bge model should prepend the BGE instruction prefix."""
    captured: list[str] = []

    class _PrefixModel:
        def encode(self, texts, **kw):
            captured.extend(texts)
            return np.array([[1.0, 0.0] for _ in texts])

    monkeypatch.setattr(embed, "get_model", lambda s: _PrefixModel())

    embed.embed_texts(_Settings(), ["hello world"], task_type="RETRIEVAL_QUERY")

    assert len(captured) == 1
    assert captured[0].startswith("Represent this sentence for searching relevant passages: ")
    assert "hello world" in captured[0]


def test_document_task_type_no_prefix(monkeypatch):
    """RETRIEVAL_DOCUMENT should NOT prepend any prefix."""
    captured: list[str] = []

    class _PrefixModel:
        def encode(self, texts, **kw):
            captured.extend(texts)
            return np.array([[1.0, 0.0] for _ in texts])

    monkeypatch.setattr(embed, "get_model", lambda s: _PrefixModel())

    embed.embed_texts(_Settings(), ["hello world"], task_type="RETRIEVAL_DOCUMENT")

    assert captured[0] == "hello world"  # exactly as-is, no prefix


def test_non_bge_query_no_prefix(monkeypatch):
    """Non-BGE models (e.g. all-MiniLM) should NOT get the BGE prefix even for queries."""
    captured: list[str] = []

    class _PrefixModel:
        def encode(self, texts, **kw):
            captured.extend(texts)
            return np.array([[1.0, 0.0] for _ in texts])

    class _MiniLMSettings:
        local_embed_model = "all-MiniLM-L6-v2"
        embed_delay = 0.0

    monkeypatch.setattr(embed, "get_model", lambda s: _PrefixModel())

    embed.embed_texts(_MiniLMSettings(), ["hello"], task_type="RETRIEVAL_QUERY")

    assert captured[0] == "hello"  # no prefix for non-bge models


def test_delay_param_accepted_without_error(monkeypatch):
    """delay kwarg is a no-op but must not raise."""
    monkeypatch.setattr(embed, "get_model", lambda s: _FakeModel())
    out = embed.embed_texts(_Settings(), ["a"], delay=5.0)
    assert len(out) == 1


def test_client_param_accepted_without_error(monkeypatch):
    """client kwarg is a no-op but must not raise."""
    monkeypatch.setattr(embed, "get_model", lambda s: _FakeModel())
    out = embed.embed_texts(_Settings(), ["a"], client=object())
    assert len(out) == 1


def test_model_cached_after_first_load(monkeypatch):
    """get_model should only instantiate SentenceTransformer once per model name."""
    calls = []

    class _FakeST:
        def __init__(self, name):
            calls.append(name)

        def encode(self, texts, **kw):
            return np.array([[1.0] for _ in texts])

    import sentence_transformers as _st_mod
    monkeypatch.setattr(_st_mod, "SentenceTransformer", _FakeST)

    s = _Settings()
    embed.get_model(s)
    embed.get_model(s)  # second call — must not re-instantiate

    assert len(calls) == 1
