"""Tests for gdrive_rag.embed — the Gemini client is faked (no network)."""

import math

from gdrive_rag import embed


class _Emb:
    def __init__(self, values):
        self.values = values


class _Resp:
    def __init__(self, rows):
        self.embeddings = [_Emb(r) for r in rows]


class _FakeModels:
    def __init__(self):
        self.calls = []

    def embed_content(self, *, model, contents, config):
        self.calls.append(
            {"model": model, "n": len(contents),
             "task_type": config.task_type, "dims": config.output_dimensionality}
        )
        return _Resp([[float(i), 0.0, 0.0] for i in range(len(contents))])


class _FakeClient:
    def __init__(self):
        self.models = _FakeModels()


class _Settings:
    embed_model = "gemini-embedding-001"
    embed_dims = 768
    gemini_api_key = "test"


def test_l2_normalize_unit():
    assert embed._l2_normalize([3.0, 4.0]) == [0.6, 0.8]


def test_l2_normalize_zero_vector():
    assert embed._l2_normalize([0.0, 0.0]) == [0.0, 0.0]


def test_embed_texts_empty():
    assert embed.embed_texts(_Settings(), []) == []


def test_embed_texts_batches_and_normalizes():
    client = _FakeClient()
    out = embed.embed_texts(_Settings(), [f"t{i}" for i in range(150)], client=client, batch_size=100)

    assert len(out) == 150
    assert [c["n"] for c in client.models.calls] == [100, 50]  # two batches
    assert client.models.calls[0]["task_type"] == "RETRIEVAL_DOCUMENT"
    assert client.models.calls[0]["dims"] == 768
    for v in out:  # every vector is unit-norm or the zero vector
        n = math.sqrt(sum(x * x for x in v))
        assert n < 1e-6 or abs(n - 1.0) < 1e-6
    assert any(math.sqrt(sum(x * x for x in v)) > 0.5 for v in out)  # some are non-trivial


def test_embed_retries_on_rate_limit(monkeypatch):
    from google.genai import errors

    monkeypatch.setattr(embed.time, "sleep", lambda _s: None)

    class _FlakyModels:
        def __init__(self):
            self.n = 0

        def embed_content(self, *, model, contents, config):
            self.n += 1
            if self.n == 1:
                raise errors.APIError(
                    429, {"error": {"code": 429, "message": "rate", "status": "RESOURCE_EXHAUSTED"}}
                )
            return _Resp([[1.0, 0.0] for _ in contents])

    class _FlakyClient:
        def __init__(self):
            self.models = _FlakyModels()

    client = _FlakyClient()
    out = embed.embed_texts(_Settings(), ["a", "b"], client=client)

    assert client.models.n == 2  # failed once, retried, succeeded
    assert len(out) == 2
