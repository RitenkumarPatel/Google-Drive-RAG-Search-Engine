"""Tests for gdrive_rag.retrieve — embedding + store are faked (no network, no Chroma)."""

from gdrive_rag import retrieve


class _Settings:
    embed_model = "gemini-embedding-001"
    embed_dims = 768
    gemini_api_key = "test"


class _FakeStore:
    def __init__(self, results):
        self._results = results
        self.queried = None

    def query(self, embedding, k=6):
        self.queried = (embedding, k)
        return self._results


def test_search_embeds_query_and_maps_hits(monkeypatch):
    captured = {}

    def fake_embed(settings, texts, *, task_type, client=None):
        captured["task_type"] = task_type
        captured["texts"] = texts
        return [[0.1, 0.2, 0.3]]

    monkeypatch.setattr(retrieve.embed, "embed_texts", fake_embed)
    store = _FakeStore([
        {
            "id": "1",
            "document": "Notes — Intro\n\nbody one",
            "score": 0.91,
            "metadata": {
                "name": "Notes", "loc_type": "heading", "loc_value": "Intro",
                "drive_url": "https://drive/1", "chunk_index": 0,
            },
        }
    ])

    hits = retrieve.search(_Settings(), store, "what is a process", k=3)

    assert captured["task_type"] == "RETRIEVAL_QUERY"  # asymmetric with document embeds
    assert captured["texts"] == ["what is a process"]
    assert store.queried == ([0.1, 0.2, 0.3], 3)
    assert len(hits) == 1
    h = hits[0]
    assert h.score == 0.91
    assert h.name == "Notes"
    assert h.locator == {"type": "heading", "value": "Intro"}
    assert h.drive_url == "https://drive/1"
    assert "body one" in h.text


def test_search_empty_embedding_returns_nothing(monkeypatch):
    monkeypatch.setattr(retrieve.embed, "embed_texts", lambda *a, **k: [])
    assert retrieve.search(_Settings(), _FakeStore([]), "q") == []
