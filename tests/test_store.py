"""Tests for gdrive_rag.store — real Chroma in a tmp dir, fake vectors, no network."""

import pytest

pytest.importorskip("chromadb")

from gdrive_rag import store as store_mod
from gdrive_rag.chunker import Chunk


class _Settings:
    def __init__(self, data_dir):
        self.data_dir = data_dir


def _chunk(i):
    return Chunk("f1", "Doc", "text/markdown", "markdown", i, f"chunk {i}",
                 {"type": "heading", "value": "H"})


def _vec():
    return [0.1, 0.2, 0.3]


def _meta(*, fid="f1", name="Doc", mod="2026-01-01T00:00:00Z"):
    return {"id": fid, "name": name, "mimeType": "text/markdown", "modifiedTime": mod}


def test_add_and_stats(tmp_path):
    s = store_mod.Store(_Settings(tmp_path))
    s.replace_file(_meta(), [_chunk(0), _chunk(1)], [_vec(), _vec()])
    assert s.stats() == {"files": 1, "chunks": 2}
    assert s.get_indexed_version("f1") == "2026-01-01T00:00:00Z"
    assert s.get_indexed_version("nope") is None
    s.close()


def test_idempotent_same_version(tmp_path):
    s = store_mod.Store(_Settings(tmp_path))
    s.replace_file(_meta(), [_chunk(0), _chunk(1)], [_vec(), _vec()])
    s.replace_file(_meta(), [_chunk(0), _chunk(1)], [_vec(), _vec()])  # same cv → same ids
    assert s.stats()["chunks"] == 2
    s.close()


def test_shrink_purges_orphans(tmp_path):
    s = store_mod.Store(_Settings(tmp_path))
    s.replace_file(_meta(mod="v1"), [_chunk(0), _chunk(1), _chunk(2)], [_vec()] * 3)
    assert s.stats()["chunks"] == 3
    s.replace_file(_meta(mod="v2"), [_chunk(0)], [_vec()])  # new version, fewer chunks
    assert s.stats()["chunks"] == 1  # v1's orphan chunks are gone
    s.close()


def test_purge_and_reconcile(tmp_path):
    s = store_mod.Store(_Settings(tmp_path))
    s.replace_file(_meta(fid="A", name="A"), [_chunk(0)], [_vec()])
    s.replace_file(_meta(fid="B", name="B"), [_chunk(0)], [_vec()])
    assert s.stats()["files"] == 2

    removed = s.reconcile(["A"])  # B no longer live
    assert removed == ["B"]
    assert s.stats() == {"files": 1, "chunks": 1}

    s.purge_file("A")
    assert s.stats() == {"files": 0, "chunks": 0}
    s.close()
