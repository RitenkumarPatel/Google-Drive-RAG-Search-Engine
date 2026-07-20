"""Dense retrieval: embed a query and return the most similar stored chunks.

The query is embedded with ``task_type=RETRIEVAL_QUERY`` (asymmetric with the chunks'
``RETRIEVAL_DOCUMENT`` — better recall) and matched against the Chroma cosine index.
This is retrieval only: no LLM, no answer generation (that's Stage 5).
"""

from __future__ import annotations

from dataclasses import dataclass

from . import embed


@dataclass(frozen=True)
class SearchHit:
    score: float  # cosine similarity in [~0, 1]
    name: str
    locator: dict  # {"type": "page"|"heading", "value": ...}
    drive_url: str
    text: str  # stored chunk text (doc-name/locator prefix + body)
    chunk_index: int


def search(settings, store, query: str, *, k: int = 6, client=None) -> list[SearchHit]:
    """Return the top-``k`` chunks most similar to ``query`` (highest score first)."""
    vectors = embed.embed_texts(settings, [query], task_type="RETRIEVAL_QUERY", client=client)
    if not vectors:
        return []
    hits: list[SearchHit] = []
    for r in store.query(vectors[0], k=k):
        meta = r.get("metadata") or {}
        hits.append(
            SearchHit(
                score=r.get("score", 0.0),
                name=meta.get("name", ""),
                locator={"type": meta.get("loc_type", ""), "value": meta.get("loc_value", "")},
                drive_url=meta.get("drive_url", ""),
                text=r.get("document") or "",
                chunk_index=meta.get("chunk_index", 0),
            )
        )
    return hits
