"""Embed chunk / query text with a local sentence-transformers model.

The model (``BAAI/bge-base-en-v1.5`` by default) is loaded once on first call and
cached in-process as a module-level singleton. No API calls, no rate limits, and no
network required after the one-time HuggingFace model download.

BGE models use asymmetric retrieval prompts:
  - ``RETRIEVAL_QUERY``    → prepend the BGE query instruction prefix
  - ``RETRIEVAL_DOCUMENT`` → no prefix (raw chunk text)

Vectors are L2-normalized so that cosine similarity equals dot product, consistent
with ChromaDB's ``hnsw:space: "cosine"`` index.

The ``client`` and ``delay`` parameters are accepted but silently ignored; they are
preserved so existing call sites (``cli.py``, ``retrieve.py``) need no changes.
"""

from __future__ import annotations

import math

_BATCH = 64
_MODEL_CACHE: dict = {}

# BGE's documented query instruction for asymmetric retrieval
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _is_bge(model_name: str) -> bool:
    return "bge" in model_name.lower()


def get_model(settings):
    """Load (or return cached) SentenceTransformer for the configured model name."""
    from sentence_transformers import SentenceTransformer

    name = settings.local_embed_model
    if name not in _MODEL_CACHE:
        _MODEL_CACHE[name] = SentenceTransformer(name)
    return _MODEL_CACHE[name]


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if not norm:
        return list(vec)
    return [x / norm for x in vec]


def embed_texts(
    settings,
    texts,
    *,
    task_type: str = "RETRIEVAL_DOCUMENT",
    client=None,          # ignored — kept for call-site compatibility
    batch_size: int = _BATCH,
    delay: float | None = None,  # ignored — kept for call-site compatibility
) -> list[list[float]]:
    """Return one L2-normalized embedding per input text (order preserved)."""
    if not texts:
        return []

    model = get_model(settings)
    is_query = task_type == "RETRIEVAL_QUERY"

    # BGE asymmetric prefix: prepend only for queries, not for document chunks
    if is_query and _is_bge(settings.local_embed_model):
        texts = [_BGE_QUERY_PREFIX + t for t in texts]

    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        raw = model.encode(batch, normalize_embeddings=False, show_progress_bar=False)
        vectors.extend(_l2_normalize(list(map(float, v))) for v in raw)
    return vectors
