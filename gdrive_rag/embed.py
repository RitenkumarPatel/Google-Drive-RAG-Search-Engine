"""Embed chunk / query text with the Gemini embeddings model.

``gemini-embedding-001`` with retrieval task types (RETRIEVAL_DOCUMENT for chunks,
RETRIEVAL_QUERY for queries — asymmetric retrieval improves recall). Vectors are
L2-normalized (required for output dims < 3072, so cosine == dot product), batched,
and retried with exponential backoff on rate-limit (HTTP 429) errors.
"""

from __future__ import annotations

import math
import time

_BATCH = 100
_MAX_RETRIES = 5
_BASE_DELAY = 2.0


def get_client(settings):
    from google import genai

    return genai.Client(api_key=settings.gemini_api_key)


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if not norm:
        return list(vec)
    return [x / norm for x in vec]


def _is_rate_limit(exc) -> bool:
    return getattr(exc, "code", None) == 429


def _embed_batch(client, settings, batch, task_type) -> list[list[float]]:
    from google.genai import errors, types

    delay = _BASE_DELAY
    for attempt in range(_MAX_RETRIES):
        try:
            resp = client.models.embed_content(
                model=settings.embed_model,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=settings.embed_dims,
                ),
            )
            return [list(e.values) for e in resp.embeddings]
        except errors.APIError as exc:
            if _is_rate_limit(exc) and attempt < _MAX_RETRIES - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise
    return []  # pragma: no cover - loop always returns or raises


def embed_texts(
    settings,
    texts,
    *,
    task_type: str = "RETRIEVAL_DOCUMENT",
    client=None,
    batch_size: int = _BATCH,
) -> list[list[float]]:
    """Return one L2-normalized embedding per input text (order preserved)."""
    if not texts:
        return []
    client = client or get_client(settings)
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        raw = _embed_batch(client, settings, texts[i:i + batch_size], task_type)
        vectors.extend(_l2_normalize(v) for v in raw)
    return vectors
