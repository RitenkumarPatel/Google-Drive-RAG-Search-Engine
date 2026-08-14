"""Grounded synthesis: generate answers with citations from retrieved Drive chunks.

Takes top-k SearchHits from retrieve.py, formats a numbered context block, prompts
Gemini (via settings.chat_model) to answer strictly from the context with [1], [2]
inline citations, and pairs the response with clickable Drive source URLs.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import retrieve
from .retrieve import SearchHit


@dataclass(frozen=True)
class Citation:
    index: int
    name: str
    locator: dict  # {"type": "page"|"heading", "value": ...}
    drive_url: str


@dataclass(frozen=True)
class Answer:
    query: str
    text: str
    citations: list[Citation]
    hits: list[SearchHit]


def _format_locator(locator: dict) -> str:
    value = locator.get("value")
    if not value:
        return ""
    return f"p. {value}" if locator.get("type") == "page" else str(value)


def format_context(hits: list[SearchHit]) -> tuple[str, list[Citation]]:
    """Convert search hits into a numbered context block and a Citation list."""
    context_parts: list[str] = []
    citations: list[Citation] = []
    for i, h in enumerate(hits, 1):
        loc_str = _format_locator(h.locator)
        header = f"[{i}] Document: {h.name}"
        if loc_str and loc_str != h.name:
            header += f" ({loc_str})"
        context_parts.append(f"{header}\n{h.text}")
        citations.append(
            Citation(
                index=i,
                name=h.name,
                locator=h.locator,
                drive_url=h.drive_url,
            )
        )
    return "\n\n".join(context_parts), citations


_SYSTEM_PROMPT = """You are a helpful and precise assistant answering questions based on the user's personal Google Drive documents.

Instructions:
1. Answer the question directly and concisely, relying strictly on the provided Context.
2. For every factual claim or detail you state, include an inline citation referencing the corresponding source passage number, e.g. [1], [2], or [1][3].
3. Do not make assumptions or extrapolate beyond what is stated in the Context.
4. If the provided Context does not contain the answer or enough information to answer the question, clearly state: "I could not find information to answer this question in your indexed Google Drive documents."
"""


def build_prompt(query: str, context_str: str) -> str:
    """Combine system prompt, context passages, and user query."""
    return f"""{_SYSTEM_PROMPT}

---
Context:
{context_str}
---

Question: {query}

Answer:"""


def generate_answer(
    settings,
    query: str,
    hits: list[SearchHit],
    *,
    client=None,
) -> Answer:
    """Synthesize a grounded answer with citations for ``query`` given ``hits``."""
    if not hits:
        return Answer(
            query=query,
            text="No relevant documents found in your indexed Google Drive.",
            citations=[],
            hits=[],
        )

    context_str, citations = format_context(hits)
    prompt = build_prompt(query, context_str)

    from google import genai
    from google.genai import errors as genai_errors

    client = client or genai.Client(api_key=settings.gemini_api_key)

    try:
        if hasattr(client, "chats") and hasattr(client.chats, "create"):
            chat = client.chats.create(model=settings.chat_model)
            resp = chat.send_message(prompt)
        else:
            resp = client.models.generate_content(
                model=settings.chat_model,
                contents=prompt,
            )
    except genai_errors.APIError as e:
        raise RuntimeError(f"Gemini generation failed ({type(e).__name__}): {e}") from e

    answer_text = (resp.text or "").strip()
    return Answer(
        query=query,
        text=answer_text,
        citations=citations,
        hits=hits,
    )


def answer_query(
    settings,
    store,
    query: str,
    *,
    k: int = 6,
    client=None,
) -> Answer:
    """Retrieve top-k chunks and generate a grounded citation-backed answer."""
    hits = retrieve.search(settings, store, query, k=k, client=client)
    return generate_answer(settings, query, hits, client=client)
