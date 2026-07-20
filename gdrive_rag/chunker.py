"""Split a ParsedDoc into overlapping, citation-tagged chunks for embedding.

A dependency-free recursive character splitter (no LangChain, no tokenizer): it prefers
to break on paragraph → line → sentence → word boundaries and only hard-wraps as a last
resort. Char offsets are preserved back into ``ParsedDoc.text`` so each chunk carries a
format-adaptive locator (page for PDF, heading breadcrumb otherwise) for citations, and
each chunk's text is prefixed with the doc name + locator hint to give retrieval context.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import parsers
from .parsers import ParsedDoc

_SIZE = 2000  # ~500 tokens; stays well under the 2048-token embedding input limit
_OVERLAP = 240
_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@dataclass(frozen=True)
class Chunk:
    file_id: str
    name: str
    mime_type: str
    fmt: str
    index: int
    text: str  # heading/title-prefixed body, ready to embed + store
    locator: dict  # {"type": "page"|"heading", "value": ...}


def _split_keep(text: str, sep: str) -> list[str]:
    """Split on ``sep`` but re-attach it so the pieces concatenate back to ``text``."""
    parts = text.split(sep)
    out = []
    for i, p in enumerate(parts):
        if i < len(parts) - 1:
            out.append(p + sep)
        elif p:
            out.append(p)
    return out


def _split_with_offsets(text: str, start: int, separators: list[str], size: int):
    """Yield ``(offset, piece)`` where each piece is <= ``size`` where possible.

    ``offset`` indexes into the original document text.
    """
    if len(text) <= size:
        if text:
            yield (start, text)
        return
    if not separators or separators[0] == "":
        for i in range(0, len(text), size):  # hard wrap, last resort
            yield (start + i, text[i:i + size])
        return
    offset = start
    for piece in _split_keep(text, separators[0]):
        if len(piece) <= size:
            if piece:
                yield (offset, piece)
        else:
            yield from _split_with_offsets(piece, offset, separators[1:], size)
        offset += len(piece)


def _merge(pieces, size: int, overlap: int) -> list[tuple[int, str]]:
    """Pack ``(offset, piece)`` pieces into <=size chunks with an ``overlap``-char tail.

    Returns ``[(content_start, text)]``; ``content_start`` is the offset where the chunk's
    text begins (inside the previous chunk when an overlap tail is carried over).
    """
    chunks: list[tuple[int, str]] = []
    cur_text = ""
    cur_start = 0
    for off, piece in pieces:
        if cur_text and len(cur_text) + len(piece) > size:
            chunks.append((cur_start, cur_text))
            tail = cur_text[-overlap:] if overlap else ""
            cur_text = tail + piece
            cur_start = off - len(tail)
        else:
            if not cur_text:
                cur_start = off
            cur_text += piece
    if cur_text:
        chunks.append((cur_start, cur_text))
    return chunks


def _prefix(doc: ParsedDoc, locator: dict) -> str:
    name = doc.name or ""
    if locator["type"] == "page":
        return f"{name} — p.{locator['value']}".strip(" —")
    value = locator.get("value") or ""
    if value and value != name:
        return f"{name} — {value}".strip(" —")
    return name


def chunk_document(doc: ParsedDoc, *, size: int = _SIZE, overlap: int = _OVERLAP) -> list[Chunk]:
    """Split ``doc`` into overlapping, locator-tagged, context-prefixed chunks."""
    merged = _merge(_split_with_offsets(doc.text, 0, _SEPARATORS, size), size, overlap)
    chunks: list[Chunk] = []
    index = 0
    for start, raw in merged:
        body = raw.strip()
        if not body:
            continue
        locator = parsers.locator_at(doc, start)
        prefix = _prefix(doc, locator)
        text = f"{prefix}\n\n{body}" if prefix else body
        chunks.append(Chunk(doc.file_id, doc.name, doc.mime_type, doc.fmt, index, text, locator))
        index += 1
    return chunks
