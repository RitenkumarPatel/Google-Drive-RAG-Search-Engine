"""Tests for gdrive_rag.chunker — pure, offline."""

from gdrive_rag import parsers
from gdrive_rag.chunker import chunk_document


def _doc(text, *, sections=None, pages=None, name="Doc", fmt="markdown", mime="text/markdown"):
    return parsers.ParsedDoc("f1", name, mime, fmt, text, sections or [], pages or [])


def test_empty_doc_no_chunks():
    assert chunk_document(_doc("")) == []


def test_small_doc_single_chunk():
    doc = _doc("short body", sections=[parsers.Section(0, "Intro")])
    chunks = chunk_document(doc)
    assert len(chunks) == 1
    assert chunks[0].index == 0
    assert "short body" in chunks[0].text
    assert chunks[0].locator == {"type": "heading", "value": "Intro"}


def test_splits_into_contiguous_indices():
    paras = [f"Paragraph {i} " + "x" * 90 for i in range(12)]
    chunks = chunk_document(_doc("\n\n".join(paras)), size=200, overlap=50)
    assert len(chunks) > 1
    assert [c.index for c in chunks] == list(range(len(chunks)))
    for c in chunks:  # body (after the prefix) stays within size + overlap
        body = c.text.split("\n\n", 1)[-1]
        assert len(body) <= 200 + 50


def test_overlap_hard_wrap():
    text = "abcdefghij" * 30  # 300 chars, no separators → deterministic hard wrap
    chunks = chunk_document(_doc(text, name=""), size=100, overlap=20)  # name="" ⇒ no prefix
    assert len(chunks) >= 2
    assert chunks[0].text[-20:] == chunks[1].text[:20]  # tail of one == head of next


def test_heading_prefix():
    c = chunk_document(_doc("intro body", sections=[parsers.Section(0, "A > B")], name="Notes"))[0]
    assert c.text.startswith("Notes — A > B")
    assert c.locator == {"type": "heading", "value": "A > B"}


def test_pdf_page_prefix_and_locator():
    doc = _doc(
        "body text here",
        pages=[parsers.PageSpan(3, 0, 50)],
        name="Lecture",
        fmt="pdf",
        mime="application/pdf",
    )
    c = chunk_document(doc)[0]
    assert c.locator == {"type": "page", "value": 3}
    assert c.text.startswith("Lecture — p.3")
