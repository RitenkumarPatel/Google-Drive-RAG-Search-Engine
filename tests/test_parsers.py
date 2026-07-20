"""Tests for gdrive_rag.parsers — all parsing is exercised offline (no network)."""

import io

import pytest

from gdrive_rag import drive, parsers

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


# --- classification ---------------------------------------------------------

def test_classify_supported():
    assert parsers.classify("application/vnd.google-apps.document") == "gdoc"
    assert parsers.classify("application/pdf") == "pdf"
    assert parsers.classify(DOCX_MIME) == "docx"
    assert parsers.classify("text/markdown") == "markdown"
    assert parsers.classify("text/x-markdown") == "markdown"
    assert parsers.classify("text/plain; charset=utf-8") == "text"


def test_classify_unsupported():
    for mime in (
        "application/vnd.google-apps.spreadsheet",
        "application/vnd.google-apps.presentation",
        "application/vnd.google-apps.folder",
        "application/vnd.google-apps.shortcut",
        "image/png",
        "",
    ):
        assert parsers.classify(mime) is None


# --- markdown ---------------------------------------------------------------

def test_parse_markdown_breadcrumbs():
    text = "# A\nintro\n## B\nbody b\n## C\nbody c\n### D\ndeep\n"
    doc = parsers.parse_markdown(
        text, meta={"id": "x", "name": "n", "mimeType": "text/markdown"}, fmt="markdown"
    )
    assert [s.path for s in doc.sections] == ["A", "A > B", "A > C", "A > C > D"]
    for s in doc.sections:  # each offset points at the heading's leading '#'
        assert text[s.offset] == "#"
    assert doc.text == text  # body preserved verbatim
    assert doc.pages == []


# --- plain text -------------------------------------------------------------

def test_parse_text_no_sections():
    doc = parsers.parse_text(
        "just some text\nmore", meta={"id": "x", "name": "n", "mimeType": "text/plain"}
    )
    assert doc.fmt == "text"
    assert doc.sections == []
    assert doc.pages == []


# --- docx (real round-trip via python-docx's writer) ------------------------

def test_parse_docx_roundtrip():
    docx = pytest.importorskip("docx")
    document = docx.Document()
    document.add_heading("Processes", level=1)
    document.add_paragraph("A process is a program in execution.")
    document.add_heading("Scheduling", level=2)
    document.add_paragraph("The scheduler picks the next process.")
    buf = io.BytesIO()
    document.save(buf)

    doc = parsers.parse_docx(buf.getvalue(), meta={"id": "x", "name": "OS", "mimeType": DOCX_MIME})
    assert doc.fmt == "docx"
    assert [s.path for s in doc.sections] == ["Processes", "Processes > Scheduling"]
    assert "program in execution" in doc.text
    assert "scheduler picks" in doc.text
    for s, expected in zip(doc.sections, ("Processes", "Scheduling")):
        assert doc.text[s.offset:s.offset + len(expected)] == expected


# --- pdf (assembly logic, pdfplumber faked) ---------------------------------

class _FakePage:
    def __init__(self, page_number, text):
        self.page_number = page_number
        self._text = text

    def extract_text(self):
        return self._text


class _FakePDF:
    def __init__(self, pages):
        self.pages = pages

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakePdfplumber:
    def __init__(self, pages):
        self._pages = pages

    def open(self, _fileobj):
        return _FakePDF(self._pages)


def test_parse_pdf_page_spans(monkeypatch):
    pages = [_FakePage(1, "page one text"), _FakePage(2, "second page")]
    monkeypatch.setattr(parsers, "pdfplumber", _FakePdfplumber(pages))

    doc = parsers.parse_pdf(b"ignored", meta={"id": "x", "name": "n", "mimeType": "application/pdf"})
    assert doc.fmt == "pdf"
    assert [p.page for p in doc.pages] == [1, 2]
    for span, expected in zip(doc.pages, ("page one text", "second page")):
        assert doc.text[span.start:span.end] == expected  # spans slice back to page text
    assert doc.sections == []


# --- router (drive fetch calls faked) ---------------------------------------

def test_fetch_document_routes_gdoc_to_export(monkeypatch):
    calls = {}

    def fake_export(session, file_id, mime):
        calls["export"] = (file_id, mime)
        return "# Title\nbody"

    def fake_download(session, file_id):
        calls["download"] = file_id
        return b""

    monkeypatch.setattr(drive, "export_file", fake_export)
    monkeypatch.setattr(drive, "download_file", fake_download)

    meta = {"id": "g1", "name": "Doc", "mimeType": "application/vnd.google-apps.document"}
    doc = parsers.fetch_document(object(), meta)

    assert calls["export"] == ("g1", "text/markdown")
    assert "download" not in calls
    assert doc.fmt == "gdoc"
    assert [s.path for s in doc.sections] == ["Title"]


def test_fetch_document_routes_pdf_to_download(monkeypatch):
    calls = {}

    def fake_download(session, file_id):
        calls["download"] = file_id
        return b"%PDF"

    monkeypatch.setattr(drive, "download_file", fake_download)
    monkeypatch.setattr(
        parsers,
        "parse_pdf",
        lambda data, *, meta: parsers.ParsedDoc(
            meta["id"], meta["name"], meta["mimeType"], "pdf", "x", [], []
        ),
    )

    meta = {"id": "p1", "name": "PDF", "mimeType": "application/pdf"}
    doc = parsers.fetch_document(object(), meta)

    assert calls["download"] == "p1"
    assert doc.fmt == "pdf"


def test_fetch_document_routes_text_to_download(monkeypatch):
    monkeypatch.setattr(drive, "download_file", lambda s, fid: b"hello world")
    meta = {"id": "t1", "name": "notes.txt", "mimeType": "text/plain"}
    doc = parsers.fetch_document(object(), meta)
    assert doc.fmt == "text"
    assert doc.text == "hello world"


def test_fetch_document_unsupported_raises():
    meta = {"id": "s1", "name": "Sheet", "mimeType": "application/vnd.google-apps.spreadsheet"}
    with pytest.raises(parsers.UnsupportedFormat) as ei:
        parsers.fetch_document(object(), meta)
    assert ei.value.mime_type == "application/vnd.google-apps.spreadsheet"


# --- locator_at -------------------------------------------------------------

def test_locator_at_pdf():
    doc = parsers.ParsedDoc(
        "x", "n", "application/pdf", "pdf", "abcdefgh",
        sections=[], pages=[parsers.PageSpan(1, 0, 4), parsers.PageSpan(2, 4, 8)],
    )
    assert parsers.locator_at(doc, 0) == {"type": "page", "value": 1}
    assert parsers.locator_at(doc, 5) == {"type": "page", "value": 2}
    assert parsers.locator_at(doc, 100) == {"type": "page", "value": 2}


def test_locator_at_heading():
    doc = parsers.ParsedDoc(
        "x", "Doc", "text/markdown", "markdown", "." * 30,
        sections=[parsers.Section(0, "A"), parsers.Section(10, "A > B")], pages=[],
    )
    assert parsers.locator_at(doc, 5) == {"type": "heading", "value": "A"}
    assert parsers.locator_at(doc, 20) == {"type": "heading", "value": "A > B"}


def test_locator_at_text_fallback():
    doc = parsers.ParsedDoc("x", "notes.txt", "text/plain", "text", "hi", sections=[], pages=[])
    assert parsers.locator_at(doc, 0) == {"type": "heading", "value": "notes.txt"}
