"""Fetch + parse a single Drive file into clean text plus location anchors.

Each supported format becomes a :class:`ParsedDoc`: plain text plus *format-adaptive*
anchors — heading breadcrumbs (:class:`Section`) for reflowable docs, page spans
(:class:`PageSpan`) for PDFs. Later stages chunk this text (prepending heading paths)
and cite it (``page N`` for PDF, heading-path otherwise).

`drive.py` stays pure HTTP (bytes in/out); this module owns classification, the pure
byte→text parsers, and the router that wires them to `drive`. The pure parsers take
bytes/str already in hand, so they unit-test offline with no network.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass

from requests.exceptions import HTTPError

from . import drive

try:  # optional at import time so `parsers` imports even without the wheel
    import pdfplumber
except Exception:  # pragma: no cover - exercised only when the dep is absent
    pdfplumber = None


# --- data model -------------------------------------------------------------

@dataclass(frozen=True)
class Section:
    """A heading anchor within reflowable text."""

    offset: int  # char offset in ParsedDoc.text where the heading line starts
    path: str  # breadcrumb, e.g. "Processes > Scheduling"


@dataclass(frozen=True)
class PageSpan:
    """A PDF page's character span within ParsedDoc.text."""

    page: int  # 1-based page number
    start: int
    end: int


@dataclass(frozen=True)
class ParsedDoc:
    file_id: str
    name: str
    mime_type: str
    fmt: str  # "gdoc" | "pdf" | "docx" | "markdown" | "text"
    text: str
    sections: list[Section]  # populated for gdoc/markdown/docx; [] for pdf/text
    pages: list[PageSpan]  # populated for pdf; [] otherwise


class UnsupportedFormat(Exception):
    """Raised by the router for a MIME type outside the allowlist."""

    def __init__(self, mime_type: str) -> None:
        self.mime_type = mime_type
        super().__init__(f"Unsupported file type: {mime_type or 'unknown'} (skipped)")


# --- classification ---------------------------------------------------------

_FMT_BY_MIME = {
    "application/vnd.google-apps.document": "gdoc",
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "text/markdown": "markdown",
    "text/x-markdown": "markdown",
    "text/plain": "text",
}

_GDOC_EXPORT_MIME = "text/markdown"
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def classify(mime_type: str | None) -> str | None:
    """Map a MIME type to a parser format, or None if it's unsupported."""
    base = (mime_type or "").split(";")[0].strip()  # drop "; charset=utf-8"
    return _FMT_BY_MIME.get(base)


# --- pure parsers -----------------------------------------------------------

def _make(meta: dict, fmt: str, text: str, *, sections=None, pages=None) -> ParsedDoc:
    return ParsedDoc(
        file_id=meta.get("id", ""),
        name=meta.get("name", ""),
        mime_type=meta.get("mimeType", ""),
        fmt=fmt,
        text=text,
        sections=sections or [],
        pages=pages or [],
    )


def _breadcrumb(stack: list[tuple[int, str]], level: int, title: str) -> str:
    """Update the heading level-stack and return the "A > B > C" breadcrumb."""
    while stack and stack[-1][0] >= level:
        stack.pop()
    stack.append((level, title))
    return " > ".join(t for _, t in stack)


def parse_markdown(text: str, *, meta: dict, fmt: str = "markdown") -> ParsedDoc:
    """Parse markdown (also Google Docs exported as markdown): ATX headings → sections."""
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        m = _HEADING_RE.match(line)
        if m:
            title = m.group(2).strip().rstrip("#").strip()  # tolerate closing '#'s
            if title:
                sections.append(Section(offset=offset, path=_breadcrumb(stack, len(m.group(1)), title)))
        offset += len(line)
    return _make(meta, fmt, text, sections=sections)


def parse_text(text: str, *, meta: dict, fmt: str = "text") -> ParsedDoc:
    """Plain-text passthrough: no heading anchors."""
    return _make(meta, fmt, text)


def parse_docx(data: bytes, *, meta: dict) -> ParsedDoc:
    """Parse a .docx: paragraphs joined by newlines; Heading-styled ones → sections.

    MVP: paragraphs only. Tables / text-boxes are deferred to v2.
    """
    from docx import Document

    document = Document(io.BytesIO(data))
    parts: list[str] = []
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    offset = 0
    for para in document.paragraphs:
        txt = para.text
        style = getattr(getattr(para, "style", None), "name", "") or ""
        m = re.match(r"Heading (\d+)", style)
        if txt.strip() and (m or style == "Title"):
            level = int(m.group(1)) if m else 0  # Title outranks all Headings
            sections.append(Section(offset=offset, path=_breadcrumb(stack, level, txt.strip())))
        parts.append(txt)
        offset += len(txt) + 1  # +1 for the "\n" join
    return _make(meta, "docx", "\n".join(parts), sections=sections)


def parse_pdf(data: bytes, *, meta: dict) -> ParsedDoc:
    """Parse a PDF with pdfplumber: per-page text + page spans (no heading anchors).

    Scanned/image-only PDFs extract no text → empty ``text`` (callers must not index
    empty text; the ``fetch`` command still reports ``chars: 0``).
    """
    if pdfplumber is None:  # pragma: no cover - only when the dep is missing
        raise RuntimeError("pdfplumber is not installed; run `pip install -e '.[dev]'`.")
    parts: list[str] = []
    pages: list[PageSpan] = []
    offset = 0
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            ptext = page.extract_text() or ""
            start = offset
            parts.append(ptext)
            pages.append(PageSpan(page=page.page_number, start=start, end=start + len(ptext)))
            offset += len(ptext) + 2  # +2 for the "\n\n" join
    return _make(meta, "pdf", "\n\n".join(parts), pages=pages)


# --- router + locator -------------------------------------------------------

def fetch_document(session, meta: dict) -> ParsedDoc:
    """Fetch a file's content and parse it, routing on MIME type.

    Native Google Docs are exported to markdown (falling back to plain text if the
    markdown export is unavailable); binaries are downloaded via ``alt=media``.
    Raises :class:`UnsupportedFormat` for anything outside the allowlist.
    """
    mime = meta.get("mimeType", "")
    fmt = classify(mime)
    if fmt is None:
        raise UnsupportedFormat(mime)

    file_id = meta["id"]
    if fmt == "gdoc":
        try:
            return parse_markdown(drive.export_file(session, file_id, _GDOC_EXPORT_MIME), meta=meta, fmt="gdoc")
        except HTTPError:  # some Docs can't export markdown → plain text
            return parse_text(drive.export_file(session, file_id, "text/plain"), meta=meta, fmt="text")

    data = drive.download_file(session, file_id)
    if fmt == "pdf":
        return parse_pdf(data, meta=meta)
    if fmt == "docx":
        return parse_docx(data, meta=meta)
    text = data.decode("utf-8", errors="replace")
    if fmt == "markdown":
        return parse_markdown(text, meta=meta, fmt="markdown")
    return parse_text(text, meta=meta, fmt="text")


def locator_at(doc: ParsedDoc, offset: int) -> dict:
    """Format-adaptive citation locator for a char offset in ``doc.text``.

    Page for PDFs, else the nearest preceding heading breadcrumb, else the file name.
    Never emits "page" for a non-paged format.
    """
    if doc.pages:
        page = doc.pages[0].page
        for span in doc.pages:
            if span.start <= offset:
                page = span.page
            else:
                break
        return {"type": "page", "value": page}
    if doc.sections:
        path = None
        for s in doc.sections:
            if s.offset <= offset:
                path = s.path
            else:
                break
        if path is not None:
            return {"type": "heading", "value": path}
    return {"type": "heading", "value": doc.name}
