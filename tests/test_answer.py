"""Tests for gdrive_rag.answer — synthesis and citation prompt formatting (Gemini mocked)."""

import pytest

from gdrive_rag import answer
from gdrive_rag.retrieve import SearchHit


class _FakeResp:
    def __init__(self, text):
        self.text = text


class _FakeChat:
    def __init__(self, reply):
        self.reply = reply
        self.messages = []

    def send_message(self, message):
        self.messages.append(message)
        return _FakeResp(self.reply)


class _FakeChats:
    def __init__(self, reply):
        self.reply = reply
        self.created = []

    def create(self, *, model):
        chat = _FakeChat(self.reply)
        self.created.append({"model": model, "chat": chat})
        return chat


class _FakeModels:
    def __init__(self, reply="A process is a program in execution [1]."):
        self.reply = reply
        self.calls = []

    def generate_content(self, *, model, contents):
        self.calls.append({"model": model, "contents": contents})
        return _FakeResp(self.reply)


class _FakeClientWithChats:
    def __init__(self, reply="A process is a program in execution [1]."):
        self.chats = _FakeChats(reply)


class _FakeClientWithModels:
    def __init__(self, reply="A process is a program in execution [1]."):
        self.models = _FakeModels(reply)


class _Settings:
    chat_model = "gemini-flash-latest"
    embed_model = "gemini-embedding-001"
    embed_dims = 768
    gemini_api_key = "test-key"


def test_format_locator():
    assert answer._format_locator({"type": "page", "value": "4"}) == "p. 4"
    assert answer._format_locator({"type": "heading", "value": "Processes > Scheduling"}) == "Processes > Scheduling"
    assert answer._format_locator({}) == ""


def test_format_context():
    hits = [
        SearchHit(
            score=0.9,
            name="OS Notes",
            locator={"type": "heading", "value": "Processes"},
            drive_url="https://drive.google.com/open?id=123",
            text="OS Notes — Processes\n\nA process is a program in execution.",
            chunk_index=0,
        ),
        SearchHit(
            score=0.85,
            name="Textbook.pdf",
            locator={"type": "page", "value": "12"},
            drive_url="https://drive.google.com/open?id=456",
            text="Textbook.pdf — p.12\n\nContext switching involves saving state.",
            chunk_index=2,
        ),
    ]

    context_str, citations = answer.format_context(hits)

    assert "[1] Document: OS Notes (Processes)" in context_str
    assert "A process is a program in execution." in context_str
    assert "[2] Document: Textbook.pdf (p. 12)" in context_str
    assert "Context switching involves saving state." in context_str

    assert len(citations) == 2
    assert citations[0].index == 1
    assert citations[0].name == "OS Notes"
    assert citations[0].locator == {"type": "heading", "value": "Processes"}
    assert citations[0].drive_url == "https://drive.google.com/open?id=123"

    assert citations[1].index == 2
    assert citations[1].name == "Textbook.pdf"
    assert citations[1].locator == {"type": "page", "value": "12"}
    assert citations[1].drive_url == "https://drive.google.com/open?id=456"


def test_build_prompt():
    prompt = answer.build_prompt("What is a process?", "[1] Document: OS\nContent")
    assert "What is a process?" in prompt
    assert "[1] Document: OS" in prompt
    assert "inline citation" in prompt
    assert "Instructions:" in prompt


def test_generate_answer_empty_hits():
    ans = answer.generate_answer(_Settings(), "What is X?", [])
    assert "No relevant documents found" in ans.text
    assert ans.citations == []
    assert ans.hits == []


def test_generate_answer_with_chats_client():
    client = _FakeClientWithChats("Processes run code [1].")
    hits = [
        SearchHit(
            score=0.9,
            name="OS",
            locator={"type": "heading", "value": "Intro"},
            drive_url="https://drive.google.com/open?id=1",
            text="OS — Intro\nProcesses run code.",
            chunk_index=0,
        )
    ]
    ans = answer.generate_answer(_Settings(), "What runs code?", hits, client=client)

    assert ans.text == "Processes run code [1]."
    assert len(ans.citations) == 1
    assert ans.citations[0].name == "OS"
    assert client.chats.created[0]["model"] == "gemini-flash-latest"
    assert "What runs code?" in client.chats.created[0]["chat"].messages[0]


def test_generate_answer_with_models_fallback():
    client = _FakeClientWithModels("Fallback reply [1].")
    hits = [
        SearchHit(
            score=0.9,
            name="OS",
            locator={"type": "heading", "value": "Intro"},
            drive_url="https://drive.google.com/open?id=1",
            text="OS — Intro\nProcesses run code.",
            chunk_index=0,
        )
    ]
    ans = answer.generate_answer(_Settings(), "query", hits, client=client)
    assert ans.text == "Fallback reply [1]."
    assert client.models.calls[0]["model"] == "gemini-flash-latest"


def test_generate_answer_api_error_raises():
    from google.genai import errors

    class _ErrModels:
        def generate_content(self, **kwargs):
            raise errors.APIError(500, {"error": {"code": 500, "message": "server error"}})

    class _ErrClient:
        models = _ErrModels()

    hits = [
        SearchHit(
            score=0.9,
            name="OS",
            locator={"type": "heading", "value": "Intro"},
            drive_url="https://drive.google.com/open?id=1",
            text="text",
            chunk_index=0,
        )
    ]

    with pytest.raises(RuntimeError) as exc_info:
        answer.generate_answer(_Settings(), "query", hits, client=_ErrClient())
    assert "Gemini generation failed" in str(exc_info.value)


def test_answer_query_integration(monkeypatch):
    class _FakeStore:
        pass

    hit = SearchHit(
        score=0.95,
        name="Guide",
        locator={"type": "page", "value": "1"},
        drive_url="https://drive/1",
        text="Guide — p.1\nContent",
        chunk_index=0,
    )

    monkeypatch.setattr(answer.retrieve, "search", lambda settings, store, query, k=6, client=None: [hit])
    client = _FakeClientWithChats("Synthesized answer [1].")

    ans = answer.answer_query(_Settings(), _FakeStore(), "How to guide?", client=client)

    assert ans.text == "Synthesized answer [1]."
    assert len(ans.citations) == 1
    assert ans.citations[0].name == "Guide"
