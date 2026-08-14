"""Tests for the CLI. The Gemini client is mocked — no network calls."""

from click.testing import CliRunner

from gdrive_rag.cli import cli


class _FakeResp:
    text = "pong"


class _FakeModels:
    def generate_content(self, model, contents):
        return _FakeResp()


class _FakeClient:
    def __init__(self, api_key):
        assert api_key  # ensure the key is threaded through
        self.models = _FakeModels()


def test_ping_ok_with_mocked_client(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test")
    monkeypatch.setenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")

    import google.genai as genai_mod

    monkeypatch.setattr(genai_mod, "Client", _FakeClient)

    result = CliRunner().invoke(cli, ["ping"])

    assert result.exit_code == 0, result.output
    assert "pong" in result.output
    assert "gemini-2.5-flash" in result.output


def test_ping_missing_key_fails_cleanly(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    # Don't let a real .env repopulate the key during this test.
    monkeypatch.setattr("gdrive_rag.config.load_dotenv", lambda *a, **k: None)

    result = CliRunner().invoke(cli, ["ping"])

    assert result.exit_code != 0
    assert "GEMINI_API_KEY" in result.output


def test_config_reports_missing_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("gdrive_rag.config.load_dotenv", lambda *a, **k: None)

    result = CliRunner().invoke(cli, ["config"])

    assert result.exit_code == 0, result.output
    assert "MISSING" in result.output


def test_list_requires_login(monkeypatch):
    monkeypatch.setattr("gdrive_rag.config.load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr("gdrive_rag.auth.load_credentials", lambda settings: None)

    result = CliRunner().invoke(cli, ["list"])

    assert result.exit_code != 0
    assert "login" in result.output.lower()


def test_list_prints_files(monkeypatch):
    monkeypatch.setattr("gdrive_rag.config.load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr("gdrive_rag.auth.load_credentials", lambda settings: object())
    monkeypatch.setattr("gdrive_rag.drive.get_service", lambda creds: object())
    monkeypatch.setattr(
        "gdrive_rag.drive.list_files",
        lambda service, limit: [
            {
                "id": "1",
                "name": "CS 213 Notes",
                "mimeType": "application/vnd.google-apps.document",
                "modifiedTime": "2026-01-01T00:00:00Z",
            }
        ],
    )

    result = CliRunner().invoke(cli, ["list", "--limit", "5"])

    assert result.exit_code == 0, result.output
    assert "CS 213 Notes" in result.output


def test_list_reports_network_error(monkeypatch):
    monkeypatch.setattr("gdrive_rag.config.load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr("gdrive_rag.auth.load_credentials", lambda settings: object())
    monkeypatch.setattr("gdrive_rag.drive.get_service", lambda creds: object())

    def _boom(service, limit):
        raise RuntimeError("Tunnel connection failed")

    monkeypatch.setattr("gdrive_rag.drive.list_files", _boom)

    result = CliRunner().invoke(cli, ["list"])

    assert result.exit_code != 0
    assert "failed" in result.output.lower()


def test_fetch_requires_login(monkeypatch):
    monkeypatch.setattr("gdrive_rag.config.load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr("gdrive_rag.auth.load_credentials", lambda settings: None)

    result = CliRunner().invoke(cli, ["fetch", "abc"])

    assert result.exit_code != 0
    assert "login" in result.output.lower()


def test_fetch_prints_summary(monkeypatch):
    from gdrive_rag import parsers

    monkeypatch.setattr("gdrive_rag.config.load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr("gdrive_rag.auth.load_credentials", lambda settings: object())
    monkeypatch.setattr("gdrive_rag.drive.get_service", lambda creds: object())
    monkeypatch.setattr(
        "gdrive_rag.drive.get_file_metadata",
        lambda session, file_id: {"id": file_id, "name": "CS 213", "mimeType": "text/markdown"},
    )
    doc = parsers.ParsedDoc(
        "abc", "CS 213", "text/markdown", "markdown", "# Intro\nbody",
        sections=[parsers.Section(0, "Intro")], pages=[],
    )
    monkeypatch.setattr("gdrive_rag.parsers.fetch_document", lambda session, meta: doc)

    result = CliRunner().invoke(cli, ["fetch", "abc"])

    assert result.exit_code == 0, result.output
    assert "CS 213" in result.output
    assert "markdown" in result.output
    assert "Intro" in result.output


def test_fetch_unsupported(monkeypatch):
    monkeypatch.setattr("gdrive_rag.config.load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr("gdrive_rag.auth.load_credentials", lambda settings: object())
    monkeypatch.setattr("gdrive_rag.drive.get_service", lambda creds: object())
    monkeypatch.setattr(
        "gdrive_rag.drive.get_file_metadata",
        lambda session, file_id: {
            "id": file_id,
            "name": "Sheet",
            "mimeType": "application/vnd.google-apps.spreadsheet",
        },
    )

    result = CliRunner().invoke(cli, ["fetch", "abc"])

    assert result.exit_code != 0
    assert "unsupported" in result.output.lower()


def test_index_requires_login(monkeypatch):
    monkeypatch.setattr("gdrive_rag.config.load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test")
    monkeypatch.setattr("gdrive_rag.auth.load_credentials", lambda settings: None)

    result = CliRunner().invoke(cli, ["index"])

    assert result.exit_code != 0
    assert "login" in result.output.lower()


def test_index_missing_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("gdrive_rag.config.load_dotenv", lambda *a, **k: None)

    result = CliRunner().invoke(cli, ["index"])

    assert result.exit_code != 0
    assert "GEMINI_API_KEY" in result.output


def test_stats_empty(monkeypatch, tmp_path):
    import pytest

    pytest.importorskip("chromadb")
    monkeypatch.setattr("gdrive_rag.config.load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("GDRIVE_RAG_DATA_DIR", str(tmp_path))

    result = CliRunner().invoke(cli, ["stats"])

    assert result.exit_code == 0, result.output
    assert "files    : 0" in result.output
    assert "chunks   : 0" in result.output


def test_status_empty(monkeypatch, tmp_path):
    import pytest

    pytest.importorskip("chromadb")
    monkeypatch.setattr("gdrive_rag.config.load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("GDRIVE_RAG_DATA_DIR", str(tmp_path))

    result = CliRunner().invoke(cli, ["status"])

    assert result.exit_code == 0, result.output
    assert "No documents indexed yet" in result.output


def test_status_with_records(monkeypatch, tmp_path):
    import pytest

    pytest.importorskip("chromadb")
    from gdrive_rag.chunker import Chunk
    from gdrive_rag.store import Store

    monkeypatch.setattr("gdrive_rag.config.load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("GDRIVE_RAG_DATA_DIR", str(tmp_path))

    class _Settings:
        data_dir = tmp_path

    s = Store(_Settings())
    chunk = Chunk("f1", "OS Notes", "text/markdown", "markdown", 0, "text", {"type": "heading", "value": "A"})
    s.replace_file({"id": "f1", "name": "OS Notes", "mimeType": "text/markdown", "modifiedTime": "2026-08-14T10:00:00Z"}, [chunk], [[0.1, 0.2]])
    s.close()

    result = CliRunner().invoke(cli, ["status"])

    assert result.exit_code == 0, result.output
    assert "OS Notes" in result.output
    assert "Total Files Indexed : 1" in result.output


def test_index_delta_sync(monkeypatch, tmp_path):
    import pytest

    pytest.importorskip("chromadb")
    from gdrive_rag import parsers

    monkeypatch.setattr("gdrive_rag.config.load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test")
    monkeypatch.setenv("GDRIVE_RAG_DATA_DIR", str(tmp_path))
    monkeypatch.setattr("gdrive_rag.auth.load_credentials", lambda settings: object())
    monkeypatch.setattr("gdrive_rag.drive.get_service", lambda creds: object())

    files = [
        {"id": "1", "name": "Doc A", "mimeType": "text/markdown", "modifiedTime": "2026-08-14T10:00:00Z"},
        {"id": "2", "name": "Doc B", "mimeType": "text/markdown", "modifiedTime": "2026-08-14T11:00:00Z"},
    ]
    monkeypatch.setattr("gdrive_rag.drive.list_all_metadata", lambda session: files)
    monkeypatch.setattr(
        "gdrive_rag.parsers.fetch_document",
        lambda session, meta: parsers.ParsedDoc(
            meta["id"], meta["name"], meta["mimeType"], "markdown", "Some body content", [], []
        ),
    )
    monkeypatch.setattr("gdrive_rag.embed.embed_texts", lambda settings, texts, delay=None: [[0.1, 0.2, 0.3]] * len(texts))

    # First index run: indexes 2 files
    res1 = CliRunner().invoke(cli, ["index"])
    assert res1.exit_code == 0, res1.output
    assert "Indexed 2 file(s)" in res1.output

    # Second index run: both files are unchanged
    res2 = CliRunner().invoke(cli, ["index"])
    assert res2.exit_code == 0, res2.output
    assert "All 2 document(s) are up-to-date" in res2.output


def test_search_prints_hits(monkeypatch, tmp_path):
    import pytest

    pytest.importorskip("chromadb")
    from gdrive_rag import retrieve

    monkeypatch.setattr("gdrive_rag.config.load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test")
    monkeypatch.setenv("GDRIVE_RAG_DATA_DIR", str(tmp_path))
    hit = retrieve.SearchHit(
        score=0.884,
        name="CS 213",
        locator={"type": "page", "value": "4"},
        drive_url="https://drive/x",
        text="CS 213 — p.4\n\nA process is a program in execution.",
        chunk_index=0,
    )
    monkeypatch.setattr(
        "gdrive_rag.retrieve.search", lambda settings, store, query, k=6, client=None: [hit]
    )

    result = CliRunner().invoke(cli, ["search", "what is a process"])

    assert result.exit_code == 0, result.output
    assert "CS 213" in result.output
    assert "0.884" in result.output
    assert "p.4" in result.output
    assert "program in execution" in result.output


def test_search_missing_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("gdrive_rag.config.load_dotenv", lambda *a, **k: None)

    result = CliRunner().invoke(cli, ["search", "q"])

    assert result.exit_code != 0
    assert "GEMINI_API_KEY" in result.output


def test_login_missing_credentials(monkeypatch, tmp_path):
    monkeypatch.setattr("gdrive_rag.config.load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("GDRIVE_RAG_CREDENTIALS", str(tmp_path / "nope.json"))
    monkeypatch.setenv("GDRIVE_RAG_TOKEN", str(tmp_path / "token.json"))

    result = CliRunner().invoke(cli, ["login"])

    assert result.exit_code != 0
    assert "not found" in result.output.lower()


def test_ask_prints_answer_and_citations(monkeypatch, tmp_path):
    import pytest

    pytest.importorskip("chromadb")
    from gdrive_rag import answer

    monkeypatch.setattr("gdrive_rag.config.load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test")
    monkeypatch.setenv("GDRIVE_RAG_DATA_DIR", str(tmp_path))

    ans = answer.Answer(
        query="what is a process",
        text="A process is an instance of a program in execution [1].",
        citations=[
            answer.Citation(
                index=1,
                name="OS Notes",
                locator={"type": "page", "value": "4"},
                drive_url="https://drive.google.com/open?id=123",
            )
        ],
        hits=[],
    )

    monkeypatch.setattr(
        "gdrive_rag.answer.answer_query", lambda settings, store, query, k=6, client=None: ans
    )

    result = CliRunner().invoke(cli, ["ask", "what is a process"])

    assert result.exit_code == 0, result.output
    assert "instance of a program in execution [1]" in result.output
    assert "Sources:" in result.output
    assert "[1] OS Notes · p.4 — https://drive.google.com/open?id=123" in result.output


def test_ask_missing_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("gdrive_rag.config.load_dotenv", lambda *a, **k: None)

    result = CliRunner().invoke(cli, ["ask", "what is a process"])

    assert result.exit_code != 0
    assert "GEMINI_API_KEY" in result.output


def test_ask_reports_error(monkeypatch, tmp_path):
    import pytest

    pytest.importorskip("chromadb")

    monkeypatch.setattr("gdrive_rag.config.load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test")
    monkeypatch.setenv("GDRIVE_RAG_DATA_DIR", str(tmp_path))

    def _boom(*a, **k):
        raise RuntimeError("Gemini connection timed out")

    monkeypatch.setattr("gdrive_rag.answer.answer_query", _boom)

    result = CliRunner().invoke(cli, ["ask", "what is a process"])

    assert result.exit_code != 0
    assert "Ask failed" in result.output


def test_index_prints_failure_details(monkeypatch, tmp_path):
    import pytest

    pytest.importorskip("chromadb")

    monkeypatch.setattr("gdrive_rag.config.load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test")
    monkeypatch.setenv("GDRIVE_RAG_DATA_DIR", str(tmp_path))
    monkeypatch.setattr("gdrive_rag.auth.load_credentials", lambda settings: object())
    monkeypatch.setattr("gdrive_rag.drive.get_service", lambda creds: object())

    files = [
        {"id": "1", "name": "Broken_File.pdf", "mimeType": "application/pdf", "modifiedTime": "2026-08-14T10:00:00Z"},
    ]
    monkeypatch.setattr("gdrive_rag.drive.list_all_metadata", lambda session: files)

    def _fail_fetch(session, meta):
        raise RuntimeError("Corrupt PDF stream")

    monkeypatch.setattr("gdrive_rag.parsers.fetch_document", _fail_fetch)

    result = CliRunner().invoke(cli, ["index"])

    assert result.exit_code == 0, result.output
    assert "failed 1" in result.output
    assert "Failed document(s) (1):" in result.output
    assert "Broken_File.pdf: Corrupt PDF stream" in result.output

