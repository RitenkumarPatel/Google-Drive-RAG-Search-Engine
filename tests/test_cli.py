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
