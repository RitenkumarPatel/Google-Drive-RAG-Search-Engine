"""Tests for gdrive_rag.config. All use load_env=False to stay isolated from a real .env."""

import pytest

from gdrive_rag.config import ConfigError, Settings, load_settings


def test_reads_env_and_applies_defaults(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test")
    monkeypatch.setenv("GEMINI_CHAT_MODEL", "gemini-x")
    monkeypatch.delenv("LOCAL_EMBED_MODEL", raising=False)

    s = load_settings(load_env=False)

    assert isinstance(s, Settings)
    assert s.gemini_api_key == "AIza-test"
    assert s.chat_model == "gemini-x"                        # from env
    assert s.local_embed_model == "BAAI/bge-base-en-v1.5"   # default


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ConfigError):
        load_settings(load_env=False)


def test_missing_key_ok_when_not_required(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    s = load_settings(require_api_key=False, load_env=False)
    assert s.gemini_api_key == ""


def test_blank_key_treated_as_missing(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "   ")
    with pytest.raises(ConfigError):
        load_settings(load_env=False)
