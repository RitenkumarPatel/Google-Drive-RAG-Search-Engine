"""Tests for gdrive_rag.auth. No network/browser — Credentials + Request are stubbed."""

import json
import stat

from gdrive_rag import auth
from gdrive_rag.config import Settings


def _settings(tmp_path) -> Settings:
    return Settings(
        gemini_api_key="",
        chat_model="m",
        embed_model="e",
        embed_dims=768,
        data_dir=tmp_path,
        credentials_path=tmp_path / "credentials.json",
        token_path=tmp_path / "token.json",
    )


class _FakeCreds:
    def __init__(self, valid=True, expired=False, refresh_token="r"):
        self.valid = valid
        self.expired = expired
        self.refresh_token = refresh_token
        self.refreshed = False

    def to_json(self):
        return json.dumps({"token": "x"})

    def refresh(self, request):
        self.refreshed = True
        self.valid = True
        self.expired = False


def test_load_credentials_none_when_missing(tmp_path):
    assert auth.load_credentials(_settings(tmp_path)) is None


def test_load_credentials_valid(tmp_path, monkeypatch):
    s = _settings(tmp_path)
    s.token_path.write_text("{}")
    fake = _FakeCreds(valid=True)
    monkeypatch.setattr(auth.Credentials, "from_authorized_user_file", lambda *a, **k: fake)
    assert auth.load_credentials(s) is fake


def test_load_credentials_refreshes_and_resaves(tmp_path, monkeypatch):
    s = _settings(tmp_path)
    s.token_path.write_text("{}")
    fake = _FakeCreds(valid=False, expired=True, refresh_token="r")
    monkeypatch.setattr(auth.Credentials, "from_authorized_user_file", lambda *a, **k: fake)
    monkeypatch.setattr(auth, "Request", lambda: object())

    out = auth.load_credentials(s)

    assert out is fake and fake.refreshed
    assert s.token_path.read_text()  # re-saved
    assert stat.S_IMODE(s.token_path.stat().st_mode) == 0o600


def test_load_credentials_none_when_unrefreshable(tmp_path, monkeypatch):
    s = _settings(tmp_path)
    s.token_path.write_text("{}")
    fake = _FakeCreds(valid=False, expired=True, refresh_token=None)
    monkeypatch.setattr(auth.Credentials, "from_authorized_user_file", lambda *a, **k: fake)
    assert auth.load_credentials(s) is None


def test_save_credentials_is_0600(tmp_path):
    s = _settings(tmp_path)
    auth._save_credentials(_FakeCreds(), s)
    assert stat.S_IMODE(s.token_path.stat().st_mode) == 0o600
