"""Google Drive OAuth: obtain, store (token.json, 0600), load, and refresh credentials.

Uses a manual copy-paste flow so it works on a headless server (no local browser and no
port forwarding required): we print the consent URL, the user approves in any browser, and
pastes back the redirected ``http://localhost`` URL that carries the authorization code.
"""

from __future__ import annotations

import os
import stat

import click
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow

from .config import ConfigError, Settings

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
_LOOPBACK_REDIRECT = "http://localhost:8765"


def load_credentials(settings: Settings) -> Credentials | None:
    """Return valid Credentials from the token file, refreshing if needed.

    Returns None if there is no token yet or it can't be refreshed (e.g. the 7-day
    Testing-mode refresh token has expired) — the caller should tell the user to log in.
    """
    token_path = settings.token_path
    if not token_path.exists():
        return None

    creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    if creds.valid:
        return creds
    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
        except Exception:
            return None
        _save_credentials(creds, settings)
        return creds
    return None


def run_manual_login(settings: Settings) -> Credentials:
    """Interactive copy-paste OAuth flow. Writes the token file (0600) and returns creds."""
    creds_path = settings.credentials_path
    if not creds_path.exists():
        raise ConfigError(
            f"OAuth client file not found at {creds_path}. Download a Desktop-app OAuth "
            "client JSON from Google Cloud Console and save it there (see the Stage 1 setup)."
        )

    # oauthlib rejects a plain-http redirect and errors on any returned-scope reordering;
    # both are safe to relax for a localhost loopback redirect.
    os.environ.setdefault("OAUTHLIB_INSECURE_TRANSPORT", "1")
    os.environ.setdefault("OAUTHLIB_RELAX_TOKEN_SCOPE", "1")

    flow = Flow.from_client_secrets_file(
        str(creds_path), scopes=SCOPES, redirect_uri=_LOOPBACK_REDIRECT
    )
    auth_url, _ = flow.authorization_url(access_type="offline", prompt="consent")

    click.echo("\n1. Open this URL in a browser (on any machine) and approve access:\n")
    click.echo(f"   {auth_url}\n")
    click.echo(
        "2. Your browser will redirect to a 'localhost' page that fails to load — that's\n"
        "   expected. Copy the FULL address-bar URL (it contains ?code=...) and paste it below.\n"
    )
    redirect_response = click.prompt("Pasted redirect URL").strip()

    flow.fetch_token(authorization_response=redirect_response)
    creds = flow.credentials
    _save_credentials(creds, settings)
    return creds


def _save_credentials(creds: Credentials, settings: Settings) -> None:
    token_path = settings.token_path
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(creds.to_json())
    token_path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0600 — secret
