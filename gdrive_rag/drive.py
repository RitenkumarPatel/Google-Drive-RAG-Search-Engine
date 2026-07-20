"""Google Drive API helpers via a requests-based AuthorizedSession.

We use requests (via google-auth's AuthorizedSession) rather than the httplib2-based
google-api-python-client: httplib2 does not reliably route through the http(s)_proxy
environment variables, which breaks on proxied networks. requests honors them (the same
reason the httpx-based Gemini client works).
"""

from __future__ import annotations

from google.auth.transport.requests import AuthorizedSession

_API = "https://www.googleapis.com/drive/v3"
_PAGE = 100  # Drive API max page size is 1000; 100 is plenty for interactive listing.
_TIMEOUT = 30


def get_service(creds) -> AuthorizedSession:
    """Return an authorized requests Session for the Drive REST API."""
    return AuthorizedSession(creds)


def list_files(session, limit: int = 20) -> list[dict]:
    """Return up to ``limit`` non-trashed files, newest first.

    Each item has id / name / mimeType / modifiedTime. Handles pagination.
    """
    files: list[dict] = []
    page_token = None
    while len(files) < limit:
        params = {
            "pageSize": min(_PAGE, limit - len(files)),
            "fields": "nextPageToken, files(id,name,mimeType,modifiedTime)",
            "orderBy": "modifiedTime desc",
            "q": "trashed=false",
        }
        if page_token:
            params["pageToken"] = page_token
        resp = session.get(f"{_API}/files", params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        files.extend(data.get("files", []))
        page_token = data.get("nextPageToken")
        if not page_token:
            break
    return files[:limit]


def get_user_email(session) -> str | None:
    """Best-effort authenticated account email (None if unavailable)."""
    try:
        resp = session.get(
            f"{_API}/about",
            params={"fields": "user(emailAddress,displayName)"},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json().get("user", {}).get("emailAddress")
    except Exception:
        return None
