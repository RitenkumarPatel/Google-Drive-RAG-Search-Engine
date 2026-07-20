"""Tests for gdrive_rag.drive.list_files pagination — the HTTP session is faked."""

from gdrive_rag import drive


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeSession:
    """Fakes AuthorizedSession.get, chaining pages by pageToken."""

    def __init__(self, pages):
        self._by_token = {None: pages[0]}
        for i, p in enumerate(pages[:-1]):
            self._by_token[p["nextPageToken"]] = pages[i + 1]
        self.calls = []

    def get(self, url, params=None, timeout=None):
        params = params or {}
        self.calls.append(params)
        return _FakeResp(self._by_token[params.get("pageToken")])


def test_list_files_paginates():
    pages = [
        {
            "files": [
                {"id": "1", "name": "a", "mimeType": "x", "modifiedTime": "t1"},
                {"id": "2", "name": "b", "mimeType": "y", "modifiedTime": "t2"},
            ],
            "nextPageToken": "T2",
        },
        {"files": [{"id": "3", "name": "c", "mimeType": "z", "modifiedTime": "t3"}], "nextPageToken": None},
    ]
    out = drive.list_files(_FakeSession(pages), limit=3)
    assert [f["id"] for f in out] == ["1", "2", "3"]


def test_list_files_respects_limit():
    pages = [{"files": [{"id": "1"}, {"id": "2"}, {"id": "3"}], "nextPageToken": "T2"}]
    out = drive.list_files(_FakeSession(pages), limit=2)
    assert len(out) == 2


def test_list_files_empty():
    pages = [{"files": [], "nextPageToken": None}]
    assert drive.list_files(_FakeSession(pages), limit=10) == []
