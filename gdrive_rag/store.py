"""Persist embedded chunks: Chroma (cosine vectors + metadata) + SQLite sync state.

Chroma holds the vectors, chunk text, and citation metadata; SQLite tracks per-file
state (``content_version``, chunk count) so re-indexing can skip unchanged files and
reconcile deletions. Chunk IDs are deterministic — ``uuid5(file_id:content_version:index)``
— and each file is replaced wholesale (delete-then-add), so edits and shrinks leave no
orphan chunks. Chroma is always given explicit embeddings, so its default (ONNX) embedding
function is never instantiated and nothing is downloaded.
"""

from __future__ import annotations

import sqlite3
import time
import uuid

_COLLECTION = "documents"
_NAMESPACE = uuid.NAMESPACE_URL


def content_version(meta: dict) -> str:
    """The signal that a file's content changed (drives cache invalidation + IDs)."""
    return meta.get("modifiedTime") or meta.get("md5Checksum") or meta.get("version") or ""


def chunk_id(file_id: str, cv: str, index: int) -> str:
    return str(uuid.uuid5(_NAMESPACE, f"{file_id}:{cv}:{index}"))


def drive_url(file_id: str) -> str:
    return f"https://drive.google.com/open?id={file_id}"


class Store:
    def __init__(self, settings) -> None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        data_dir = settings.data_dir
        data_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(data_dir / "chroma"),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION, metadata={"hnsw:space": "cosine"}
        )
        self._db = sqlite3.connect(str(data_dir / "state.db"))
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS files (
                 file_id TEXT PRIMARY KEY,
                 name TEXT,
                 content_version TEXT,
                 chunk_count INTEGER,
                 indexed_at REAL
               )"""
        )
        self._db.commit()

    # --- reads ---
    def get_indexed_version(self, file_id: str):
        row = self._db.execute(
            "SELECT content_version FROM files WHERE file_id = ?", (file_id,)
        ).fetchone()
        return row[0] if row else None

    def list_indexed_ids(self) -> list[str]:
        return [r[0] for r in self._db.execute("SELECT file_id FROM files").fetchall()]

    def stats(self) -> dict:
        files = self._db.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        return {"files": files, "chunks": self._collection.count()}

    def query(self, embedding, k: int = 6) -> list[dict]:
        """Return the ``k`` nearest chunks to ``embedding`` (cosine), best first.

        Each result is ``{id, document, metadata, score}`` where ``score`` is cosine
        similarity (``1 - distance``, since the collection uses the cosine space).
        """
        res = self._collection.query(
            query_embeddings=[list(embedding)],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        return [
            {"id": ids[i], "document": docs[i], "metadata": metas[i], "score": 1.0 - dists[i]}
            for i in range(len(ids))
        ]

    # --- writes ---
    def replace_file(self, meta: dict, chunks, embeddings) -> None:
        """Delete a file's existing chunks, then insert the new ones (idempotent)."""
        file_id = meta["id"]
        cv = content_version(meta)
        self._collection.delete(where={"file_id": file_id})
        if chunks:
            self._collection.add(
                ids=[chunk_id(file_id, cv, c.index) for c in chunks],
                embeddings=list(embeddings),
                documents=[c.text for c in chunks],
                metadatas=[self._metadata(meta, cv, c) for c in chunks],
            )
        self._db.execute(
            """INSERT INTO files (file_id, name, content_version, chunk_count, indexed_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(file_id) DO UPDATE SET
                 name=excluded.name, content_version=excluded.content_version,
                 chunk_count=excluded.chunk_count, indexed_at=excluded.indexed_at""",
            (file_id, meta.get("name", ""), cv, len(chunks), time.time()),
        )
        self._db.commit()

    def purge_file(self, file_id: str) -> None:
        self._collection.delete(where={"file_id": file_id})
        self._db.execute("DELETE FROM files WHERE file_id = ?", (file_id,))
        self._db.commit()

    def reconcile(self, live_ids) -> list[str]:
        """Purge indexed files whose IDs are no longer present in ``live_ids``."""
        live = set(live_ids)
        removed = [fid for fid in self.list_indexed_ids() if fid not in live]
        for fid in removed:
            self.purge_file(fid)
        return removed

    def close(self) -> None:
        self._db.close()

    @staticmethod
    def _metadata(meta: dict, cv: str, chunk) -> dict:
        return {
            "file_id": meta["id"],
            "name": meta.get("name", ""),
            "mime_type": meta.get("mimeType", ""),
            "fmt": chunk.fmt,
            "chunk_index": chunk.index,
            "content_version": cv,
            "loc_type": chunk.locator.get("type", ""),
            "loc_value": str(chunk.locator.get("value", "")),
            "drive_url": drive_url(meta["id"]),
        }
