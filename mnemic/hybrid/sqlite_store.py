"""
Copyright 2026, Abishek (Mnemic project).

A persistent, async-safe vector store backed by stdlib sqlite3 (no native
extensions). Blocking DB I/O is offloaded to a thread and serialised with a
lock, so it never blocks the event loop. Cosine is computed in Python — fine for
moderate scale; swap for sqlite-vec / Qdrant / pgvector (same protocol) at scale.
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from collections.abc import Mapping, Sequence
from datetime import datetime

from mnemic.hybrid.errors import DimensionMismatch, DuplicateItem, InvalidInput
from mnemic.hybrid.types import MemoryItem, SearchHit
from mnemic.hybrid.vector_store import _dot, _matches, _norm

_SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    embedding TEXT NOT NULL,
    metadata TEXT NOT NULL,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


class SQLiteVectorStore:
    """Durable, async-safe VectorStore. Enforces one embedding dimension."""

    def __init__(self, path: str = ':memory:') -> None:
        # check_same_thread=False + a lock lets us safely call from a thread pool
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
            row = self._conn.execute("SELECT value FROM meta WHERE key = 'dim'").fetchone()
        self._dim: int | None = int(row['value']) if row else None

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    async def add(self, item: MemoryItem) -> None:
        _norm(item.embedding)  # reject zero-magnitude (pure, before offload)
        try:
            metadata_json = json.dumps(dict(item.metadata))
        except TypeError as exc:
            raise InvalidInput(f'metadata must be JSON-serialisable: {exc}') from exc
        embedding_json = json.dumps(list(item.embedding))
        created = item.created_at.isoformat() if item.created_at else None
        await asyncio.to_thread(self._sync_add, item, embedding_json, metadata_json, created)

    def _sync_add(
        self, item: MemoryItem, embedding_json: str, metadata_json: str, created: str | None
    ) -> None:
        with self._lock:
            dim = len(item.embedding)
            if self._dim is None:
                self._dim = dim
                self._conn.execute(
                    "INSERT OR REPLACE INTO meta (key, value) VALUES ('dim', ?)", (str(dim),)
                )
            elif dim != self._dim:
                raise DimensionMismatch(f'expected dim {self._dim}, got {dim}')
            try:
                self._conn.execute(
                    'INSERT INTO items (id, content, embedding, metadata, created_at) '
                    'VALUES (?, ?, ?, ?, ?)',
                    (item.id, item.content, embedding_json, metadata_json, created),
                )
            except sqlite3.IntegrityError as exc:
                raise DuplicateItem(f'item id already exists: {item.id!r}') from exc
            self._conn.commit()

    async def search(
        self,
        embedding: Sequence[float],
        *,
        k: int = 10,
        where: Mapping[str, object] | None = None,
    ) -> list[SearchHit]:
        if k <= 0:
            raise InvalidInput('k must be a positive integer')
        query = tuple(float(x) for x in embedding)
        query_norm = _norm(query)
        return await asyncio.to_thread(self._sync_search, query, query_norm, k, where)

    def _sync_search(
        self,
        query: tuple[float, ...],
        query_norm: float,
        k: int,
        where: Mapping[str, object] | None,
    ) -> list[SearchHit]:
        with self._lock:
            if self._dim is not None and len(query) != self._dim:
                raise DimensionMismatch(f'expected dim {self._dim}, got {len(query)}')
            rows = self._conn.execute('SELECT * FROM items').fetchall()
        hits: list[SearchHit] = []
        for row in rows:
            item = _row_to_item(row)
            if where and not _matches(item.metadata, where):
                continue
            score = _dot(query, item.embedding) / (query_norm * _norm(item.embedding))
            hits.append(SearchHit(item=item, score=score))
        hits.sort(key=lambda h: (-h.score, h.item.id))
        return hits[:k]

    async def get(self, item_id: str) -> MemoryItem | None:
        return await asyncio.to_thread(self._sync_get, item_id)

    def _sync_get(self, item_id: str) -> MemoryItem | None:
        with self._lock:
            row = self._conn.execute('SELECT * FROM items WHERE id = ?', (item_id,)).fetchone()
        return _row_to_item(row) if row else None

    async def delete(self, item_id: str) -> bool:
        return await asyncio.to_thread(self._sync_delete, item_id)

    def _sync_delete(self, item_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute('DELETE FROM items WHERE id = ?', (item_id,))
            self._conn.commit()
            return cur.rowcount > 0

    async def count(self) -> int:
        return await asyncio.to_thread(self._sync_count)

    def _sync_count(self) -> int:
        with self._lock:
            return int(self._conn.execute('SELECT COUNT(*) FROM items').fetchone()[0])

    async def all_items(self) -> list[MemoryItem]:
        return await asyncio.to_thread(self._sync_all_items)

    def _sync_all_items(self) -> list[MemoryItem]:
        with self._lock:
            rows = self._conn.execute('SELECT * FROM items').fetchall()
        return [_row_to_item(row) for row in rows]


def _row_to_item(row: sqlite3.Row) -> MemoryItem:
    created = row['created_at']
    return MemoryItem(
        id=row['id'],
        content=row['content'],
        embedding=tuple(json.loads(row['embedding'])),
        metadata=json.loads(row['metadata']),
        created_at=datetime.fromisoformat(created) if created else None,
    )
