"""
Copyright 2026, Abishek (Mnemic project).

A persistent vector store backed by stdlib sqlite3 (no native extensions).
Cosine is computed in Python — fine for moderate scale and dev/single-node use;
swap for sqlite-vec / Qdrant / pgvector (same VectorStore protocol) at scale.
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import json
import sqlite3
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
    """Durable VectorStore. Enforces a single embedding dimension across restarts."""

    def __init__(self, path: str = ':memory:') -> None:
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        row = self._conn.execute("SELECT value FROM meta WHERE key = 'dim'").fetchone()
        self._dim: int | None = int(row['value']) if row else None

    def close(self) -> None:
        self._conn.close()

    async def add(self, item: MemoryItem) -> None:
        dim = len(item.embedding)
        if self._dim is None:
            self._dim = dim
            self._conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('dim', ?)", (str(dim),)
            )
        elif dim != self._dim:
            raise DimensionMismatch(f'expected dim {self._dim}, got {dim}')
        _norm(item.embedding)  # reject zero-magnitude

        try:
            metadata_json = json.dumps(dict(item.metadata))
        except TypeError as exc:
            raise InvalidInput(f'metadata must be JSON-serialisable: {exc}') from exc

        created = item.created_at.isoformat() if item.created_at else None
        try:
            self._conn.execute(
                'INSERT INTO items (id, content, embedding, metadata, created_at) '
                'VALUES (?, ?, ?, ?, ?)',
                (item.id, item.content, json.dumps(list(item.embedding)), metadata_json, created),
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
        if self._dim is not None and len(query) != self._dim:
            raise DimensionMismatch(f'expected dim {self._dim}, got {len(query)}')
        query_norm = _norm(query)

        hits: list[SearchHit] = []
        for row in self._conn.execute('SELECT * FROM items'):
            item = _row_to_item(row)
            if where and not _matches(item.metadata, where):
                continue
            score = _dot(query, item.embedding) / (query_norm * _norm(item.embedding))
            hits.append(SearchHit(item=item, score=score))
        hits.sort(key=lambda h: (-h.score, h.item.id))
        return hits[:k]

    async def get(self, item_id: str) -> MemoryItem | None:
        row = self._conn.execute('SELECT * FROM items WHERE id = ?', (item_id,)).fetchone()
        return _row_to_item(row) if row else None

    async def delete(self, item_id: str) -> bool:
        cur = self._conn.execute('DELETE FROM items WHERE id = ?', (item_id,))
        self._conn.commit()
        return cur.rowcount > 0

    async def count(self) -> int:
        return int(self._conn.execute('SELECT COUNT(*) FROM items').fetchone()[0])

    async def all_items(self) -> list[MemoryItem]:
        return [_row_to_item(row) for row in self._conn.execute('SELECT * FROM items')]


def _row_to_item(row: sqlite3.Row) -> MemoryItem:
    created = row['created_at']
    return MemoryItem(
        id=row['id'],
        content=row['content'],
        embedding=tuple(json.loads(row['embedding'])),
        metadata=json.loads(row['metadata']),
        created_at=datetime.fromisoformat(created) if created else None,
    )
