"""
Copyright 2026, Abishek (Mnemic project).

The cheap, high-volume vector tier. Defines the VectorStore protocol and a
deterministic in-memory implementation. Production stores (sqlite-vec, Qdrant,
pgvector) implement the same protocol.
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from mnemic.hybrid.errors import DimensionMismatch, DuplicateItem, InvalidInput
from mnemic.hybrid.types import MemoryItem, SearchHit


@runtime_checkable
class VectorStore(Protocol):
    """Storage + cosine retrieval for embedded memories."""

    async def add(self, item: MemoryItem) -> None: ...

    async def search(
        self,
        embedding: Sequence[float],
        *,
        k: int = 10,
        where: Mapping[str, object] | None = None,
    ) -> list[SearchHit]: ...

    async def get(self, item_id: str) -> MemoryItem | None: ...

    async def delete(self, item_id: str) -> bool: ...

    async def count(self) -> int: ...

    async def all_items(self) -> list[MemoryItem]: ...


class InMemoryVectorStore:
    """Dependency-free vector store for dev, tests, and small deployments.

    Enforces a single embedding dimension and rejects duplicate ids and
    zero-magnitude vectors, so failures surface loudly rather than silently.
    """

    def __init__(self) -> None:
        self._items: dict[str, MemoryItem] = {}
        self._dim: int | None = None

    async def add(self, item: MemoryItem) -> None:
        if item.id in self._items:
            raise DuplicateItem(f'item id already exists: {item.id!r}')
        dim = len(item.embedding)
        if self._dim is None:
            self._dim = dim
        elif dim != self._dim:
            raise DimensionMismatch(f'expected dim {self._dim}, got {dim}')
        _norm(item.embedding)  # validates non-zero magnitude
        self._items[item.id] = item

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
        for item in self._items.values():
            if where and not _matches(item.metadata, where):
                continue
            score = _dot(query, item.embedding) / (query_norm * _norm(item.embedding))
            hits.append(SearchHit(item=item, score=score))

        # deterministic ordering: highest score first, ties broken by id
        hits.sort(key=lambda h: (-h.score, h.item.id))
        return hits[:k]

    async def get(self, item_id: str) -> MemoryItem | None:
        return self._items.get(item_id)

    async def delete(self, item_id: str) -> bool:
        return self._items.pop(item_id, None) is not None

    async def count(self) -> int:
        return len(self._items)

    async def all_items(self) -> list[MemoryItem]:
        return list(self._items.values())


def _matches(metadata: Mapping[str, object], where: Mapping[str, object]) -> bool:
    return all(metadata.get(key) == value for key, value in where.items())


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise DimensionMismatch(f'dimension mismatch: {len(a)} vs {len(b)}')
    return math.fsum(x * y for x, y in zip(a, b, strict=True))


def _norm(vec: Sequence[float]) -> float:
    magnitude = math.sqrt(math.fsum(x * x for x in vec))
    if magnitude == 0.0:
        raise InvalidInput('embedding must have non-zero magnitude')
    return magnitude
