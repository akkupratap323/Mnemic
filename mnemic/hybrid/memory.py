"""
Copyright 2026, Abishek (Mnemic project).

HybridMemory: the facade that ties the cheap vector tier to the premium graph
tier via the write router. The vector tier is the source of truth — a graph
write failure never loses a memory.
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from mnemic.hybrid.embedder import Embedder
from mnemic.hybrid.errors import InvalidInput
from mnemic.hybrid.router import WriteRouter
from mnemic.hybrid.types import MemoryItem, RememberResult, SearchHit
from mnemic.hybrid.vector_store import VectorStore


@runtime_checkable
class GraphMemory(Protocol):
    """The premium tier — anything exposing an async ``remember`` (e.g. Mnemic)."""

    async def remember(self, **kwargs: object) -> object: ...


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return uuid.uuid4().hex


@dataclass
class HybridMemory:
    """Routes every memory to the vector tier, and signal to the graph tier."""

    embedder: Embedder
    vector_store: VectorStore
    router: WriteRouter = field(default_factory=WriteRouter)
    graph: GraphMemory | None = None
    clock: Callable[[], datetime] = _utcnow
    id_factory: Callable[[], str] = _uuid

    async def remember(
        self,
        content: str,
        *,
        metadata: Mapping[str, object] | None = None,
        name: str | None = None,
        source_description: str = '',
        reference_time: datetime | None = None,
        group_id: str | None = None,
    ) -> RememberResult:
        if not isinstance(content, str) or not content.strip():
            raise InvalidInput('content must be a non-empty string')

        embedding = await self.embedder.embed(content)
        created_at = self.clock()
        item = MemoryItem(
            id=self.id_factory(),
            content=content,
            embedding=embedding,
            metadata=dict(metadata or {}),
            created_at=created_at,
        )
        await self.vector_store.add(item)  # vector tier is the source of truth

        decision = self.router.route(content, metadata)
        graph_written = False
        graph_error: str | None = None
        if decision.writes_graph and self.graph is not None:
            try:
                kwargs: dict[str, object] = {
                    'name': name or f'memory-{item.id}',
                    'episode_body': content,
                    'source_description': source_description,
                    'reference_time': reference_time or created_at,
                }
                if group_id is not None:
                    kwargs['group_id'] = group_id
                await self.graph.remember(**kwargs)
                graph_written = True
            except Exception as exc:  # graph failure must not lose the vector write
                graph_error = f'{type(exc).__name__}: {exc}'

        return RememberResult(
            item=item,
            decision=decision,
            graph_written=graph_written,
            graph_error=graph_error,
        )

    async def recall(
        self,
        query: str,
        *,
        k: int = 10,
        where: Mapping[str, object] | None = None,
    ) -> list[SearchHit]:
        if not isinstance(query, str) or not query.strip():
            raise InvalidInput('query must be a non-empty string')
        embedding = await self.embedder.embed(query)
        return await self.vector_store.search(embedding, k=k, where=where)
