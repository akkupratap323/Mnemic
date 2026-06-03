"""
Copyright 2026, Abishek (Mnemic project).

Read-side fusion: blend ranked results from the vector tier and the graph tier
into a single ranking via Reciprocal Rank Fusion (RRF).
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from mnemic.hybrid.errors import InvalidInput

DEFAULT_RRF_K = 60


@dataclass(frozen=True)
class GraphFact:
    """A fact returned from the graph tier."""

    id: str
    fact: str
    metadata: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True)
class FusedHit:
    """A unified, ranked result blended across tiers."""

    content: str
    score: float
    sources: tuple[str, ...]


@runtime_checkable
class GraphSearcher(Protocol):
    """Anything that can return ranked facts for a query (e.g. a Mnemic client)."""

    async def search(self, query: str, *, num_results: int = 10) -> list[GraphFact]: ...


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[str]], *, k: int = DEFAULT_RRF_K
) -> dict[str, float]:
    """Fuse multiple ranked key-lists into combined scores.

    A key appearing high in several lists scores higher than one appearing in
    only one — the core reason fusion beats either tier alone.
    """
    if k <= 0:
        raise InvalidInput('rrf k must be a positive integer')
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, key in enumerate(ranked, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    return scores


def normalize_text(text: str) -> str:
    """Collapse whitespace and lowercase, for cross-tier content matching."""
    return ' '.join(text.lower().split())


class MnemicGraphSearcher:
    """Adapts a Mnemic client's ``search`` (returns EntityEdges) to GraphFacts."""

    def __init__(self, client: object) -> None:
        self._client = client

    async def search(self, query: str, *, num_results: int = 10) -> list[GraphFact]:
        if not isinstance(query, str) or not query.strip():
            raise InvalidInput('query must be a non-empty string')
        edges = await self._client.search(query, num_results=num_results)  # type: ignore[attr-defined]
        return [
            GraphFact(id=str(getattr(edge, 'uuid', '')), fact=str(getattr(edge, 'fact', edge)))
            for edge in edges
        ]
