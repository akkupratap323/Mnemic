"""
Copyright 2026, Abishek (Mnemic project).

Immutable value types for the hybrid memory layer.
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from types import MappingProxyType

from mnemic.hybrid.errors import InvalidInput


class Tier(str, Enum):
    """Which memory tiers an item is written to."""

    VECTOR_ONLY = 'vector_only'
    VECTOR_AND_GRAPH = 'vector_and_graph'


@dataclass(frozen=True)
class MemoryItem:
    """A single stored memory. Immutable; metadata is defensively copied."""

    id: str
    content: str
    embedding: tuple[float, ...]
    metadata: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))
    created_at: datetime | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise InvalidInput('MemoryItem.id must be a non-empty string')
        if not isinstance(self.content, str) or not self.content.strip():
            raise InvalidInput('MemoryItem.content must be a non-empty string')
        embedding = tuple(float(x) for x in self.embedding)
        if len(embedding) == 0:
            raise InvalidInput('MemoryItem.embedding must be non-empty')
        # freeze derived/defensive copies (dataclass is frozen)
        object.__setattr__(self, 'embedding', embedding)
        object.__setattr__(self, 'metadata', MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class SearchHit:
    """A search result: an item and its similarity score."""

    item: MemoryItem
    score: float


@dataclass(frozen=True)
class RouteDecision:
    """The router's verdict for where a memory should be written."""

    tier: Tier
    reasons: tuple[str, ...] = ()

    @property
    def writes_graph(self) -> bool:
        return self.tier is Tier.VECTOR_AND_GRAPH


@dataclass(frozen=True)
class RememberResult:
    """The outcome of a HybridMemory.remember call."""

    item: MemoryItem
    decision: RouteDecision
    graph_written: bool = False
    graph_error: str | None = None
