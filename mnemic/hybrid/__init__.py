"""
Copyright 2026, Abishek (Mnemic project).

Hybrid tiered-memory layer for Mnemic: a cheap, high-volume vector tier plus a
write router that promotes only signal to the premium temporal-graph tier.
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

from mnemic.hybrid.access import AccessLog
from mnemic.hybrid.classifier import SignalClassifier
from mnemic.hybrid.consolidation import (
    ConsolidationPolicy,
    ConsolidationReport,
    Consolidator,
)
from mnemic.hybrid.embedder import Embedder, EmbedderClientAdapter
from mnemic.hybrid.errors import (
    DimensionMismatch,
    DuplicateItem,
    HybridMemoryError,
    InvalidInput,
)
from mnemic.hybrid.fusion import (
    FusedHit,
    GraphFact,
    GraphSearcher,
    MnemicGraphSearcher,
    normalize_text,
    reciprocal_rank_fusion,
)
from mnemic.hybrid.memory import GraphMemory, HybridMemory
from mnemic.hybrid.router import WriteRouter
from mnemic.hybrid.types import (
    MemoryItem,
    RememberResult,
    RouteDecision,
    SearchHit,
    Tier,
)
from mnemic.hybrid.vector_store import InMemoryVectorStore, VectorStore

__all__ = [
    'AccessLog',
    'ConsolidationPolicy',
    'ConsolidationReport',
    'Consolidator',
    'DimensionMismatch',
    'DuplicateItem',
    'Embedder',
    'EmbedderClientAdapter',
    'FusedHit',
    'GraphFact',
    'GraphMemory',
    'GraphSearcher',
    'HybridMemory',
    'HybridMemoryError',
    'InMemoryVectorStore',
    'InvalidInput',
    'MemoryItem',
    'MnemicGraphSearcher',
    'RememberResult',
    'RouteDecision',
    'SearchHit',
    'SignalClassifier',
    'Tier',
    'VectorStore',
    'WriteRouter',
    'normalize_text',
    'reciprocal_rank_fusion',
]
