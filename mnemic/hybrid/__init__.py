"""
Copyright 2026, Abishek (Mnemic project).

Hybrid tiered-memory layer for Mnemic: a cheap, high-volume vector tier plus a
write router that promotes only signal to the premium temporal-graph tier.
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

from mnemic.hybrid.access import AccessLog
from mnemic.hybrid.cache import CachingEmbedder
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
from mnemic.hybrid.factory import build_hybrid_memory
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
from mnemic.hybrid.schema import CODE_EDGE_TYPES, CODE_ENTITY_TYPES
from mnemic.hybrid.sqlite_store import SQLiteVectorStore
from mnemic.hybrid.tools import MemoryTools
from mnemic.hybrid.types import (
    MemoryItem,
    RememberResult,
    RouteDecision,
    SearchHit,
    Tier,
)
from mnemic.hybrid.vector_store import InMemoryVectorStore, VectorStore

__all__ = [
    'CODE_EDGE_TYPES',
    'CODE_ENTITY_TYPES',
    'AccessLog',
    'CachingEmbedder',
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
    'MemoryTools',
    'MnemicGraphSearcher',
    'RememberResult',
    'RouteDecision',
    'SQLiteVectorStore',
    'SearchHit',
    'SignalClassifier',
    'Tier',
    'VectorStore',
    'WriteRouter',
    'build_hybrid_memory',
    'normalize_text',
    'reciprocal_rank_fusion',
]
