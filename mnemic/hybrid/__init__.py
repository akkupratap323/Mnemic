"""
Copyright 2026, Abishek (Mnemic project).

Hybrid tiered-memory layer for Mnemic: a cheap, high-volume vector tier plus a
write router that promotes only signal to the premium temporal-graph tier.
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

from mnemic.hybrid.access import AccessLog
from mnemic.hybrid.agreement import (
    AgreementReport,
    GoldExample,
    compute_agreement,
    load_gold,
    measure_agreement,
)
from mnemic.hybrid.api_teacher import ApiTeacher
from mnemic.hybrid.cache import CachingEmbedder
from mnemic.hybrid.classifier import SignalClassifier
from mnemic.hybrid.config import HybridConfig
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
from mnemic.hybrid.eval import EvalInstance, EvalReport, evaluate, load_longmemeval
from mnemic.hybrid.factory import build_hybrid_memory
from mnemic.hybrid.fusion import (
    FusedHit,
    GraphFact,
    GraphSearcher,
    MnemicGraphSearcher,
    normalize_text,
    reciprocal_rank_fusion,
)
from mnemic.hybrid.hashing_embedder import HashingEmbedder
from mnemic.hybrid.learned_router import LearnedRouter
from mnemic.hybrid.mcp import register_memory_tools
from mnemic.hybrid.memory import GraphMemory, HybridMemory
from mnemic.hybrid.metrics import Metrics
from mnemic.hybrid.ollama_embedder import OllamaEmbedder
from mnemic.hybrid.ollama_teacher import OllamaTeacher
from mnemic.hybrid.router import Router, WriteRouter
from mnemic.hybrid.router_training import (
    HeuristicLabeler,
    LabeledExample,
    TeacherLabeler,
    add_failure_labels,
    build_training_set,
    evaluate_router,
    train_logistic,
)
from mnemic.hybrid.scheduler import ConsolidationScheduler
from mnemic.hybrid.schema import CODE_EDGE_TYPES, CODE_ENTITY_TYPES
from mnemic.hybrid.sqlite_store import SQLiteVectorStore
from mnemic.hybrid.st_embedder import SentenceTransformerEmbedder
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
    'AgreementReport',
    'ApiTeacher',
    'CachingEmbedder',
    'GoldExample',
    'compute_agreement',
    'load_gold',
    'measure_agreement',
    'ConsolidationPolicy',
    'ConsolidationReport',
    'ConsolidationScheduler',
    'Consolidator',
    'DimensionMismatch',
    'EvalInstance',
    'EvalReport',
    'HashingEmbedder',
    'HeuristicLabeler',
    'HybridConfig',
    'LabeledExample',
    'LearnedRouter',
    'Metrics',
    'OllamaEmbedder',
    'OllamaTeacher',
    'Router',
    'SentenceTransformerEmbedder',
    'TeacherLabeler',
    'add_failure_labels',
    'build_training_set',
    'evaluate_router',
    'train_logistic',
    'evaluate',
    'load_longmemeval',
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
    'register_memory_tools',
    'reciprocal_rank_fusion',
]
