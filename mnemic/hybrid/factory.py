"""
Copyright 2026, Abishek (Mnemic project).

One-call wiring of the hybrid memory stack on top of a real Mnemic client:
caching embedder -> persistent vector tier -> write router -> graph tier.
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

from typing import cast

from mnemic.hybrid.access import AccessLog
from mnemic.hybrid.cache import CachingEmbedder
from mnemic.hybrid.embedder import EmbedderClientAdapter
from mnemic.hybrid.fusion import MnemicGraphSearcher
from mnemic.hybrid.memory import GraphMemory, HybridMemory
from mnemic.hybrid.router import WriteRouter
from mnemic.hybrid.sqlite_store import SQLiteVectorStore
from mnemic.hybrid.vector_store import VectorStore


def build_hybrid_memory(
    *,
    mnemic_client: object,
    embedder_client: object,
    vector_store: VectorStore | None = None,
    db_path: str = 'mnemic_memory.db',
    router: WriteRouter | None = None,
    cache_maxsize: int = 10_000,
    enable_access_tracking: bool = True,
) -> HybridMemory:
    """Assemble a production HybridMemory around an existing Mnemic client.

    Parameters
    ----------
    mnemic_client:
        A Mnemic instance (provides async ``remember`` and ``search``).
    embedder_client:
        A Mnemic EmbedderClient (provides async ``create``).
    vector_store:
        Override the default persistent SQLite store (e.g. an in-memory one).
    db_path:
        Path for the default SQLite store when ``vector_store`` is not given.
    """
    embedder = CachingEmbedder(EmbedderClientAdapter(embedder_client), maxsize=cache_maxsize)
    store = vector_store if vector_store is not None else SQLiteVectorStore(db_path)
    return HybridMemory(
        embedder=embedder,
        vector_store=store,
        router=router or WriteRouter(),
        graph=cast(GraphMemory, mnemic_client),  # Mnemic.remember matches the protocol
        graph_searcher=MnemicGraphSearcher(mnemic_client),
        access_log=AccessLog() if enable_access_tracking else None,
    )
