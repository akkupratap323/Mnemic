"""Tests for the build_hybrid_memory factory (end-to-end wiring)."""

from __future__ import annotations

from mnemic.hybrid.cache import CachingEmbedder
from mnemic.hybrid.factory import build_hybrid_memory
from mnemic.hybrid.fusion import MnemicGraphSearcher
from mnemic.hybrid.sqlite_store import SQLiteVectorStore
from mnemic.hybrid.vector_store import InMemoryVectorStore
from tests.hybrid.conftest import FakeEmbedderClient, FakeMnemicClient


def test_factory_wires_components():
    client = FakeMnemicClient()
    mem = build_hybrid_memory(
        mnemic_client=client,
        embedder_client=FakeEmbedderClient([1.0, 0.0, 0.0]),
        vector_store=InMemoryVectorStore(),
    )
    assert isinstance(mem.embedder, CachingEmbedder)
    assert mem.graph is client
    assert isinstance(mem.graph_searcher, MnemicGraphSearcher)
    assert mem.access_log is not None


def test_factory_defaults_to_sqlite_store():
    mem = build_hybrid_memory(
        mnemic_client=FakeMnemicClient(),
        embedder_client=FakeEmbedderClient(),
        db_path=':memory:',
    )
    assert isinstance(mem.vector_store, SQLiteVectorStore)


def test_factory_can_disable_access_tracking():
    mem = build_hybrid_memory(
        mnemic_client=FakeMnemicClient(),
        embedder_client=FakeEmbedderClient(),
        vector_store=InMemoryVectorStore(),
        enable_access_tracking=False,
    )
    assert mem.access_log is None


async def test_end_to_end_remember_routes_to_both_tiers():
    client = FakeMnemicClient(facts=[('g1', 'decision: adopt sqlite tier')])
    mem = build_hybrid_memory(
        mnemic_client=client,
        embedder_client=FakeEmbedderClient([1.0, 2.0, 3.0]),
        vector_store=InMemoryVectorStore(),
    )
    result = await mem.remember('decision: adopt sqlite tier for persistence')
    # vector tier stored it
    assert await mem.vector_store.count() == 1
    # graph tier (the real client) was called because it's signal
    assert result.graph_written is True
    assert len(client.remember_calls) == 1


async def test_end_to_end_recall_fused_blends_client_facts():
    client = FakeMnemicClient(facts=[('g1', 'shared content')])
    mem = build_hybrid_memory(
        mnemic_client=client,
        embedder_client=FakeEmbedderClient([1.0, 2.0, 3.0]),
        vector_store=InMemoryVectorStore(),
    )
    await mem.remember('shared content')
    hits = await mem.recall_fused('shared content', k=5)
    top = hits[0]
    assert top.content == 'shared content'
    assert top.sources == ('graph', 'vector')
