"""Tests for Metrics and HybridMemory metric integration."""

from __future__ import annotations

from datetime import datetime, timezone

from mnemic.hybrid.memory import HybridMemory
from mnemic.hybrid.metrics import Metrics
from mnemic.hybrid.vector_store import InMemoryVectorStore
from tests.hybrid.conftest import FakeEmbedder, RecordingGraph, make_counter_ids

FIXED = datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_record_remember_vector_only():
    m = Metrics()
    m.record_remember(graph_written=False)
    assert m.remembers == 1
    assert m.vector_writes == 1
    assert m.graph_writes == 0


def test_record_remember_with_graph():
    m = Metrics()
    m.record_remember(graph_written=True)
    assert m.graph_writes == 1


def test_graph_write_ratio():
    m = Metrics()
    assert m.graph_write_ratio == 0.0
    m.record_remember(graph_written=True)
    m.record_remember(graph_written=False)
    assert m.graph_write_ratio == 0.5


def test_snapshot_keys():
    m = Metrics()
    snap = m.snapshot()
    assert set(snap) == {
        'remembers', 'vector_writes', 'graph_writes', 'recalls',
        'fused_recalls', 'graph_write_ratio',
    }


async def test_hybrid_memory_records_metrics():
    metrics = Metrics()
    mem = HybridMemory(
        embedder=FakeEmbedder(),
        vector_store=InMemoryVectorStore(),
        graph=RecordingGraph(),
        metrics=metrics,
        clock=lambda: FIXED,
        id_factory=make_counter_ids(),
    )
    await mem.remember('plain chatter')          # vector only
    await mem.remember('decision: ship it')      # vector + graph
    await mem.recall('chatter', k=5)
    await mem.recall_fused('chatter', k=5)

    assert metrics.remembers == 2
    assert metrics.vector_writes == 2
    assert metrics.graph_writes == 1
    # recall_fused internally performs a vector recall, so recalls counts both
    assert metrics.recalls == 2
    assert metrics.fused_recalls == 1
