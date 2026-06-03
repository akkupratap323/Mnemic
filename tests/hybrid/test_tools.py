"""Tests for the MCP-friendly MemoryTools facade."""

from __future__ import annotations

from datetime import datetime, timezone

from mnemic.hybrid.memory import HybridMemory
from mnemic.hybrid.tools import MemoryTools
from mnemic.hybrid.vector_store import InMemoryVectorStore
from tests.hybrid.conftest import FakeEmbedder, RecordingGraph, make_counter_ids

FIXED = datetime(2026, 1, 1, tzinfo=timezone.utc)


def build_tools(graph=None) -> MemoryTools:
    mem = HybridMemory(
        embedder=FakeEmbedder(),
        vector_store=InMemoryVectorStore(),
        graph=graph,
        clock=lambda: FIXED,
        id_factory=make_counter_ids(),
    )
    return MemoryTools(memory=mem)


async def test_remember_returns_serializable_dict():
    tools = build_tools(RecordingGraph())
    out = await tools.remember('decision: pick FalkorDB')
    assert out['id'] == 'id-1'
    assert out['tier'] == 'vector_and_graph'
    assert out['graph_written'] is True
    assert isinstance(out['reasons'], list)


async def test_remember_noise_is_vector_only():
    tools = build_tools(RecordingGraph())
    out = await tools.remember('just saying hi')
    assert out['tier'] == 'vector_only'
    assert out['graph_written'] is False


async def test_recall_returns_fused_dicts():
    tools = build_tools()
    await tools.remember('alpha beta gamma')
    results = await tools.recall('alpha beta gamma', k=5)
    assert results
    assert set(results[0]) == {'content', 'score', 'sources'}
    assert results[0]['content'] == 'alpha beta gamma'


async def test_recall_code_filters_by_entity():
    tools = build_tools()
    await tools.remember('login function in auth.py', metadata={'entity': 'Function'})
    await tools.remember('users table migration', metadata={'entity': 'File'})
    results = await tools.recall_code('anything', entity='Function', k=10)
    assert len(results) == 1
    assert results[0]['metadata']['entity'] == 'Function'


async def test_forget_deletes():
    tools = build_tools()
    out = await tools.remember('temporary note')
    forgotten = await tools.forget(str(out['id']))
    assert forgotten == {'id': out['id'], 'deleted': True}
    assert (await tools.forget('missing'))['deleted'] is False
