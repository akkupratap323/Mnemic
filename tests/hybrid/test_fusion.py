"""Tests for read-side fusion (RRF) and recall_fused."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from mnemic.hybrid.errors import InvalidInput
from mnemic.hybrid.fusion import (
    GraphFact,
    MnemicGraphSearcher,
    normalize_text,
    reciprocal_rank_fusion,
)
from mnemic.hybrid.memory import HybridMemory
from mnemic.hybrid.vector_store import InMemoryVectorStore
from tests.hybrid.conftest import FakeEmbedder, FakeGraphSearcher, make_counter_ids

FIXED = datetime(2026, 1, 1, tzinfo=timezone.utc)


# ---- RRF unit tests ----

def test_rrf_item_in_both_lists_outranks_single_list_item():
    scores = reciprocal_rank_fusion([['a', 'b'], ['a', 'c']])
    assert scores['a'] > scores['b']
    assert scores['a'] > scores['c']


def test_rrf_respects_rank_order():
    scores = reciprocal_rank_fusion([['a', 'b', 'c']])
    assert scores['a'] > scores['b'] > scores['c']


def test_rrf_empty_input():
    assert reciprocal_rank_fusion([]) == {}
    assert reciprocal_rank_fusion([[], []]) == {}


def test_rrf_invalid_k():
    with pytest.raises(InvalidInput):
        reciprocal_rank_fusion([['a']], k=0)


def test_normalize_text():
    assert normalize_text('  Hello   WORLD ') == 'hello world'


# ---- recall_fused integration ----

def build(searcher=None) -> HybridMemory:
    return HybridMemory(
        embedder=FakeEmbedder(),
        vector_store=InMemoryVectorStore(),
        graph_searcher=searcher,
        clock=lambda: FIXED,
        id_factory=make_counter_ids(),
    )


async def test_recall_fused_vector_only_when_no_searcher():
    mem = build(searcher=None)
    await mem.remember('alpha beta gamma')
    hits = await mem.recall_fused('alpha beta gamma', k=5)
    assert hits
    assert hits[0].sources == ('vector',)


async def test_recall_fused_merges_shared_content():
    searcher = FakeGraphSearcher(facts=[GraphFact(id='g1', fact='alpha beta gamma')])
    mem = build(searcher)
    await mem.remember('alpha beta gamma')
    await mem.remember('unrelated chatter here')
    hits = await mem.recall_fused('alpha beta gamma', k=5)
    top = hits[0]
    assert top.content == 'alpha beta gamma'
    assert top.sources == ('graph', 'vector')  # found by both -> boosted


async def test_recall_fused_includes_graph_only_facts():
    searcher = FakeGraphSearcher(facts=[GraphFact(id='g1', fact='only in graph')])
    mem = build(searcher)
    await mem.remember('alpha beta')
    contents = {h.content for h in await mem.recall_fused('alpha beta', k=10)}
    assert 'only in graph' in contents


async def test_recall_fused_k_limits_results():
    searcher = FakeGraphSearcher(
        facts=[GraphFact(id=f'g{i}', fact=f'fact {i}') for i in range(5)]
    )
    mem = build(searcher)
    for i in range(5):
        await mem.remember(f'vector item {i}')
    assert len(await mem.recall_fused('item', k=3)) == 3


async def test_recall_fused_rejects_empty_query():
    mem = build()
    with pytest.raises(InvalidInput):
        await mem.recall_fused('   ')


async def test_recall_fused_rejects_invalid_k():
    mem = build()
    with pytest.raises(InvalidInput):
        await mem.recall_fused('x', k=0)


# ---- MnemicGraphSearcher adapter ----

class _Edge:
    def __init__(self, uuid, fact):
        self.uuid = uuid
        self.fact = fact


class _FakeClient:
    def __init__(self, edges):
        self.edges = edges
        self.calls = []

    async def search(self, query, *, num_results=10):
        self.calls.append((query, num_results))
        return self.edges[:num_results]


async def test_mnemic_graph_searcher_adapts_edges():
    client = _FakeClient([_Edge('u1', 'fact one'), _Edge('u2', 'fact two')])
    searcher = MnemicGraphSearcher(client)
    facts = await searcher.search('q', num_results=2)
    assert [f.id for f in facts] == ['u1', 'u2']
    assert [f.fact for f in facts] == ['fact one', 'fact two']
    assert client.calls == [('q', 2)]


async def test_mnemic_graph_searcher_rejects_empty_query():
    searcher = MnemicGraphSearcher(_FakeClient([]))
    with pytest.raises(InvalidInput):
        await searcher.search('')
