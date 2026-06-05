"""Tests for the 3-way cost-vs-recall benchmark harness."""

from __future__ import annotations

import pytest

from mnemic.hybrid.benchmark import (
    AlwaysGraphRouter,
    AlwaysVectorRouter,
    CostModel,
    run_three_way,
)
from mnemic.hybrid.eval import EvalInstance, Turn
from mnemic.hybrid.fusion import GraphFact
from mnemic.hybrid.hashing_embedder import HashingEmbedder
from mnemic.hybrid.router import WriteRouter
from mnemic.hybrid.types import Tier

# ---- routers ----

def test_always_vector_router():
    assert AlwaysVectorRouter().route('x').tier is Tier.VECTOR_ONLY


def test_always_graph_router():
    assert AlwaysGraphRouter().route('x').tier is Tier.VECTOR_AND_GRAPH


# ---- cost model ----

def test_cost_model_scales_with_graph_writes():
    m = CostModel(graph_llm_calls_per_write=5)
    a = m.cost(total_writes=100, graph_writes=0)
    b = m.cost(total_writes=100, graph_writes=100)
    c = m.cost(total_writes=100, graph_writes=20)
    assert a.llm_calls == 0
    assert b.llm_calls == 500
    assert c.llm_calls == 100
    assert a.embed_calls == 100  # everything still gets embedded
    assert b.est_usd > c.est_usd > a.est_usd


# ---- fake graph tier (records writes, searches by token overlap) ----

class _FakeGraphTier:
    def __init__(self) -> None:
        self.facts: list[str] = []

    async def remember(self, **kwargs: object) -> object:
        self.facts.append(str(kwargs.get('episode_body', '')))
        return {}

    async def search(self, query: str, *, num_results: int = 10) -> list[GraphFact]:
        q = set(query.lower().split())
        out = [GraphFact(id=str(i), fact=f) for i, f in enumerate(self.facts) if q & set(f.lower().split())]
        return out[:num_results]


def _graph_factory():
    tier = _FakeGraphTier()
    return tier, tier


def _instances():
    return [
        EvalInstance(
            question_id='q1',
            question='what database did we choose',
            answer='postgres',
            question_type='single-session',
            sessions=(
                (
                    Turn('user', 'we choose postgres for the database', True),
                    Turn('user', 'lunch was great today', False),
                ),
            ),
        )
    ]


# ---- the benchmark ----

async def test_three_way_cost_ordering():
    report = await run_three_way(
        _instances(), embedder=HashingEmbedder(dim=64), router=WriteRouter()
    )
    a = report.by_name('A_vector_only')
    b = report.by_name('B_full_graph')
    a_cost, b_cost = a.cost.llm_calls, b.cost.llm_calls
    # A never touches the graph; B sends everything to it
    assert a.graph_writes == 0
    assert b.graph_writes == b.total_writes
    assert a_cost == 0
    assert b_cost > a_cost
    assert a.total_writes == b.total_writes  # same memories, different routing


async def test_three_way_graph_improves_recall_with_live_graph():
    report = await run_three_way(
        _instances(),
        embedder=HashingEmbedder(dim=64),
        router=WriteRouter(),
        graph_factory=_graph_factory,
        k=10,
    )
    b = report.by_name('B_full_graph')
    # full-graph stored the evidence and recalled it
    assert b.evidence_recall_at_k == 1.0
    assert b.by_type['single-session'] == 1.0


async def test_three_way_respects_limit():
    many = _instances() * 5
    report = await run_three_way(
        many, embedder=HashingEmbedder(dim=64), router=WriteRouter(), limit=2
    )
    # 2 instances * 2 turns each = 4 writes per config
    assert report.by_name('A_vector_only').total_writes == 4


async def test_by_name_unknown_raises():
    report = await run_three_way(
        _instances(), embedder=HashingEmbedder(dim=64), router=WriteRouter()
    )
    with pytest.raises(KeyError):
        report.by_name('nope')


def test_graph_fraction():
    from mnemic.hybrid.benchmark import ConfigResult, CostBreakdown

    r = ConfigResult('x', total_writes=10, graph_writes=2,
                     cost=CostBreakdown(10, 10, 0.0, 0.0), evidence_recall_at_k=0.0, by_type={})
    assert r.graph_fraction == 0.2
    empty = ConfigResult('y', 0, 0, CostBreakdown(0, 0, 0.0, 0.0), 0.0, {})
    assert empty.graph_fraction == 0.0
