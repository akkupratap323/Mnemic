"""Tests for the HybridMemory facade (router + vector + graph wiring)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from mnemic.hybrid.errors import InvalidInput
from mnemic.hybrid.memory import HybridMemory
from mnemic.hybrid.types import Tier
from mnemic.hybrid.vector_store import InMemoryVectorStore
from tests.hybrid.conftest import FakeEmbedder, RecordingGraph, make_counter_ids

FIXED = datetime(2026, 1, 1, tzinfo=timezone.utc)


def build(graph: RecordingGraph | None = None) -> HybridMemory:
    return HybridMemory(
        embedder=FakeEmbedder(),
        vector_store=InMemoryVectorStore(),
        graph=graph,
        clock=lambda: FIXED,
        id_factory=make_counter_ids(),
    )


async def test_noise_goes_to_vector_only_not_graph():
    graph = RecordingGraph()
    mem = build(graph)
    result = await mem.remember('had lunch at noon')
    assert result.decision.tier is Tier.VECTOR_ONLY
    assert result.graph_written is False
    assert graph.calls == []
    assert await mem.vector_store.count() == 1


async def test_signal_writes_both_tiers():
    graph = RecordingGraph()
    mem = build(graph)
    result = await mem.remember('decision: adopt FalkorDB for the graph tier')
    assert result.decision.writes_graph is True
    assert result.graph_written is True
    assert len(graph.calls) == 1
    call = graph.calls[0]
    assert call['episode_body'] == 'decision: adopt FalkorDB for the graph tier'
    assert call['reference_time'] == FIXED
    assert call['name'].startswith('memory-')


async def test_signal_with_no_graph_configured_is_safe():
    mem = build(graph=None)
    result = await mem.remember('important decision made')
    assert result.decision.writes_graph is True
    assert result.graph_written is False
    assert result.graph_error is None
    assert await mem.vector_store.count() == 1


async def test_graph_failure_never_loses_vector_write():
    graph = RecordingGraph(raise_on_call=True)
    mem = build(graph)
    result = await mem.remember('bug: null pointer on login')
    assert result.graph_written is False
    assert result.graph_error is not None
    assert result.graph_error.startswith('RuntimeError')
    # the vector memory is still durably stored
    assert await mem.vector_store.count() == 1
    assert await mem.vector_store.get(result.item.id) is not None


async def test_deterministic_id_and_clock():
    mem = build(RecordingGraph())
    r1 = await mem.remember('first note')
    r2 = await mem.remember('second note')
    assert r1.item.id == 'id-1'
    assert r2.item.id == 'id-2'
    assert r1.item.created_at == FIXED


async def test_stored_metadata_is_immutable_snapshot():
    mem = build(RecordingGraph())
    md = {'kind': 'note'}
    result = await mem.remember('a thing', metadata=md)
    md['kind'] = 'mutated'  # mutate caller's dict afterwards
    assert result.item.metadata['kind'] == 'note'
    with pytest.raises(TypeError):
        result.item.metadata['kind'] = 'x'  # type: ignore[index]


@pytest.mark.parametrize('bad', ['', '   ', None])
async def test_remember_rejects_empty_content(bad):
    mem = build(RecordingGraph())
    with pytest.raises(InvalidInput):
        await mem.remember(bad)  # type: ignore[arg-type]


@pytest.mark.parametrize('bad', ['', '   ', None])
async def test_recall_rejects_empty_query(bad):
    mem = build(RecordingGraph())
    with pytest.raises(InvalidInput):
        await mem.recall(bad)  # type: ignore[arg-type]


async def test_recall_exact_match_ranks_first():
    mem = build(RecordingGraph())
    await mem.remember('alpha beta gamma')
    await mem.remember('delta epsilon zeta')
    await mem.remember('lambda mu nu')
    hits = await mem.recall('alpha beta gamma', k=3)
    # identical text -> cosine 1.0 -> guaranteed top regardless of hash collisions
    assert hits[0].item.content == 'alpha beta gamma'
    assert hits[0].score == pytest.approx(1.0)


async def test_default_id_and_clock_factories_used_when_not_injected():
    # no clock / id_factory injected -> real uuid + tz-aware utc timestamp
    mem = HybridMemory(embedder=FakeEmbedder(), vector_store=InMemoryVectorStore())
    result = await mem.remember('a note')
    assert len(result.item.id) == 32  # uuid4().hex
    assert result.item.created_at is not None
    assert result.item.created_at.tzinfo is not None


async def test_group_id_passed_through_to_graph():
    graph = RecordingGraph()
    mem = build(graph)
    await mem.remember('decision: use group', group_id='team-42')
    assert graph.calls[0]['group_id'] == 'team-42'


async def test_recall_where_filter_passthrough():
    mem = build(RecordingGraph())
    await mem.remember('shared words here', metadata={'kind': 'a'})
    await mem.remember('shared words here', metadata={'kind': 'b'})
    hits = await mem.recall('shared words here', k=10, where={'kind': 'b'})
    assert [h.item.metadata['kind'] for h in hits] == ['b']
