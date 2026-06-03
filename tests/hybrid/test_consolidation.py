"""Tests for the consolidation worker (promote vector -> graph, prune stale)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from mnemic.hybrid.access import AccessLog
from mnemic.hybrid.consolidation import (
    ConsolidationPolicy,
    Consolidator,
)
from mnemic.hybrid.memory import HybridMemory
from mnemic.hybrid.types import MemoryItem
from mnemic.hybrid.vector_store import InMemoryVectorStore
from tests.hybrid.conftest import FakeEmbedder, RecordingGraph, make_counter_ids

NOW = datetime(2026, 6, 1, tzinfo=timezone.utc)


def item(id_: str, created_at: datetime) -> MemoryItem:
    return MemoryItem(id=id_, content=f'content {id_}', embedding=(1.0, 0.0), created_at=created_at)


# ---- AccessLog ----

def test_access_log_record_and_count():
    log = AccessLog()
    log.record('a')
    log.record('a', 2)
    assert log.count('a') == 3
    assert log.count('missing') == 0


def test_access_log_promotion_ledger():
    log = AccessLog()
    assert log.is_promoted('a') is False
    log.mark_promoted('a')
    assert log.is_promoted('a') is True


def test_access_log_rejects_negative():
    with pytest.raises(ValueError):
        AccessLog().record('a', -1)


def test_access_log_snapshot_is_a_copy():
    log = AccessLog()
    log.record('a', 2)
    snap = log.snapshot()
    assert snap == {'a': 2}
    snap['a'] = 99  # mutating the snapshot must not affect the log
    assert log.count('a') == 2


async def test_consolidator_uses_real_clock_when_not_injected():
    store = InMemoryVectorStore()
    cons = Consolidator(vector_store=store, graph=RecordingGraph(), access_log=AccessLog())
    report = await cons.run()  # default clock -> _utcnow()
    assert report.scanned == 0


# ---- Policy ----

def test_policy_promote_threshold():
    p = ConsolidationPolicy(promote_after_accesses=3)
    assert p.should_promote(access_count=3, already_promoted=False) is True
    assert p.should_promote(access_count=2, already_promoted=False) is False
    assert p.should_promote(access_count=9, already_promoted=True) is False


def test_policy_prune_rules():
    p = ConsolidationPolicy(prune_after_seconds=100, prune_max_accesses=0)
    assert p.should_prune(access_count=0, age_seconds=200, already_promoted=False) is True
    assert p.should_prune(access_count=1, age_seconds=200, already_promoted=False) is False
    assert p.should_prune(access_count=0, age_seconds=50, already_promoted=False) is False
    assert p.should_prune(access_count=0, age_seconds=999, already_promoted=True) is False


# ---- Consolidator ----

async def consolidator(store, graph, log, **policy_kw) -> Consolidator:
    return Consolidator(
        vector_store=store,
        graph=graph,
        access_log=log,
        policy=ConsolidationPolicy(**policy_kw),
        clock=lambda: NOW,
    )


async def test_promotes_frequently_accessed_item():
    store = InMemoryVectorStore()
    await store.add(item('a', NOW))
    log = AccessLog()
    log.record('a', 5)
    graph = RecordingGraph()
    report = await (await consolidator(store, graph, log, promote_after_accesses=3)).run()
    assert report.promoted == ('a',)
    assert len(graph.calls) == 1
    assert graph.calls[0]['episode_body'] == 'content a'
    assert log.is_promoted('a') is True


async def test_promotion_is_idempotent_across_runs():
    store = InMemoryVectorStore()
    await store.add(item('a', NOW))
    log = AccessLog()
    log.record('a', 5)
    graph = RecordingGraph()
    cons = await consolidator(store, graph, log, promote_after_accesses=3)
    await cons.run()
    report2 = await cons.run()
    assert report2.promoted == ()
    assert len(graph.calls) == 1  # not promoted twice


async def test_prunes_stale_unused_item():
    store = InMemoryVectorStore()
    old = NOW - timedelta(days=60)
    await store.add(item('a', old))
    log = AccessLog()
    graph = RecordingGraph()
    report = await (
        await consolidator(store, graph, log, prune_after_seconds=3600, prune_max_accesses=0)
    ).run()
    assert report.pruned == ('a',)
    assert await store.count() == 0


async def test_promoted_item_is_not_pruned():
    store = InMemoryVectorStore()
    old = NOW - timedelta(days=60)
    await store.add(item('a', old))
    log = AccessLog()
    log.mark_promoted('a')
    graph = RecordingGraph()
    report = await (
        await consolidator(store, graph, log, prune_after_seconds=3600)
    ).run()
    assert report.pruned == ()
    assert await store.count() == 1


async def test_graph_failure_during_promotion_is_recoverable():
    store = InMemoryVectorStore()
    await store.add(item('a', NOW))
    log = AccessLog()
    log.record('a', 5)
    graph = RecordingGraph(raise_on_call=True)
    report = await (await consolidator(store, graph, log, promote_after_accesses=3)).run()
    assert report.promoted == ()
    assert len(report.errors) == 1
    assert log.is_promoted('a') is False  # will retry next pass
    assert await store.count() == 1  # not lost


async def test_scanned_count():
    store = InMemoryVectorStore()
    for i in range(4):
        await store.add(item(f'i{i}', NOW))
    report = await (await consolidator(store, RecordingGraph(), AccessLog())).run()
    assert report.scanned == 4


# ---- HybridMemory access integration ----

async def test_recall_records_access_and_consolidator_promotes():
    store = InMemoryVectorStore()
    log = AccessLog()
    graph = RecordingGraph()
    mem = HybridMemory(
        embedder=FakeEmbedder(),
        vector_store=store,
        graph=graph,
        access_log=log,
        clock=lambda: NOW,
        id_factory=make_counter_ids(),
    )
    result = await mem.remember('how do i reset my password')  # chatter -> vector only
    assert result.graph_written is False
    for _ in range(3):
        await mem.recall('how do i reset my password', k=1)
    assert log.count(result.item.id) == 3

    report = await Consolidator(
        vector_store=store, graph=graph, access_log=log,
        policy=ConsolidationPolicy(promote_after_accesses=3), clock=lambda: NOW,
    ).run()
    assert result.item.id in report.promoted
    assert len(graph.calls) == 1  # the frequently-recalled chatter graduated to the graph
