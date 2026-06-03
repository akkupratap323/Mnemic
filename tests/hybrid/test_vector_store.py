"""Tests for the in-memory vector store."""

from __future__ import annotations

import asyncio

import pytest

from mnemic.hybrid.errors import DimensionMismatch, DuplicateItem, InvalidInput
from mnemic.hybrid.types import MemoryItem
from mnemic.hybrid.vector_store import InMemoryVectorStore, _dot, _norm


def test_dot_guards_against_dimension_mismatch():
    with pytest.raises(DimensionMismatch):
        _dot((1.0, 2.0), (1.0, 2.0, 3.0))


def test_norm_rejects_zero_vector():
    with pytest.raises(InvalidInput):
        _norm((0.0, 0.0))


def item(id_: str, emb, **meta) -> MemoryItem:
    return MemoryItem(id=id_, content=f'content {id_}', embedding=tuple(emb), metadata=meta)


async def test_add_and_count():
    s = InMemoryVectorStore()
    assert await s.count() == 0
    await s.add(item('a', [1, 0, 0]))
    await s.add(item('b', [0, 1, 0]))
    assert await s.count() == 2


async def test_duplicate_id_rejected():
    s = InMemoryVectorStore()
    await s.add(item('a', [1, 0, 0]))
    with pytest.raises(DuplicateItem):
        await s.add(item('a', [0, 1, 0]))


async def test_dimension_mismatch_on_add():
    s = InMemoryVectorStore()
    await s.add(item('a', [1, 0, 0]))
    with pytest.raises(DimensionMismatch):
        await s.add(item('b', [1, 0]))


async def test_dimension_mismatch_on_search():
    s = InMemoryVectorStore()
    await s.add(item('a', [1, 0, 0]))
    with pytest.raises(DimensionMismatch):
        await s.search([1, 0])


async def test_zero_vector_rejected_on_add():
    s = InMemoryVectorStore()
    with pytest.raises(InvalidInput):
        await s.add(item('a', [0, 0, 0]))


async def test_zero_query_rejected_on_search():
    s = InMemoryVectorStore()
    await s.add(item('a', [1, 0, 0]))
    with pytest.raises(InvalidInput):
        await s.search([0, 0, 0])


async def test_invalid_k_rejected():
    s = InMemoryVectorStore()
    await s.add(item('a', [1, 0, 0]))
    with pytest.raises(InvalidInput):
        await s.search([1, 0, 0], k=0)
    with pytest.raises(InvalidInput):
        await s.search([1, 0, 0], k=-1)


async def test_cosine_scores_are_correct():
    s = InMemoryVectorStore()
    await s.add(item('same', [1, 0, 0]))
    await s.add(item('orth', [0, 1, 0]))
    await s.add(item('opp', [-1, 0, 0]))
    by_id = {h.item.id: h.score for h in await s.search([1, 0, 0], k=10)}
    assert by_id['same'] == pytest.approx(1.0)
    assert by_id['orth'] == pytest.approx(0.0)
    assert by_id['opp'] == pytest.approx(-1.0)


async def test_results_sorted_by_score_then_id():
    s = InMemoryVectorStore()
    # 'a' and 'c' have identical embeddings -> identical scores -> id tiebreak
    await s.add(item('c', [1, 0, 0]))
    await s.add(item('a', [1, 0, 0]))
    await s.add(item('b', [0, 1, 0]))
    ordered = [h.item.id for h in await s.search([1, 0, 0], k=10)]
    assert ordered == ['a', 'c', 'b']  # equal top scores tie-break a<c, then b


async def test_k_limits_results():
    s = InMemoryVectorStore()
    for i in range(5):
        await s.add(item(f'i{i}', [1, i, 0]))
    assert len(await s.search([1, 0, 0], k=2)) == 2


async def test_k_larger_than_size_returns_all():
    s = InMemoryVectorStore()
    await s.add(item('a', [1, 0, 0]))
    assert len(await s.search([1, 0, 0], k=100)) == 1


async def test_where_filter():
    s = InMemoryVectorStore()
    await s.add(item('a', [1, 0, 0], kind='note'))
    await s.add(item('b', [1, 0, 0], kind='code'))
    hits = await s.search([1, 0, 0], k=10, where={'kind': 'code'})
    assert [h.item.id for h in hits] == ['b']


async def test_empty_store_search_returns_empty():
    s = InMemoryVectorStore()
    # no dim established yet -> any non-zero query is allowed and returns nothing
    assert await s.search([1, 2, 3], k=5) == []


async def test_get_and_delete():
    s = InMemoryVectorStore()
    await s.add(item('a', [1, 0, 0]))
    assert (await s.get('a')).id == 'a'
    assert await s.get('missing') is None
    assert await s.delete('a') is True
    assert await s.delete('a') is False
    assert await s.count() == 0


async def test_concurrent_adds_are_consistent():
    s = InMemoryVectorStore()
    await asyncio.gather(*(s.add(item(f'i{i}', [1, i, 1])) for i in range(20)))
    assert await s.count() == 20
