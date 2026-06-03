"""Tests for the persistent SQLite vector store."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from mnemic.hybrid.errors import DimensionMismatch, DuplicateItem, InvalidInput
from mnemic.hybrid.sqlite_store import SQLiteVectorStore
from mnemic.hybrid.types import MemoryItem

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def item(id_, emb, **meta) -> MemoryItem:
    return MemoryItem(id=id_, content=f'content {id_}', embedding=tuple(emb),
                      metadata=meta, created_at=NOW)


@pytest.fixture
def store():
    s = SQLiteVectorStore(':memory:')
    yield s
    s.close()


async def test_add_count_get(store):
    await store.add(item('a', [1, 0, 0]))
    assert await store.count() == 1
    got = await store.get('a')
    assert got.id == 'a'
    assert got.embedding == (1.0, 0.0, 0.0)
    assert got.created_at == NOW
    assert await store.get('missing') is None


async def test_duplicate_rejected(store):
    await store.add(item('a', [1, 0, 0]))
    with pytest.raises(DuplicateItem):
        await store.add(item('a', [0, 1, 0]))


async def test_dimension_enforced(store):
    await store.add(item('a', [1, 0, 0]))
    with pytest.raises(DimensionMismatch):
        await store.add(item('b', [1, 0]))
    with pytest.raises(DimensionMismatch):
        await store.search([1, 0])


async def test_zero_vector_rejected(store):
    with pytest.raises(InvalidInput):
        await store.add(item('a', [0, 0, 0]))


async def test_invalid_k(store):
    await store.add(item('a', [1, 0, 0]))
    with pytest.raises(InvalidInput):
        await store.search([1, 0, 0], k=0)


async def test_cosine_ranking_and_tiebreak(store):
    await store.add(item('c', [1, 0, 0]))
    await store.add(item('a', [1, 0, 0]))
    await store.add(item('b', [0, 1, 0]))
    ordered = [h.item.id for h in await store.search([1, 0, 0], k=10)]
    assert ordered == ['a', 'c', 'b']
    assert (await store.search([1, 0, 0], k=10))[0].score == pytest.approx(1.0)


async def test_where_filter(store):
    await store.add(item('a', [1, 0, 0], kind='note'))
    await store.add(item('b', [1, 0, 0], kind='code'))
    hits = await store.search([1, 0, 0], k=10, where={'kind': 'code'})
    assert [h.item.id for h in hits] == ['b']


async def test_delete_and_all_items(store):
    await store.add(item('a', [1, 0, 0]))
    await store.add(item('b', [0, 1, 0]))
    assert len(await store.all_items()) == 2
    assert await store.delete('a') is True
    assert await store.delete('a') is False
    assert await store.count() == 1


async def test_non_json_metadata_rejected(store):
    bad = MemoryItem(id='a', content='c', embedding=(1.0,), metadata={'x': {1, 2, 3}})
    with pytest.raises(InvalidInput):
        await store.add(bad)


async def test_persistence_across_reopen(tmp_path):
    db = str(tmp_path / 'mem.db')
    s1 = SQLiteVectorStore(db)
    await s1.add(item('a', [1, 0, 0], kind='note'))
    s1.close()

    s2 = SQLiteVectorStore(db)  # reopen same file
    assert await s2.count() == 1
    got = await s2.get('a')
    assert got.content == 'content a'
    assert got.metadata['kind'] == 'note'
    # dimension persisted -> mismatch still enforced after restart
    with pytest.raises(DimensionMismatch):
        await s2.add(item('b', [1, 0]))
    s2.close()
