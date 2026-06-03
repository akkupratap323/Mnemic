"""Tests for the content-hash embedding cache."""

from __future__ import annotations

import pytest

from mnemic.hybrid.cache import CachingEmbedder
from mnemic.hybrid.errors import InvalidInput
from tests.hybrid.conftest import FakeEmbedder


async def test_identical_text_embedded_once():
    inner = FakeEmbedder()
    cache = CachingEmbedder(inner)
    a = await cache.embed('same text')
    b = await cache.embed('same text')
    assert a == b
    assert inner.calls == ['same text']  # inner invoked only once
    assert cache.hits == 1
    assert cache.misses == 1


async def test_different_text_is_a_miss():
    inner = FakeEmbedder()
    cache = CachingEmbedder(inner)
    await cache.embed('one')
    await cache.embed('two')
    assert cache.misses == 2
    assert cache.hits == 0
    assert inner.calls == ['one', 'two']


async def test_lru_eviction_respects_maxsize():
    inner = FakeEmbedder()
    cache = CachingEmbedder(inner, maxsize=2)
    await cache.embed('a')
    await cache.embed('b')
    await cache.embed('c')  # evicts 'a'
    assert cache.size == 2
    await cache.embed('a')  # 'a' was evicted -> miss again
    assert inner.calls.count('a') == 2


async def test_recently_used_survives_eviction():
    inner = FakeEmbedder()
    cache = CachingEmbedder(inner, maxsize=2)
    await cache.embed('a')
    await cache.embed('b')
    await cache.embed('a')  # touch 'a' -> now MRU
    await cache.embed('c')  # evicts 'b' (LRU), not 'a'
    await cache.embed('a')  # still cached -> hit
    assert cache.hits == 2


def test_invalid_maxsize_rejected():
    with pytest.raises(InvalidInput):
        CachingEmbedder(FakeEmbedder(), maxsize=0)


async def test_empty_text_propagates_inner_error():
    cache = CachingEmbedder(FakeEmbedder())
    with pytest.raises(InvalidInput):
        await cache.embed('')
