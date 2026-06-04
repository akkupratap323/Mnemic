"""Tests for the learned graph-worthiness router."""

from __future__ import annotations

import pytest

from mnemic.hybrid.errors import InvalidInput
from mnemic.hybrid.learned_router import LearnedRouter
from mnemic.hybrid.types import Tier
from tests.hybrid.conftest import FakeEmbedder


def test_rejects_empty_weights():
    with pytest.raises(InvalidInput):
        LearnedRouter(weights=())


def test_rejects_bad_threshold():
    with pytest.raises(InvalidInput):
        LearnedRouter(weights=(1.0,), threshold=1.5)


def test_score_is_monotonic_in_dot_product():
    r = LearnedRouter(weights=(1.0, -1.0), bias=0.0)
    high = r.score_embedding((10.0, 0.0))
    low = r.score_embedding((0.0, 10.0))
    assert high > 0.9
    assert low < 0.1


def test_dim_mismatch_rejected():
    r = LearnedRouter(weights=(1.0, 1.0))
    with pytest.raises(InvalidInput):
        r.score_embedding((1.0, 2.0, 3.0))


def test_route_thresholds_on_score():
    r = LearnedRouter(weights=(1.0, -1.0), bias=0.0, threshold=0.5)
    assert r.route('x', embedding=(10.0, 0.0)).tier is Tier.VECTOR_AND_GRAPH
    assert r.route('x', embedding=(0.0, 10.0)).tier is Tier.VECTOR_ONLY


def test_route_requires_embedding():
    r = LearnedRouter(weights=(1.0,))
    with pytest.raises(InvalidInput):
        r.route('x')


def test_route_honors_metadata_overrides():
    r = LearnedRouter(weights=(1.0,), threshold=0.5)
    # force_vector_only wins even with a graph-leaning embedding
    d = r.route('x', {'force_vector_only': True}, embedding=(100.0,))
    assert d.tier is Tier.VECTOR_ONLY
    # force_graph wins even with a vector-leaning embedding
    d2 = r.route('x', {'force_graph': True}, embedding=(-100.0,))
    assert d2.tier is Tier.VECTOR_AND_GRAPH


def test_route_reason_includes_score():
    r = LearnedRouter(weights=(1.0,), bias=0.0)
    reasons = r.route('x', embedding=(2.0,)).reasons
    assert any(rsn.startswith('learned_score:') for rsn in reasons)


async def test_classify_embeds_then_routes():
    r = LearnedRouter(weights=tuple([1.0] * 16), bias=0.0, embedder=FakeEmbedder(dim=16))
    decision = await r.classify('some content here')
    assert decision.tier in (Tier.VECTOR_ONLY, Tier.VECTOR_AND_GRAPH)


async def test_classify_without_embedder_raises():
    r = LearnedRouter(weights=(1.0,))
    with pytest.raises(InvalidInput):
        await r.classify('x')
