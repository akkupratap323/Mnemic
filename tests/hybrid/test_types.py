"""Tests for the immutable value types."""

from __future__ import annotations

import pytest

from mnemic.hybrid.errors import InvalidInput
from mnemic.hybrid.types import MemoryItem, RouteDecision, Tier


def test_empty_id_rejected():
    with pytest.raises(InvalidInput):
        MemoryItem(id='', content='c', embedding=(1.0,))


def test_empty_content_rejected():
    with pytest.raises(InvalidInput):
        MemoryItem(id='a', content='   ', embedding=(1.0,))


def test_empty_embedding_rejected():
    with pytest.raises(InvalidInput):
        MemoryItem(id='a', content='c', embedding=())


def test_embedding_coerced_to_float_tuple():
    item = MemoryItem(id='a', content='c', embedding=[1, 2, 3])
    assert item.embedding == (1.0, 2.0, 3.0)
    assert all(isinstance(x, float) for x in item.embedding)


def test_default_metadata_is_empty_mapping():
    item = MemoryItem(id='a', content='c', embedding=(1.0,))
    assert dict(item.metadata) == {}


def test_route_decision_writes_graph_property():
    assert RouteDecision(Tier.VECTOR_AND_GRAPH).writes_graph is True
    assert RouteDecision(Tier.VECTOR_ONLY).writes_graph is False
