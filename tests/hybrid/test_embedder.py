"""Tests for the embedder adapter."""

from __future__ import annotations

import pytest

from mnemic.hybrid.embedder import EmbedderClientAdapter
from mnemic.hybrid.errors import InvalidInput
from tests.hybrid.conftest import FakeEmbedderClient


async def test_adapter_returns_tuple_of_floats():
    adapter = EmbedderClientAdapter(FakeEmbedderClient([1, 2, 3]))
    out = await adapter.embed('hello')
    assert out == (1.0, 2.0, 3.0)
    assert all(isinstance(x, float) for x in out)


async def test_adapter_forwards_text_to_client():
    client = FakeEmbedderClient()
    adapter = EmbedderClientAdapter(client)
    await adapter.embed('payload')
    assert client.calls == ['payload']


@pytest.mark.parametrize('bad', ['', '   ', '\n\t'])
async def test_empty_text_rejected(bad):
    adapter = EmbedderClientAdapter(FakeEmbedderClient())
    with pytest.raises(InvalidInput):
        await adapter.embed(bad)
