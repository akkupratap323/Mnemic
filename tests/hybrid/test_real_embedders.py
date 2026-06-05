"""Tests for the Ollama and sentence-transformers (bge-m3) embedders."""

from __future__ import annotations

import pytest

from mnemic.hybrid.errors import InvalidInput
from mnemic.hybrid.ollama_embedder import OllamaEmbedder
from mnemic.hybrid.st_embedder import SentenceTransformerEmbedder

# ---- OllamaEmbedder ----

def test_ollama_embedder_defaults_to_bge_m3():
    assert OllamaEmbedder().model == 'bge-m3'


async def test_ollama_embedder_returns_float_tuple():
    async def fake(_text: str):
        return [1, 2, 3]

    emb = OllamaEmbedder(embed_fn=fake)
    out = await emb.embed('hello')
    assert out == (1.0, 2.0, 3.0)
    assert all(isinstance(x, float) for x in out)


async def test_ollama_embedder_forwards_text():
    seen = {}

    async def fake(text: str):
        seen['text'] = text
        return [0.5]

    await OllamaEmbedder(embed_fn=fake).embed('payload here')
    assert seen['text'] == 'payload here'


@pytest.mark.parametrize('bad', ['', '   ', None])
async def test_ollama_embedder_rejects_empty(bad):
    async def fake(_text: str):
        return [1.0]

    with pytest.raises(InvalidInput):
        await OllamaEmbedder(embed_fn=fake).embed(bad)  # type: ignore[arg-type]


# ---- SentenceTransformerEmbedder ----

def test_st_embedder_defaults_to_bge_m3():
    assert SentenceTransformerEmbedder().model_name == 'BAAI/bge-m3'


async def test_st_embedder_returns_float_tuple():
    emb = SentenceTransformerEmbedder(encode=lambda _t: [0.1, 0.2, 0.3])
    out = await emb.embed('hello')
    assert out == pytest.approx((0.1, 0.2, 0.3))


@pytest.mark.parametrize('bad', ['', '   '])
async def test_st_embedder_rejects_empty(bad):
    emb = SentenceTransformerEmbedder(encode=lambda _t: [1.0])
    with pytest.raises(InvalidInput):
        await emb.embed(bad)


async def test_st_embedder_uses_thread_path_when_no_encode_injected():
    # encode=None -> goes through asyncio.to_thread(self._encode_sync, ...);
    # subclass overrides _encode_sync so no model weights are loaded
    class _Sub(SentenceTransformerEmbedder):
        def _encode_sync(self, text: str):
            return [9.0, 8.0]

    out = await _Sub().embed('via thread')
    assert out == (9.0, 8.0)
