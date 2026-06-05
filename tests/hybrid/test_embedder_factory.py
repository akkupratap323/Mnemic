"""Tests for the embedder factory."""

from __future__ import annotations

import pytest

from mnemic.hybrid.embedder_factory import make_embedder
from mnemic.hybrid.errors import InvalidInput
from mnemic.hybrid.hashing_embedder import HashingEmbedder
from mnemic.hybrid.ollama_embedder import OllamaEmbedder
from mnemic.hybrid.st_embedder import SentenceTransformerEmbedder


def test_hashing_backend():
    assert isinstance(make_embedder('hashing'), HashingEmbedder)


def test_ollama_backend_defaults_to_bge_m3():
    e = make_embedder('ollama')
    assert isinstance(e, OllamaEmbedder)
    assert e.model == 'bge-m3'


def test_ollama_backend_custom_model():
    assert make_embedder('ollama', 'nomic-embed-text').model == 'nomic-embed-text'


def test_sentence_transformers_backend_defaults_to_bge_m3():
    e = make_embedder('sentence-transformers')
    assert isinstance(e, SentenceTransformerEmbedder)
    assert e.model_name == 'BAAI/bge-m3'


def test_unknown_backend_raises():
    with pytest.raises(InvalidInput):
        make_embedder('nope')
