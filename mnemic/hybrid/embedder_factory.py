"""
Copyright 2026, Abishek (Mnemic project).

Build an Embedder by name. Lets scripts and the factory pick a backend at runtime:
offline baseline, free local bge-m3 (Ollama or sentence-transformers), or OpenAI.
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

from mnemic.hybrid.embedder import Embedder
from mnemic.hybrid.errors import InvalidInput
from mnemic.hybrid.hashing_embedder import HashingEmbedder
from mnemic.hybrid.ollama_embedder import OllamaEmbedder
from mnemic.hybrid.st_embedder import SentenceTransformerEmbedder

EMBEDDER_BACKENDS = ('hashing', 'ollama', 'sentence-transformers', 'openai')


def make_embedder(backend: str, model: str | None = None) -> Embedder:
    """Construct an Embedder for the given backend.

    - hashing: offline bag-of-words baseline (no deps)
    - ollama: bge-m3 (default) via a local Ollama server
    - sentence-transformers: BAAI/bge-m3 (default) loaded in-process
    - openai: text-embedding-3-small (default) via the OpenAI API
    """
    if backend == 'hashing':
        return HashingEmbedder()
    if backend == 'ollama':
        return OllamaEmbedder(model=model or 'bge-m3')
    if backend == 'sentence-transformers':
        return SentenceTransformerEmbedder(model_name=model or 'BAAI/bge-m3')
    if backend == 'openai':  # pragma: no cover - provider glue, needs SDK + key
        from mnemic.embedder import OpenAIEmbedder
        from mnemic.hybrid.embedder import EmbedderClientAdapter

        return EmbedderClientAdapter(OpenAIEmbedder())
    raise InvalidInput(f'unknown embedder backend: {backend!r} (choose from {EMBEDDER_BACKENDS})')
