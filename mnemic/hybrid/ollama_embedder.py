"""
Copyright 2026, Abishek (Mnemic project).

An embedder backed by a local Ollama embedding model (default BAAI/bge-m3, a
strong open-source 1024-dim multilingual model). Free, local, no API key.
Implements the hybrid Embedder protocol.

Setup:
    ollama pull bge-m3

Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from mnemic.hybrid.errors import InvalidInput


@dataclass
class OllamaEmbedder:
    """Embeds text via Ollama's embeddings API (default model: bge-m3).

    ``embed_fn`` is injectable so the logic is unit-testable without a server.
    """

    model: str = 'bge-m3'
    host: str = 'http://localhost:11434'
    timeout: float = 60.0
    embed_fn: Callable[[str], Awaitable[Sequence[float]]] | None = None

    async def embed(self, text: str) -> tuple[float, ...]:
        if not isinstance(text, str) or not text.strip():
            raise InvalidInput('text to embed must be a non-empty string')
        embed_fn = self.embed_fn or self._ollama_embed
        vector = await embed_fn(text)
        return tuple(float(x) for x in vector)

    async def _ollama_embed(self, text: str) -> Sequence[float]:  # pragma: no cover - network I/O
        import httpx

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f'{self.host}/api/embeddings',
                json={'model': self.model, 'prompt': text},
            )
            resp.raise_for_status()
            return resp.json()['embedding']
