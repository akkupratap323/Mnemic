"""
Copyright 2026, Abishek (Mnemic project).

An embedder backed by sentence-transformers (default BAAI/bge-m3), loaded
in-process at full precision. Heavier than the Ollama path but no server needed.
Requires the optional dependency: pip install mnemic[sentence-transformers]
Implements the hybrid Embedder protocol.
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from mnemic.hybrid.errors import InvalidInput


@dataclass
class SentenceTransformerEmbedder:
    """Embeds text with a local sentence-transformers model (default bge-m3).

    ``encode`` is injectable so the logic is unit-testable without loading weights.
    The real model is loaded lazily and inference runs off the event loop.
    """

    model_name: str = 'BAAI/bge-m3'
    normalize: bool = True
    encode: Callable[[str], Sequence[float]] | None = None
    _model: object = field(default=None, init=False, repr=False)

    async def embed(self, text: str) -> tuple[float, ...]:
        if not isinstance(text, str) or not text.strip():
            raise InvalidInput('text to embed must be a non-empty string')
        if self.encode is not None:
            vector = self.encode(text)
        else:
            vector = await asyncio.to_thread(self._encode_sync, text)
        return tuple(float(x) for x in vector)

    def _encode_sync(self, text: str) -> Sequence[float]:  # pragma: no cover - loads weights
        from sentence_transformers import SentenceTransformer

        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        encoded = self._model.encode(text, normalize_embeddings=self.normalize)  # type: ignore[attr-defined]
        return list(encoded.tolist())
