"""
Copyright 2026, Abishek (Mnemic project).

Content-hash embedding cache: identical text is embedded once. Critical at high
volume and for coding agents where the same file/snippet recurs constantly.
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict

from mnemic.hybrid.embedder import Embedder
from mnemic.hybrid.errors import InvalidInput


class CachingEmbedder:
    """Wraps an Embedder, caching results by SHA-256 of the input text (LRU)."""

    def __init__(self, inner: Embedder, *, maxsize: int = 10_000) -> None:
        if maxsize <= 0:
            raise InvalidInput('maxsize must be a positive integer')
        self._inner = inner
        self._maxsize = maxsize
        self._cache: OrderedDict[str, tuple[float, ...]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    async def embed(self, text: str) -> tuple[float, ...]:
        key = hashlib.sha256(text.encode()).hexdigest()
        cached = self._cache.get(key)
        if cached is not None:
            self.hits += 1
            self._cache.move_to_end(key)
            return cached

        self.misses += 1
        vector = await self._inner.embed(text)
        self._cache[key] = vector
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)  # evict least-recently-used
        return vector

    @property
    def size(self) -> int:
        return len(self._cache)
