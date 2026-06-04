"""
Copyright 2026, Abishek (Mnemic project).

A dependency-free, deterministic bag-of-words embedder. No API keys, no models —
useful for offline dev, tests, CI, and as an eval baseline. Swap for a real
embedder (OpenAI, Voyage, sentence-transformers) in production.
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import hashlib
import math
import re

from mnemic.hybrid.errors import InvalidInput

_TOKEN_RE = re.compile(r'[a-z0-9]+')


class HashingEmbedder:
    """Hashes tokens into a fixed-dimension, L2-normalised vector."""

    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            raise InvalidInput('dim must be a positive integer')
        self.dim = dim

    async def embed(self, text: str) -> tuple[float, ...]:
        if not isinstance(text, str) or not text.strip():
            raise InvalidInput('text to embed must be a non-empty string')
        vec = [0.0] * self.dim
        for token in _TOKEN_RE.findall(text.lower()):
            digest = hashlib.md5(token.encode(), usedforsecurity=False).digest()
            idx = int.from_bytes(digest[:4], 'big') % self.dim
            vec[idx] += 1.0
        norm = math.sqrt(math.fsum(x * x for x in vec))
        if norm == 0.0:
            return tuple([1.0] + [0.0] * (self.dim - 1))
        return tuple(x / norm for x in vec)
