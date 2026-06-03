"""
Copyright 2026, Abishek (Mnemic project).

Embedder protocol and an adapter over Mnemic's EmbedderClient, so the hybrid
layer stays decoupled from any specific embedding provider.
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from mnemic.hybrid.errors import InvalidInput


@runtime_checkable
class Embedder(Protocol):
    """Turns text into a fixed-length vector."""

    async def embed(self, text: str) -> tuple[float, ...]: ...


class EmbedderClientAdapter:
    """Adapts ``mnemic.embedder.EmbedderClient`` to the hybrid Embedder protocol."""

    def __init__(self, client: object) -> None:
        self._client = client

    async def embed(self, text: str) -> tuple[float, ...]:
        if not isinstance(text, str) or not text.strip():
            raise InvalidInput('text to embed must be a non-empty string')
        vector = await self._client.create(text)  # type: ignore[attr-defined]
        return tuple(float(x) for x in vector)
