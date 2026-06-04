"""
Copyright 2026, Abishek (Mnemic project).

A learned graph-worthiness router. Instead of keyword matching, it scores each
write with a linear model over the embedding you already compute, so it captures
semantics keywords miss ("let's just go with Postgres then" -> graph) while
staying near-free at inference (one dot product on an existing vector).
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from mnemic.hybrid.errors import InvalidInput
from mnemic.hybrid.types import RouteDecision, Tier

_TRUE_STRINGS = frozenset({'1', 'true', 'yes', 'on'})


def _truthy(value: object) -> bool:
    if value is True:
        return True
    return isinstance(value, str) and value.strip().lower() in _TRUE_STRINGS


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    exp_z = math.exp(z)
    return exp_z / (1.0 + exp_z)


@dataclass(frozen=True)
class LearnedRouter:
    """Linear graph-worthiness scorer. Drop-in for WriteRouter.

    ``route`` works on a precomputed embedding (the one HybridMemory already has),
    so it adds no embedding cost. Explicit metadata overrides still win.
    """

    weights: tuple[float, ...]
    bias: float = 0.0
    threshold: float = 0.5
    embedder: object | None = None  # only needed for standalone classify()

    def __post_init__(self) -> None:
        if not self.weights:
            raise InvalidInput('weights must be non-empty')
        if not (0.0 <= self.threshold <= 1.0):
            raise InvalidInput('threshold must be in [0, 1]')

    def score_embedding(self, embedding: Sequence[float]) -> float:
        if len(embedding) != len(self.weights):
            raise InvalidInput(
                f'embedding dim {len(embedding)} != weights dim {len(self.weights)}'
            )
        z = math.fsum(w * x for w, x in zip(self.weights, embedding, strict=True)) + self.bias
        return _sigmoid(z)

    def route(
        self,
        content: str,
        metadata: Mapping[str, object] | None = None,
        *,
        embedding: Sequence[float] | None = None,
    ) -> RouteDecision:
        meta = metadata or {}
        if _truthy(meta.get('force_vector_only')):
            return RouteDecision(Tier.VECTOR_ONLY, ('metadata:force_vector_only',))
        if _truthy(meta.get('force_graph')):
            return RouteDecision(Tier.VECTOR_AND_GRAPH, ('metadata:force_graph',))
        if embedding is None:
            raise InvalidInput('LearnedRouter.route requires a precomputed embedding')
        score = self.score_embedding(embedding)
        tier = Tier.VECTOR_AND_GRAPH if score >= self.threshold else Tier.VECTOR_ONLY
        return RouteDecision(tier, (f'learned_score:{score:.3f}',))

    async def classify(
        self, content: str, metadata: Mapping[str, object] | None = None
    ) -> RouteDecision:
        """Standalone: embed then route (for use outside the write path)."""
        if self.embedder is None:
            raise InvalidInput('classify() needs an embedder; use route(embedding=...) instead')
        embedding = await self.embedder.embed(content)  # type: ignore[attr-defined]
        return self.route(content, metadata, embedding=embedding)
