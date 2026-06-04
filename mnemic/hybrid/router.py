"""
Copyright 2026, Abishek (Mnemic project).

The write router: the seam that decides which tiers a memory is written to.
The default is the heuristic SignalClassifier; a LearnedRouter is a drop-in
(both satisfy the Router protocol and accept the precomputed embedding).
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from mnemic.hybrid.classifier import SignalClassifier
from mnemic.hybrid.types import RouteDecision


@runtime_checkable
class Router(Protocol):
    """Decides the tier(s) for a write, optionally using a precomputed embedding."""

    def route(
        self,
        content: str,
        metadata: Mapping[str, object] | None = None,
        *,
        embedding: Sequence[float] | None = None,
    ) -> RouteDecision: ...


@dataclass(frozen=True)
class WriteRouter:
    """Heuristic router. Ignores the embedding (keyword/metadata only)."""

    classifier: SignalClassifier = field(default_factory=SignalClassifier)

    def route(
        self,
        content: str,
        metadata: Mapping[str, object] | None = None,
        *,
        embedding: Sequence[float] | None = None,
    ) -> RouteDecision:
        return self.classifier.classify(content, metadata)
