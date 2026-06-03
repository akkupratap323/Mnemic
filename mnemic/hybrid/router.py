"""
Copyright 2026, Abishek (Mnemic project).

The write router: the public seam that decides which tiers a memory is
written to. Thin wrapper over the classifier so policy can evolve here.
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from mnemic.hybrid.classifier import SignalClassifier
from mnemic.hybrid.types import RouteDecision


@dataclass(frozen=True)
class WriteRouter:
    """Decides which tiers a memory should be written to."""

    classifier: SignalClassifier = field(default_factory=SignalClassifier)

    def route(
        self, content: str, metadata: Mapping[str, object] | None = None
    ) -> RouteDecision:
        return self.classifier.classify(content, metadata)
