"""
Copyright 2026, Abishek (Mnemic project).

Lightweight in-process counters for observability: how much lands in the cheap
vector tier vs the premium graph tier, and recall volume. Export the snapshot to
Prometheus/StatsD/etc. at the edge.
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Metrics:
    """Mutable counters incremented by HybridMemory."""

    remembers: int = 0
    vector_writes: int = 0
    graph_writes: int = 0
    recalls: int = 0
    fused_recalls: int = 0

    def record_remember(self, *, graph_written: bool) -> None:
        self.remembers += 1
        self.vector_writes += 1
        if graph_written:
            self.graph_writes += 1

    def record_recall(self) -> None:
        self.recalls += 1

    def record_fused_recall(self) -> None:
        self.fused_recalls += 1

    @property
    def graph_write_ratio(self) -> float:
        """Fraction of remembers that reached the expensive graph tier."""
        return self.graph_writes / self.remembers if self.remembers else 0.0

    def snapshot(self) -> dict[str, float]:
        return {
            'remembers': self.remembers,
            'vector_writes': self.vector_writes,
            'graph_writes': self.graph_writes,
            'recalls': self.recalls,
            'fused_recalls': self.fused_recalls,
            'graph_write_ratio': self.graph_write_ratio,
        }
