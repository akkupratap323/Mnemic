"""
Copyright 2026, Abishek (Mnemic project).

The consolidation worker — Mnemic's "sleep" pass. It promotes frequently
accessed vector memories into the premium graph tier, and prunes stale,
never-used ones. Runs off the hot path so writes stay fast.
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone

from mnemic.hybrid.access import AccessLog
from mnemic.hybrid.memory import GraphMemory
from mnemic.hybrid.vector_store import VectorStore

_THIRTY_DAYS_SECONDS = 30 * 24 * 3600


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class ConsolidationPolicy:
    """Pure rules for what to promote and what to prune."""

    promote_after_accesses: int = 3
    prune_after_seconds: float = _THIRTY_DAYS_SECONDS
    prune_max_accesses: int = 0

    def should_promote(self, *, access_count: int, already_promoted: bool) -> bool:
        return not already_promoted and access_count >= self.promote_after_accesses

    def should_prune(
        self, *, access_count: int, age_seconds: float, already_promoted: bool
    ) -> bool:
        return (
            not already_promoted
            and age_seconds >= self.prune_after_seconds
            and access_count <= self.prune_max_accesses
        )


@dataclass(frozen=True)
class ConsolidationReport:
    """Outcome of a single consolidation pass."""

    scanned: int
    promoted: tuple[str, ...] = ()
    pruned: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()


@dataclass
class Consolidator:
    """Runs a consolidation pass over the vector tier."""

    vector_store: VectorStore
    graph: GraphMemory
    access_log: AccessLog
    policy: ConsolidationPolicy = field(default_factory=ConsolidationPolicy)
    clock: Callable[[], datetime] = _utcnow

    async def run(self) -> ConsolidationReport:
        now = self.clock()
        items = await self.vector_store.all_items()
        promoted: list[str] = []
        pruned: list[str] = []
        errors: list[str] = []

        for item in items:
            count = self.access_log.count(item.id)
            already = self.access_log.is_promoted(item.id)
            age = (now - item.created_at).total_seconds() if item.created_at else 0.0

            if self.policy.should_promote(access_count=count, already_promoted=already):
                try:
                    await self.graph.remember(
                        name=f'memory-{item.id}',
                        episode_body=item.content,
                        source_description='consolidation',
                        reference_time=item.created_at or now,
                    )
                except Exception as exc:  # retry next pass; never crash the worker
                    errors.append(f'{item.id}: {type(exc).__name__}: {exc}')
                    continue
                self.access_log.mark_promoted(item.id)
                promoted.append(item.id)
                continue

            if self.policy.should_prune(
                access_count=count, age_seconds=age, already_promoted=already
            ):
                await self.vector_store.delete(item.id)
                pruned.append(item.id)

        return ConsolidationReport(
            scanned=len(items),
            promoted=tuple(promoted),
            pruned=tuple(pruned),
            errors=tuple(errors),
        )
