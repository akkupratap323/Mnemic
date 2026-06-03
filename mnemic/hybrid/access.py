"""
Copyright 2026, Abishek (Mnemic project).

Tracks how often memories are accessed and which have been promoted to the
graph tier. Used by the consolidation worker to decide what graduates.
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations


class AccessLog:
    """In-memory access counter + promotion ledger (swap for Redis at scale)."""

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}
        self._promoted: set[str] = set()

    def record(self, item_id: str, n: int = 1) -> None:
        if n < 0:
            raise ValueError('access increment must be non-negative')
        self._counts[item_id] = self._counts.get(item_id, 0) + n

    def count(self, item_id: str) -> int:
        return self._counts.get(item_id, 0)

    def mark_promoted(self, item_id: str) -> None:
        self._promoted.add(item_id)

    def is_promoted(self, item_id: str) -> bool:
        return item_id in self._promoted

    def snapshot(self) -> dict[str, int]:
        return dict(self._counts)
