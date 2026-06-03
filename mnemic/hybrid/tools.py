"""
Copyright 2026, Abishek (Mnemic project).

MCP-friendly facade over HybridMemory: high-level remember / recall / recall_code
/ forget operations that return plain JSON-serialisable dicts, ready to register
as agent tools (Claude, Cursor).
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from mnemic.hybrid.memory import HybridMemory


@dataclass
class MemoryTools:
    """Thin, serialisation-friendly wrapper around HybridMemory."""

    memory: HybridMemory

    async def remember(
        self, content: str, *, metadata: Mapping[str, object] | None = None
    ) -> dict[str, object]:
        result = await self.memory.remember(content, metadata=metadata)
        return {
            'id': result.item.id,
            'tier': result.decision.tier.value,
            'graph_written': result.graph_written,
            'reasons': list(result.decision.reasons),
        }

    async def recall(self, query: str, *, k: int = 10) -> list[dict[str, object]]:
        hits = await self.memory.recall_fused(query, k=k)
        return [
            {'content': h.content, 'score': h.score, 'sources': list(h.sources)} for h in hits
        ]

    async def recall_code(
        self, query: str, *, entity: str | None = None, k: int = 10
    ) -> list[dict[str, object]]:
        where = {'entity': entity} if entity else None
        hits = await self.memory.recall(query, k=k, where=where)
        return [
            {
                'id': h.item.id,
                'content': h.item.content,
                'score': h.score,
                'metadata': dict(h.item.metadata),
            }
            for h in hits
        ]

    async def forget(self, item_id: str) -> dict[str, object]:
        deleted = await self.memory.vector_store.delete(item_id)
        return {'id': item_id, 'deleted': deleted}
