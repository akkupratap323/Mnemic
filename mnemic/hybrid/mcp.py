"""
Copyright 2026, Abishek (Mnemic project).

Registers the hybrid memory operations as MCP tools so agents (Claude, Cursor)
can use the tiered memory directly. Decoupled from any specific MCP framework:
pass any object with a ``tool()`` decorator factory (e.g. FastMCP).
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

from typing import Any

from mnemic.hybrid.memory import HybridMemory
from mnemic.hybrid.tools import MemoryTools


def register_memory_tools(mcp: Any, memory: HybridMemory) -> list[str]:
    """Register remember/recall/recall_code/forget on an MCP server.

    Returns the list of registered tool names.
    """
    tools = MemoryTools(memory)

    async def remember(content: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        """Store a memory. Always saved to the fast vector tier; promoted to the
        temporal knowledge graph when significant (decisions, bugs, facts)."""
        return await tools.remember(content, metadata=metadata)

    async def recall(query: str, k: int = 10) -> list[dict[str, Any]]:
        """Retrieve memories for a query, blending vector similarity and graph
        facts via Reciprocal Rank Fusion."""
        return await tools.recall(query, k=k)

    async def recall_code(
        query: str, entity: str | None = None, k: int = 10
    ) -> list[dict[str, Any]]:
        """Retrieve code memories, optionally filtered by entity type
        (File, Function, Decision, Bug, ...)."""
        return await tools.recall_code(query, entity=entity, k=k)

    async def forget(item_id: str) -> dict[str, Any]:
        """Delete a stored memory by id."""
        return await tools.forget(item_id)

    registered: list[str] = []
    for fn in (remember, recall, recall_code, forget):
        mcp.tool()(fn)
        registered.append(fn.__name__)
    return registered
