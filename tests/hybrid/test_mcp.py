"""Tests for MCP tool registration (using a fake MCP server)."""

from __future__ import annotations

from datetime import datetime, timezone

from mnemic.hybrid.mcp import register_memory_tools
from mnemic.hybrid.memory import HybridMemory
from mnemic.hybrid.vector_store import InMemoryVectorStore
from tests.hybrid.conftest import FakeEmbedder, RecordingGraph, make_counter_ids

FIXED = datetime(2026, 1, 1, tzinfo=timezone.utc)


class FakeMCP:
    """Mimics FastMCP's ``mcp.tool()`` decorator factory."""

    def __init__(self) -> None:
        self.registered: dict[str, object] = {}

    def tool(self, *args: object, **kwargs: object):
        def deco(fn):
            self.registered[fn.__name__] = fn
            return fn

        return deco


def build(graph=None) -> HybridMemory:
    return HybridMemory(
        embedder=FakeEmbedder(),
        vector_store=InMemoryVectorStore(),
        graph=graph,
        clock=lambda: FIXED,
        id_factory=make_counter_ids(),
    )


def test_registers_all_four_tools():
    mcp = FakeMCP()
    names = register_memory_tools(mcp, build())
    assert set(names) == {'remember', 'recall', 'recall_code', 'forget'}
    assert set(mcp.registered) == {'remember', 'recall', 'recall_code', 'forget'}


async def test_remember_tool_routes_to_graph():
    graph = RecordingGraph()
    mcp = FakeMCP()
    register_memory_tools(mcp, build(graph))
    out = await mcp.registered['remember']('decision: adopt MCP tools')
    assert out['graph_written'] is True
    assert len(graph.calls) == 1


async def test_recall_tool_returns_results():
    mcp = FakeMCP()
    mem = build()
    register_memory_tools(mcp, mem)
    await mcp.registered['remember']('alpha beta gamma')
    results = await mcp.registered['recall']('alpha beta gamma', 5)
    assert results
    assert results[0]['content'] == 'alpha beta gamma'


async def test_recall_code_tool_filters_by_entity():
    mcp = FakeMCP()
    register_memory_tools(mcp, build())
    await mcp.registered['remember']('login fn', {'entity': 'Function'})
    await mcp.registered['remember']('schema file', {'entity': 'File'})
    results = await mcp.registered['recall_code']('x', 'Function', 10)
    assert len(results) == 1
    assert results[0]['metadata']['entity'] == 'Function'


async def test_forget_tool_deletes():
    mcp = FakeMCP()
    mem = build()
    register_memory_tools(mcp, mem)
    out = await mcp.registered['remember']('temp note')
    forgotten = await mcp.registered['forget'](str(out['id']))
    assert forgotten['deleted'] is True
