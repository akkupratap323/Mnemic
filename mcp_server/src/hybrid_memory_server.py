"""
Copyright 2026, Abishek (Mnemic project).

Standalone MCP server that exposes Mnemic's hybrid tiered memory
(remember / recall / recall_code / forget) to MCP clients like Claude and Cursor.

Run:
    export OPENAI_API_KEY=...   NEO4J_URI=bolt://localhost:7687
    export NEO4J_USER=neo4j     NEO4J_PASSWORD=...
    python -m hybrid_memory_server

Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP

from mnemic import Mnemic
from mnemic.embedder import OpenAIEmbedder
from mnemic.hybrid import build_hybrid_memory, register_memory_tools


def build_server() -> FastMCP:
    """Construct the MCP server with hybrid memory wired to a live Mnemic engine."""
    engine = Mnemic(
        uri=os.environ.get('NEO4J_URI', 'bolt://localhost:7687'),
        user=os.environ.get('NEO4J_USER', 'neo4j'),
        password=os.environ.get('NEO4J_PASSWORD', ''),
    )
    embedder = OpenAIEmbedder()

    memory = build_hybrid_memory(
        mnemic_client=engine,
        embedder_client=embedder,
        db_path=os.environ.get('MNEMIC_DB_PATH', 'mnemic_memory.db'),
    )

    mcp = FastMCP('mnemic-hybrid-memory')
    register_memory_tools(mcp, memory)
    return mcp


def main() -> None:
    build_server().run()


if __name__ == '__main__':
    main()
