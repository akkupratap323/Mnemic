<div align="center">

# 🧠 Mnemic

### Temporal knowledge-graph memory for AI agents

*Give your AI a memory that remembers **what** changed, **when** it changed, and **why** — not just a pile of text.*

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MCP](https://img.shields.io/badge/MCP-ready-8A2BE2.svg)](https://modelcontextprotocol.io)

</div>

---

## Why Mnemic?

Most "AI memory" is a vector store: it chops text into chunks and finds the ones that look similar. That forgets *time*. When a fact changes — a user switches jobs, a project ships, a preference flips — a vector store just piles the new note next to the old one and hopes similarity sorts it out.

**Mnemic models memory as a temporal knowledge graph.** Every fact is an edge with a validity window, so the graph knows that *"Kendra preferred Adidas (Mar–Oct 2026), now prefers Nike"* — and can answer both **what's true now** and **what was true then**. Updates are incremental and real-time; nothing is recomputed in batch.

It's built to plug straight into Claude, Cursor, and any MCP-compatible agent.

## ✨ Features

- **🕰️ Bi-temporal model** — every fact tracks when it became true and when it stopped, so contradictions *invalidate* old facts instead of overwriting them. Full history, full provenance.
- **🔍 Hybrid retrieval** — semantic embeddings + BM25 keyword search + graph traversal, recombined with pluggable rerankers (RRF, MMR, cross-encoder, node-distance). Sub-second, no LLM-summarize-on-every-query.
- **⚡ Real-time incremental updates** — add a memory and the graph updates immediately. No batch reprocessing.
- **🧩 Custom entity & edge types** — define your own schema with plain Pydantic models.
- **🔌 MCP-native** — ships a Model Context Protocol server so agents can `remember` and `search` out of the box.
- **🗄️ Multiple backends** — Neo4j, FalkorDB, Kuzu, or Amazon Neptune.
- **🤖 Provider-agnostic** — OpenAI, Anthropic, Gemini, Groq, or local models via Ollama.

## 🧬 How it works

```
   Episode (raw input)          Entities (nodes)            Facts (temporal edges)
 ┌──────────────────────┐    ┌────────────────────┐    ┌──────────────────────────────┐
 │ "Kendra started at   │ →  │  Kendra            │ →  │ Kendra —[WORKS_AT]→ Initech   │
 │  Initech in March."  │    │  Initech           │    │   valid_at: 2026-03-01        │
 └──────────────────────┘    └────────────────────┘    │   invalid_at: null            │
                                                        └──────────────────────────────┘
```

Each time you call `remember(...)`, Mnemic runs an extraction pipeline: an LLM pulls entities and relationships from the text, **deduplicates** them against the existing graph, and when new information contradicts an old fact it **stamps the old edge with an end date** rather than deleting it. Retrieval then blends vector, keyword, and graph-distance signals to return precise, time-aware context.

## 📦 Installation

```bash
pip install mnemic

# optional backends / providers
pip install mnemic[falkordb]      # FalkorDB driver
pip install mnemic[anthropic]     # Anthropic LLM client
pip install mnemic[google-genai]  # Gemini
```

**Requirements:** Python 3.10+, a graph database (Neo4j 5.26+ or FalkorDB 1.1.2+), and an LLM API key (`OPENAI_API_KEY` by default).

## 🚀 Quick start

```python
import asyncio
from datetime import datetime, timezone
from mnemic import Mnemic
from mnemic.nodes import EpisodeType

async def main():
    mnemic = Mnemic('bolt://localhost:7687', 'neo4j', 'password')

    # one-time setup: indices & constraints
    await mnemic.build_indices_and_constraints()

    # add a memory
    await mnemic.remember(
        name='intro',
        episode_body='Kendra loves Adidas shoes. She is a marathon runner.',
        source=EpisodeType.text,
        source_description='user message',
        reference_time=datetime.now(timezone.utc),
    )

    # the world changes — Mnemic invalidates the old fact, keeps the history
    await mnemic.remember(
        name='update',
        episode_body='Kendra now prefers Nike over Adidas.',
        source=EpisodeType.text,
        source_description='user message',
        reference_time=datetime.now(timezone.utc),
    )

    # hybrid, time-aware search
    results = await mnemic.search('What shoe brand does Kendra like?')
    for edge in results:
        print(edge.fact, '| valid:', edge.valid_at, '| invalid:', edge.invalid_at)

    await mnemic.close()

asyncio.run(main())
```

### Core API

| Method | What it does |
|--------|--------------|
| `remember(...)` | Add an episode; extract & reconcile entities/facts |
| `remember_many(...)` | Bulk-add many episodes |
| `remember_fact(...)` | Add a single entity-relationship triplet directly |
| `search(query)` | Hybrid search over facts (edges) |
| `recent_episodes(...)` | Fetch the most recent raw episodes |
| `forget_episode(uuid)` | Remove an episode and its derived data |
| `build_communities()` | Cluster related entities into communities |

## 🔌 Use it as an MCP server (Claude / Cursor)

Mnemic ships an MCP server so any MCP-compatible agent can read and write memory automatically.

```bash
cd mcp_server
docker-compose up        # starts Mnemic MCP + Neo4j
```

Point your agent at it and these tools become available: `add_memory`, `search_nodes`, `search_memory_facts`, `get_episodes`, and more. See [`mcp_server/README.md`](mcp_server/README.md) for client configuration.

## 🛠️ Development

```bash
uv sync --extra dev     # install deps
make format             # ruff format + import sort
make lint               # ruff + pyright
make test               # pytest
```

Integration tests (suffixed `_int`) need a live database; run unit tests only with `pytest -k "not _int"`.

## 🏛️ Architecture

```
mnemic/
├── core.py          # the Mnemic client — orchestrates everything
├── nodes.py         # Episodic / Entity / Community nodes
├── edges.py         # temporal edges (valid_at / invalid_at / expired_at)
├── driver/          # Neo4j · FalkorDB · Kuzu · Neptune backends
├── llm_client/      # OpenAI · Anthropic · Gemini · Groq
├── embedder/        # embedding providers
├── search/          # hybrid search + reranker recipes
└── prompts/         # extraction · dedup · summarization prompts
```

## 🙏 Acknowledgements

Mnemic is built on top of [**Graphiti**](https://github.com/getzep/graphiti) by Zep Software, Inc., released under the Apache License 2.0. Mnemic renames and reshapes the public API, removes bundled telemetry, and adds its own direction. The original temporal-knowledge-graph engine and its research are Zep's work — see their paper, *[Zep: A Temporal Knowledge Graph Architecture for Agent Memory](https://arxiv.org/abs/2501.13956)*. Full attribution is in [`NOTICE`](NOTICE).

## 📄 License

Apache License 2.0 — see [`LICENSE`](LICENSE).
