"""
Copyright 2026, Abishek (Mnemic project).

Build a small PERSISTENT Mnemic graph (group_id 'demo_view') so it can be
explored visually in the Neo4j Browser. Real pipeline: DeepSeek + bge-m3 + Neo4j.
Not cleaned up (so you can look at it); delete with the query printed at the end.

Run:
    export DEEPSEEK_API_KEY=...  OPENAI_API_KEY=sk-x
    export NEO4J_URI=bolt://localhost:7688 NEO4J_PASSWORD=mnemic_bench_pw
    python tests/evals/make_demo_graph.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))
from live_cost_probe import MeteredDeepSeek

from mnemic import Mnemic
from mnemic.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from mnemic.hybrid.eval import load_longmemeval
from mnemic.llm_client.config import LLMConfig

GROUP = 'demo_view'
_DATA = os.path.join(os.path.dirname(__file__), 'data', 'longmemeval_data', 'longmemeval_oracle.json')


async def main() -> None:
    llm = MeteredDeepSeek(
        config=LLMConfig(
            api_key=os.environ['DEEPSEEK_API_KEY'], base_url='https://api.deepseek.com',
            model='deepseek-chat', small_model='deepseek-chat',
        ),
        max_tokens=8000,
    )
    embedder = OpenAIEmbedder(
        config=OpenAIEmbedderConfig(
            api_key='ollama', base_url='http://localhost:11434/v1',
            embedding_model='bge-m3', embedding_dim=1024,
        )
    )
    engine = Mnemic(
        os.environ.get('NEO4J_URI', 'bolt://localhost:7688'),
        os.environ.get('NEO4J_USER', 'neo4j'),
        os.environ['NEO4J_PASSWORD'],
        llm_client=llm, embedder=embedder,
    )
    await engine.build_indices_and_constraints()

    # a few substantive, related turns so the graph has connected entities
    episodes = [
        'We decided to migrate the billing service from MySQL to PostgreSQL.',
        'After the migration, Maria took over as the lead on the billing service.',
        'The PostgreSQL migration fixed the payment timeout bug we had in checkout.',
    ]
    for i, text in enumerate(episodes):
        await engine.remember(
            name=f'demo-{i}', episode_body=text, source_description='demo',
            reference_time=datetime.now(timezone.utc), group_id=GROUP,
        )
        print(f'ingested demo-{i}: {text[:50]}...', flush=True)

    await engine.close()
    print('\nDONE. View it in the Neo4j Browser on bolt://localhost:7688:')
    print("  MATCH p=(n {group_id:'demo_view'})-[r]-(m) RETURN p")
    print("Delete it later with:")
    print("  MATCH (n {group_id:'demo_view'}) DETACH DELETE n")


if __name__ == '__main__':
    asyncio.run(main())
