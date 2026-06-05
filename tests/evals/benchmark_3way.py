"""
Copyright 2026, Abishek (Mnemic project).

Run the 3-way proof benchmark on LongMemEval and print the cost-vs-recall table:
A (vector-only) vs B (full-graph / "before") vs C (hybrid / "after").

Offline (default) proves the COST structure. Add a live graph for the full
quality picture (needs Neo4j + an LLM):
    NEO4J_URI=... OPENAI_API_KEY=... python tests/evals/benchmark_3way.py --live

Run (offline, free):
    python tests/evals/benchmark_3way.py --limit 100 --embedder ollama

Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import argparse
import asyncio
import os

from mnemic.hybrid.benchmark import BenchmarkReport, run_three_way
from mnemic.hybrid.embedder_factory import EMBEDDER_BACKENDS, make_embedder
from mnemic.hybrid.eval import load_longmemeval
from mnemic.hybrid.router import WriteRouter

_DEFAULT_DATA = os.path.join(
    os.path.dirname(__file__), 'data', 'longmemeval_data', 'longmemeval_oracle.json'
)


def _live_graph_factory(embedder_backend: str):  # pragma: no cover - needs Neo4j + LLM
    from mnemic import Mnemic
    from mnemic.embedder import OpenAIEmbedder
    from mnemic.hybrid.fusion import MnemicGraphSearcher

    def factory():
        engine = Mnemic(
            uri=os.environ.get('NEO4J_URI', 'bolt://localhost:7687'),
            user=os.environ.get('NEO4J_USER', 'neo4j'),
            password=os.environ.get('NEO4J_PASSWORD', ''),
            embedder=OpenAIEmbedder(),
        )
        return engine, MnemicGraphSearcher(engine)

    return factory


def _print_table(report: BenchmarkReport) -> None:
    b = report.by_name('B_full_graph')
    c = report.by_name('C_hybrid')
    print(f'\n3-way benchmark  (k={report.k})\n')
    header = f'{"config":<16}{"graph%":>8}{"llm_calls":>11}{"est_$":>10}{"est_min":>9}{"recall":>9}'
    print(header)
    print('-' * len(header))
    for r in report.results:
        print(
            f'{r.name:<16}{r.graph_fraction:>7.0%}{r.cost.llm_calls:>11}'
            f'{r.cost.est_usd:>10.3f}{r.cost.est_seconds / 60:>9.1f}{r.evidence_recall_at_k:>9.1%}'
        )
    ratio = c.cost.llm_calls / b.cost.llm_calls if b.cost.llm_calls else 0.0
    print(f'\n=> Hybrid (C) used {ratio:.0%} of full-graph (B) LLM cost '
          f'[{c.cost.llm_calls} vs {b.cost.llm_calls} calls]')
    print('   Recall by question type (C, hybrid):')
    for qtype, score in sorted(c.by_type.items(), key=lambda kv: -kv[1]):
        print(f'     {qtype:<28} {score:.1%}')


async def run(data: str, limit: int, k: int, embedder_backend: str, embed_model, live: bool) -> None:
    instances = load_longmemeval(data)
    embedder = make_embedder(embedder_backend, embed_model)
    graph_factory = _live_graph_factory(embedder_backend) if live else None
    if not live:
        print('OFFLINE mode: proves the COST structure. '
              'Add --live (Neo4j + LLM) for the full recall picture.')
    report = await run_three_way(
        instances, embedder=embedder, router=WriteRouter(),
        graph_factory=graph_factory, k=k, limit=limit,
    )
    _print_table(report)


def main() -> None:
    parser = argparse.ArgumentParser(description='3-way cost-vs-recall benchmark')
    parser.add_argument('--data', default=_DEFAULT_DATA)
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--embedder', choices=EMBEDDER_BACKENDS, default='hashing')
    parser.add_argument('--embed-model', default=None)
    parser.add_argument('--live', action='store_true', help='use a live Neo4j graph tier')
    args = parser.parse_args()
    asyncio.run(run(args.data, args.limit, args.k, args.embedder, args.embed_model, args.live))


if __name__ == '__main__':
    main()
