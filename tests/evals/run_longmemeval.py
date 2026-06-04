"""
Copyright 2026, Abishek (Mnemic project).

Run the hybrid memory against the LongMemEval benchmark and print recall metrics.

    # offline baseline (no API key needed)
    python tests/evals/run_longmemeval.py --limit 50

    # real embeddings
    OPENAI_API_KEY=... python tests/evals/run_longmemeval.py --limit 100

Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from mnemic.hybrid.eval import EvalReport, evaluate, load_longmemeval
from mnemic.hybrid.hashing_embedder import HashingEmbedder
from mnemic.hybrid.memory import HybridMemory
from mnemic.hybrid.vector_store import InMemoryVectorStore

_DEFAULT_DATA = os.path.join(
    os.path.dirname(__file__), 'data', 'longmemeval_data', 'longmemeval_oracle.json'
)


def _make_embedder():
    if os.environ.get('OPENAI_API_KEY'):
        from mnemic.embedder import OpenAIEmbedder
        from mnemic.hybrid.embedder import EmbedderClientAdapter

        print('Using OpenAIEmbedder (real embeddings).', file=sys.stderr)
        return EmbedderClientAdapter(OpenAIEmbedder())
    print('No OPENAI_API_KEY set — using offline HashingEmbedder baseline.', file=sys.stderr)
    return HashingEmbedder()


def _print_report(report: EvalReport) -> None:
    print(f'\nLongMemEval — {report.total} questions, k={report.k}')
    print(f'  evidence-recall@{report.k}: {report.evidence_recall_at_k:.1%}')
    print(f'  answer-recall@{report.k}:   {report.answer_recall_at_k:.1%}')
    print('  by question type:')
    for qtype, score in sorted(report.by_type.items(), key=lambda kv: -kv[1]):
        print(f'    {qtype:<28} {score:.1%}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate hybrid memory on LongMemEval')
    parser.add_argument('--data', default=_DEFAULT_DATA)
    parser.add_argument('--limit', type=int, default=50)
    parser.add_argument('--k', type=int, default=10)
    args = parser.parse_args()

    instances = load_longmemeval(args.data)
    embedder = _make_embedder()

    def make_memory() -> HybridMemory:
        return HybridMemory(embedder=embedder, vector_store=InMemoryVectorStore())

    report = asyncio.run(evaluate(instances, make_memory, k=args.k, limit=args.limit))
    _print_report(report)


if __name__ == '__main__':
    main()
