"""
Copyright 2026, Abishek (Mnemic project).

Proof #2 — does the LEARNED router beat the keyword router?

Train the learned router on real teacher labels (DeepSeek) over real embeddings
(bge-m3), then score it against the HUMAN gold set — head to head with the
keyword heuristic. Apples to apples: same gold answer key for both.

Run:
    export DEEPSEEK_API_KEY=sk-...      # teacher
    ollama pull bge-m3                  # embeddings
    python tests/evals/prove_router.py --train 200

Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import argparse
import asyncio
import os

from mnemic.hybrid.agreement import load_gold, measure_agreement
from mnemic.hybrid.api_teacher import ApiTeacher
from mnemic.hybrid.embedder_factory import make_embedder
from mnemic.hybrid.eval import load_longmemeval
from mnemic.hybrid.router_training import (
    HeuristicLabeler,
    build_training_set,
    train_logistic,
)

_DATA = os.path.join(
    os.path.dirname(__file__), 'data', 'longmemeval_data', 'longmemeval_oracle.json'
)
_GOLD = os.path.join(os.path.dirname(__file__), 'data', 'router_gold.json')


def _collect(limit: int) -> list[str]:
    texts: list[str] = []
    for inst in load_longmemeval(_DATA):
        for session in inst.sessions:
            for turn in session:
                if turn.role == 'user' and turn.content:
                    texts.append(turn.content)
                    if len(texts) >= limit:
                        return texts
    return texts


async def run(train_n: int, embedder_backend: str, embed_model, teacher_backend: str) -> None:
    gold = load_gold(_GOLD)
    embedder = make_embedder(embedder_backend, embed_model)
    teacher = ApiTeacher() if teacher_backend == 'api' else None

    texts = _collect(train_n)
    print(f'Labeling {len(texts)} training turns with the teacher ({teacher_backend})...')
    examples = await build_training_set(
        texts, teacher=teacher, teacher_indices=list(range(len(texts)))
    )
    pos = sum(e.label for e in examples)
    print(f'  teacher labeled {pos}/{len(examples)} as graph-worthy')

    print(f'Training learned router on {embedder_backend} embeddings...')
    router = await train_logistic(examples, embedder, epochs=400, lr=0.5)

    # --- keyword router vs human gold ---
    heuristic = HeuristicLabeler()

    async def kw_fn(text: str) -> bool:
        return heuristic.label(text)

    kw = await measure_agreement(kw_fn, gold)

    # --- learned router vs human gold ---
    async def learned_fn(text: str) -> bool:
        vec = await embedder.embed(text)
        return router.route(text, embedding=vec).writes_graph

    learned = await measure_agreement(learned_fn, gold)

    print('\n=== Proof #2: router quality on the HUMAN gold set ===')
    print(f'{"router":<20}{"accuracy":>10}{"recall":>9}{"f1":>8}{"kappa":>8}')
    print('-' * 55)
    print(f'{"keyword":<20}{kw.accuracy:>9.1%}{kw.recall:>9.1%}{kw.f1:>8.2f}{kw.kappa:>8.2f}')
    print(f'{"learned (distilled)":<20}{learned.accuracy:>9.1%}'
          f'{learned.recall:>9.1%}{learned.f1:>8.2f}{learned.kappa:>8.2f}')
    print(f'\n=> learned router accuracy {learned.accuracy - kw.accuracy:+.1%} vs keyword '
          f'(κ {kw.kappa:.2f} -> {learned.kappa:.2f})')


def main() -> None:
    parser = argparse.ArgumentParser(description='Proof #2: learned vs keyword router')
    parser.add_argument('--train', type=int, default=200)
    parser.add_argument('--embedder', default='ollama')
    parser.add_argument('--embed-model', default=None)
    parser.add_argument('--teacher', choices=['api', 'ollama'], default='api')
    args = parser.parse_args()
    asyncio.run(run(args.train, args.embedder, args.embed_model, args.teacher))


if __name__ == '__main__':
    main()
