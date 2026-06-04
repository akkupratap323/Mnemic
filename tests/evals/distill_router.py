"""
Copyright 2026, Abishek (Mnemic project).

Distill a learned router using a LOCAL Gemma teacher (via Ollama). Demonstrates
the value the teacher adds over the keyword heuristic, then trains + evaluates
the learned router.

Prereqs:
    # install Ollama from https://ollama.com, then:
    ollama pull gemma3:12b      # ~7-8 GB; runs on Apple Silicon via Metal

Run:
    python tests/evals/distill_router.py --limit 300 --teacher-fraction 0.4

Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import argparse
import asyncio
import os

from mnemic.hybrid.eval import load_longmemeval
from mnemic.hybrid.hashing_embedder import HashingEmbedder
from mnemic.hybrid.ollama_teacher import OllamaTeacher
from mnemic.hybrid.router_training import (
    HeuristicLabeler,
    build_training_set,
    evaluate_router,
    train_logistic,
)

_DEFAULT_DATA = os.path.join(
    os.path.dirname(__file__), 'data', 'longmemeval_data', 'longmemeval_oracle.json'
)


def _collect_texts(data_path: str, limit: int) -> list[str]:
    texts: list[str] = []
    for inst in load_longmemeval(data_path):
        for session in inst.sessions:
            for turn in session:
                if turn.role == 'user' and turn.content:
                    texts.append(turn.content)
                    if len(texts) >= limit:
                        return texts
    return texts


async def run(data_path: str, limit: int, teacher_fraction: float, model: str) -> None:
    texts = _collect_texts(data_path, limit)
    print(f'Collected {len(texts)} user turns.')

    teacher = OllamaTeacher(model=model)
    n_teacher = int(len(texts) * teacher_fraction)
    teacher_indices = list(range(n_teacher))

    print(f'Labeling {n_teacher} with local {model}, the rest with the heuristic...')
    examples = await build_training_set(
        texts, teacher=teacher, teacher_indices=teacher_indices
    )

    # how often did the teacher disagree with the heuristic? that disagreement is
    # exactly the signal a keyword router cannot see.
    heuristic = HeuristicLabeler()
    disagreements = sum(
        examples[i].label != heuristic.label(texts[i]) for i in range(n_teacher)
    )
    rate = disagreements / n_teacher if n_teacher else 0.0
    print(f'\nTeacher vs heuristic disagreement: {disagreements}/{n_teacher} ({rate:.1%})')
    print('  ^ these are facts the keyword router would have mis-routed.\n')

    embedder = HashingEmbedder(dim=256)
    router = await train_logistic(examples, embedder, epochs=400, lr=0.5)
    metrics = await evaluate_router(router, examples, embedder)
    print('Learned router (distilled) fit on the labeled set:')
    for key in ('precision', 'recall', 'f1', 'auc'):
        print(f'  {key}: {metrics[key]:.3f}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Distill a learned router with a local Gemma teacher')
    parser.add_argument('--data', default=_DEFAULT_DATA)
    parser.add_argument('--limit', type=int, default=300)
    parser.add_argument('--teacher-fraction', type=float, default=0.4)
    parser.add_argument('--model', default='gemma3:12b')
    args = parser.parse_args()
    asyncio.run(run(args.data, args.limit, args.teacher_fraction, args.model))


if __name__ == '__main__':
    main()
