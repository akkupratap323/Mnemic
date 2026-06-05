"""
Copyright 2026, Abishek (Mnemic project).

Quantify how good a labeler is, against a hand-labeled gold set. Scores the
keyword heuristic (offline) and — if Ollama is reachable — the local Gemma
teacher, side by side. The gap is the value the teacher adds.

Run:
    ollama serve && ollama pull gemma3:12b
    python tests/evals/check_teacher.py

Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import argparse
import asyncio
import os

from mnemic.hybrid.agreement import AgreementReport, load_gold, measure_agreement
from mnemic.hybrid.api_teacher import ApiTeacher
from mnemic.hybrid.ollama_teacher import OllamaTeacher
from mnemic.hybrid.router_training import HeuristicLabeler

_DEFAULT_GOLD = os.path.join(os.path.dirname(__file__), 'data', 'router_gold.json')


def _make_teacher(backend: str, model: str | None):
    if backend == 'api':  # DeepSeek / OpenAI-compatible
        return ApiTeacher(model=model or 'deepseek-chat')
    return OllamaTeacher(model=model or 'gemma3:12b')


def _print(name: str, r: AgreementReport) -> None:
    print(f'\n{name}  (n={r.n})')
    print(f'  accuracy : {r.accuracy:.1%}')
    print(f'  precision: {r.precision:.1%}   recall: {r.recall:.1%}   f1: {r.f1:.1%}')
    print(f"  cohen's κ: {r.kappa:.3f}")
    print(f'  confusion: tp={r.tp} fp={r.fp} tn={r.tn} fn={r.fn}')


async def run(gold_path: str, backend: str, model: str | None) -> None:
    gold = load_gold(gold_path)

    heuristic = HeuristicLabeler()

    async def heuristic_fn(text: str) -> bool:
        return heuristic.label(text)

    heuristic_report = await measure_agreement(heuristic_fn, gold)
    _print('Keyword heuristic', heuristic_report)

    teacher = _make_teacher(backend, model)
    try:
        teacher_report = await measure_agreement(teacher.label, gold)
        _print(f'Teacher ({backend}: {teacher.model})', teacher_report)
        gap = teacher_report.accuracy - heuristic_report.accuracy
        print(f'\n=> Teacher improves accuracy by {gap:+.1%} '
              f'(κ {heuristic_report.kappa:.2f} -> {teacher_report.kappa:.2f})')
    except Exception as exc:  # noqa: BLE001 - report and continue with heuristic only
        print(f'\n(Skipped teacher — {backend} not reachable: {exc})')
        if backend == 'api':
            print('  Set the key:  export DEEPSEEK_API_KEY=sk-...')
        else:
            print(f'  Start Ollama:  ollama serve && ollama pull {teacher.model}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Score labelers against the gold set')
    parser.add_argument('--gold', default=_DEFAULT_GOLD)
    parser.add_argument('--teacher', choices=['ollama', 'api'], default='ollama')
    parser.add_argument('--model', default=None)
    args = parser.parse_args()
    asyncio.run(run(args.gold, args.teacher, args.model))


if __name__ == '__main__':
    main()
