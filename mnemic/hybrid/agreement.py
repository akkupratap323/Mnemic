"""
Copyright 2026, Abishek (Mnemic project).

Measure how well a labeler (the heuristic, or a local-LLM teacher) agrees with a
hand-labeled gold set. Use this to *quantify* teacher quality before trusting its
labels for distillation — accuracy, precision/recall/F1, and Cohen's kappa
(agreement corrected for chance).
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from mnemic.hybrid.errors import InvalidInput


@dataclass(frozen=True)
class GoldExample:
    text: str
    label: bool


@dataclass(frozen=True)
class AgreementReport:
    n: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    kappa: float
    tp: int
    fp: int
    tn: int
    fn: int


def load_gold(path: str) -> list[GoldExample]:
    with open(path) as fh:
        raw = json.load(fh)
    return [GoldExample(text=str(r['text']), label=bool(r['label'])) for r in raw]


def compute_agreement(preds: Sequence[bool], gold: Sequence[bool]) -> AgreementReport:
    """Pure metric computation from predicted vs gold boolean labels."""
    if len(preds) != len(gold):
        raise InvalidInput('preds and gold must be the same length')
    n = len(gold)
    if n == 0:
        raise InvalidInput('need at least one example')

    tp = sum(p and g for p, g in zip(preds, gold, strict=True))
    fp = sum(p and not g for p, g in zip(preds, gold, strict=True))
    tn = sum((not p) and (not g) for p, g in zip(preds, gold, strict=True))
    fn = sum((not p) and g for p, g in zip(preds, gold, strict=True))

    accuracy = (tp + tn) / n
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    kappa = _cohen_kappa(tp, fp, tn, fn, n)

    return AgreementReport(
        n=n,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        kappa=kappa,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
    )


def _cohen_kappa(tp: int, fp: int, tn: int, fn: int, n: int) -> float:
    po = (tp + tn) / n
    p_yes_pred = (tp + fp) / n
    p_yes_gold = (tp + fn) / n
    pe = p_yes_pred * p_yes_gold + (1 - p_yes_pred) * (1 - p_yes_gold)
    if pe >= 1.0:
        # degenerate (one class only): perfect agreement -> 1.0, else 0.0
        return 1.0 if po >= 1.0 else 0.0
    return (po - pe) / (1 - pe)


async def measure_agreement(
    label_fn: Callable[[str], Awaitable[bool]], gold: Sequence[GoldExample]
) -> AgreementReport:
    """Run a labeler over the gold set and score its agreement."""
    if not gold:
        raise InvalidInput('gold set is empty')
    preds = [await label_fn(example.text) for example in gold]
    return compute_agreement(preds, [example.label for example in gold])
