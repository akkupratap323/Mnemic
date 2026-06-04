"""
Copyright 2026, Abishek (Mnemic project).

Distillation + self-correction pipeline for the learned router.

  teacher (heuristic + occasional LLM judgments) -> labels
  -> train a cheap linear student over embeddings  -> LearnedRouter
  -> mine retrieval failures -> relabel -> retrain (the self-correcting loop)

Training is deterministic (zero init), so results are reproducible.
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from mnemic.hybrid.classifier import SignalClassifier
from mnemic.hybrid.embedder import Embedder
from mnemic.hybrid.errors import InvalidInput
from mnemic.hybrid.learned_router import LearnedRouter


@dataclass(frozen=True)
class LabeledExample:
    """A training example: text, graph-worthiness label, and a sample weight."""

    text: str
    label: bool
    weight: float = 1.0


@runtime_checkable
class TeacherLabeler(Protocol):
    """An expensive, high-quality labeler (e.g. an LLM judge)."""

    async def label(self, text: str) -> bool: ...


@dataclass(frozen=True)
class HeuristicLabeler:
    """Cheap weak labels from the existing keyword classifier."""

    classifier: SignalClassifier = field(default_factory=SignalClassifier)

    def label(self, text: str) -> bool:
        return self.classifier.classify(text).writes_graph


async def build_training_set(
    texts: Sequence[str],
    *,
    heuristic: HeuristicLabeler | None = None,
    teacher: TeacherLabeler | None = None,
    teacher_indices: Sequence[int] | None = None,
    teacher_weight: float = 2.0,
) -> list[LabeledExample]:
    """Label texts with the heuristic, refining a chosen subset with the teacher."""
    labeler = heuristic or HeuristicLabeler()
    refine = set(teacher_indices or [])
    examples: list[LabeledExample] = []
    for i, text in enumerate(texts):
        if teacher is not None and i in refine:
            examples.append(LabeledExample(text, await teacher.label(text), teacher_weight))
        else:
            examples.append(LabeledExample(text, labeler.label(text), 1.0))
    return examples


def add_failure_labels(
    examples: Sequence[LabeledExample],
    failure_texts: Sequence[str],
    *,
    weight: float = 3.0,
) -> list[LabeledExample]:
    """Self-correction: facts that were needed but cheap-tiered become positives.

    Returns a NEW list (immutable); high weight so the next retrain prioritises
    not repeating the same retrieval miss.
    """
    return [*examples, *(LabeledExample(t, True, weight) for t in failure_texts)]


async def train_logistic(
    examples: Sequence[LabeledExample],
    embedder: Embedder,
    *,
    epochs: int = 300,
    lr: float = 0.5,
    l2: float = 1e-3,
) -> LearnedRouter:
    """Train an L2-regularised logistic regression over embeddings (deterministic)."""
    if not examples:
        raise InvalidInput('need at least one training example')
    if epochs <= 0:
        raise InvalidInput('epochs must be positive')

    rows = [list(await embedder.embed(ex.text)) for ex in examples]
    features = np.asarray(rows, dtype=float)
    labels = np.asarray([1.0 if ex.label else 0.0 for ex in examples], dtype=float)
    sample_w = np.asarray([ex.weight for ex in examples], dtype=float)

    n_samples, n_dims = features.shape
    weights = np.zeros(n_dims, dtype=float)
    bias = 0.0
    for _ in range(epochs):
        logits = features @ weights + bias
        probs = 1.0 / (1.0 + np.exp(-logits))
        error = (probs - labels) * sample_w
        grad_w = features.T @ error / n_samples + l2 * weights
        grad_b = float(error.sum() / n_samples)
        weights -= lr * grad_w
        bias -= lr * grad_b

    return LearnedRouter(
        weights=tuple(float(w) for w in weights),
        bias=float(bias),
        threshold=0.5,
        embedder=embedder,
    )


async def evaluate_router(
    router: LearnedRouter,
    examples: Sequence[LabeledExample],
    embedder: Embedder,
) -> dict[str, float]:
    """Routing-quality metrics: precision, recall, f1, and ROC-AUC."""
    scores: list[float] = []
    labels: list[bool] = []
    for ex in examples:
        embedding = await embedder.embed(ex.text)
        scores.append(router.score_embedding(embedding))
        labels.append(ex.label)

    preds = [s >= router.threshold for s in scores]
    tp = sum(p and y for p, y in zip(preds, labels, strict=True))
    fp = sum(p and not y for p, y in zip(preds, labels, strict=True))
    fn = sum((not p) and y for p, y in zip(preds, labels, strict=True))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': _roc_auc(scores, labels),
        'n': float(len(examples)),
    }


def _roc_auc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    pos = [s for s, y in zip(scores, labels, strict=True) if y]
    neg = [s for s, y in zip(scores, labels, strict=True) if not y]
    if not pos or not neg:
        return 0.5
    wins = 0.0
    for sp in pos:
        for sn in neg:
            if sp > sn:
                wins += 1.0
            elif sp == sn:
                wins += 0.5
    return wins / (len(pos) * len(neg))
