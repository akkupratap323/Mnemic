"""Tests for the router distillation + self-correction pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from mnemic.hybrid.errors import InvalidInput
from mnemic.hybrid.hashing_embedder import HashingEmbedder
from mnemic.hybrid.learned_router import LearnedRouter
from mnemic.hybrid.router_training import (
    HeuristicLabeler,
    LabeledExample,
    add_failure_labels,
    build_training_set,
    evaluate_router,
    train_logistic,
)
from mnemic.hybrid.types import Tier

# clearly separable synthetic corpus: positives share 'relational', negatives share 'chatter'
POS = [f'relational decision number {i} links entities' for i in range(8)]
NEG = [f'random chatter number {i} about lunch and weather' for i in range(8)]


def _examples() -> list[LabeledExample]:
    return [LabeledExample(t, True) for t in POS] + [LabeledExample(t, False) for t in NEG]


# ---- labelers ----

def test_heuristic_labeler_matches_classifier():
    labeler = HeuristicLabeler()
    assert labeler.label('we made a decision') is True
    assert labeler.label('lovely weather today') is False


async def test_build_training_set_uses_heuristic_by_default():
    examples = await build_training_set(['we found a bug', 'nice sunset'])
    assert examples[0].label is True
    assert examples[1].label is False


async def test_build_training_set_refines_with_teacher():
    @dataclass
    class Teacher:
        async def label(self, text: str) -> bool:
            return 'override' in text

    # index 1 ('plain') would be heuristic-False, but teacher forces True via keyword
    examples = await build_training_set(
        ['nice day', 'override this plain text'],
        teacher=Teacher(),
        teacher_indices=[1],
    )
    assert examples[1].label is True
    assert examples[1].weight == 2.0


# ---- training ----

async def test_train_logistic_learns_separable_data():
    emb = HashingEmbedder(dim=64)
    router = await train_logistic(_examples(), emb, epochs=400, lr=0.5)
    assert isinstance(router, LearnedRouter)

    metrics = await evaluate_router(router, _examples(), emb)
    assert metrics['precision'] == 1.0
    assert metrics['recall'] == 1.0
    assert metrics['auc'] == 1.0


async def test_trained_router_routes_fresh_examples():
    emb = HashingEmbedder(dim=64)
    router = await train_logistic(_examples(), emb, epochs=400, lr=0.5)
    pos_emb = await emb.embed('relational decision number 99 links entities')
    neg_emb = await emb.embed('random chatter number 99 about lunch and weather')
    assert router.route('p', embedding=pos_emb).tier is Tier.VECTOR_AND_GRAPH
    assert router.route('n', embedding=neg_emb).tier is Tier.VECTOR_ONLY


async def test_train_logistic_rejects_empty():
    with pytest.raises(InvalidInput):
        await train_logistic([], HashingEmbedder())


async def test_train_logistic_rejects_bad_epochs():
    with pytest.raises(InvalidInput):
        await train_logistic(_examples(), HashingEmbedder(), epochs=0)


# ---- self-correction loop ----

def test_add_failure_labels_appends_positives_immutably():
    base = _examples()
    augmented = add_failure_labels(base, ['a missed important fact'], weight=3.0)
    assert len(augmented) == len(base) + 1
    assert len(base) == 16  # original untouched (immutable)
    assert augmented[-1].label is True
    assert augmented[-1].weight == 3.0


async def test_self_correction_fixes_a_routing_miss():
    emb = HashingEmbedder(dim=64)
    # a fact with no positive keywords -> heuristic would cheap-tier it
    missed = 'lets just go with sqlite then'
    base = await build_training_set(POS + NEG + [missed])
    router_before = await train_logistic(base, emb, epochs=400, lr=0.5)
    miss_emb = await emb.embed(missed)
    score_before = router_before.score_embedding(miss_emb)

    # feed the retrieval failure back and retrain
    corrected = add_failure_labels(base, [missed], weight=5.0)
    router_after = await train_logistic(corrected, emb, epochs=400, lr=0.5)
    score_after = router_after.score_embedding(miss_emb)

    # after self-correction the missed fact scores more graph-worthy
    assert score_after > score_before


# ---- evaluation metrics ----

async def test_evaluate_router_reports_metrics():
    emb = HashingEmbedder(dim=64)
    router = await train_logistic(_examples(), emb, epochs=400, lr=0.5)
    metrics = await evaluate_router(router, _examples(), emb)
    assert set(metrics) == {'precision', 'recall', 'f1', 'auc', 'n'}
    assert metrics['n'] == 16.0
    assert 0.0 <= metrics['auc'] <= 1.0


async def test_auc_degenerate_single_class_is_half():
    emb = HashingEmbedder(dim=8)
    router = LearnedRouter(weights=tuple([0.0] * 8))  # constant scores
    metrics = await evaluate_router(
        router, [LabeledExample('a', True), LabeledExample('b', True)], emb
    )
    assert metrics['auc'] == 0.5  # no negatives -> undefined -> 0.5


async def test_auc_handles_score_ties():
    emb = HashingEmbedder(dim=8)
    router = LearnedRouter(weights=tuple([0.0] * 8))  # every score == 0.5 -> ties
    metrics = await evaluate_router(
        router, [LabeledExample('x', True), LabeledExample('x', False)], emb
    )
    assert metrics['auc'] == 0.5  # one pos, one neg, tied score -> 0.5
