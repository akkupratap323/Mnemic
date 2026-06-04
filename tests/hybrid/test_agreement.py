"""Tests for the teacher/labeler agreement metrics."""

from __future__ import annotations

import json

import pytest

from mnemic.hybrid.agreement import (
    GoldExample,
    compute_agreement,
    load_gold,
    measure_agreement,
)
from mnemic.hybrid.errors import InvalidInput


def test_perfect_agreement():
    preds = [True, True, False, False]
    r = compute_agreement(preds, preds)
    assert r.accuracy == 1.0
    assert r.precision == 1.0
    assert r.recall == 1.0
    assert r.kappa == 1.0


def test_worked_example_metrics_and_kappa():
    gold = [True, True, True, True, True, False, False, False, False, False]
    preds = [True, True, True, True, False, False, False, False, False, True]
    r = compute_agreement(preds, gold)
    assert r.tp == 4 and r.fn == 1 and r.tn == 4 and r.fp == 1
    assert r.accuracy == 0.8
    assert r.precision == pytest.approx(0.8)
    assert r.recall == pytest.approx(0.8)
    assert r.f1 == pytest.approx(0.8)
    assert r.kappa == pytest.approx(0.6)  # (0.8 - 0.5) / (1 - 0.5)


def test_all_wrong():
    r = compute_agreement([False, False], [True, True])
    assert r.accuracy == 0.0
    assert r.recall == 0.0


def test_single_class_perfect_kappa_guard():
    # all gold True, all preds True -> chance agreement is 1 -> guarded to 1.0
    r = compute_agreement([True, True, True], [True, True, True])
    assert r.accuracy == 1.0
    assert r.kappa == 1.0


def test_single_class_all_wrong_kappa_zero():
    r = compute_agreement([False, False], [True, True])
    assert r.kappa == 0.0


def test_length_mismatch_rejected():
    with pytest.raises(InvalidInput):
        compute_agreement([True], [True, False])


def test_empty_rejected():
    with pytest.raises(InvalidInput):
        compute_agreement([], [])


async def test_measure_agreement_with_label_fn():
    gold = [GoldExample('a', True), GoldExample('b', False), GoldExample('c', True)]

    async def label_fn(text: str) -> bool:
        return text in ('a', 'c')  # perfect labeler

    r = await measure_agreement(label_fn, gold)
    assert r.accuracy == 1.0


async def test_measure_agreement_empty_rejected():
    async def label_fn(_text: str) -> bool:
        return True

    with pytest.raises(InvalidInput):
        await measure_agreement(label_fn, [])


def test_load_gold(tmp_path):
    path = tmp_path / 'gold.json'
    path.write_text(json.dumps([{'text': 'x', 'label': True}, {'text': 'y', 'label': False}]))
    gold = load_gold(str(path))
    assert len(gold) == 2
    assert gold[0].text == 'x' and gold[0].label is True
    assert gold[1].label is False


def test_repo_gold_set_is_well_formed():
    import os

    path = os.path.join(
        os.path.dirname(__file__), '..', 'evals', 'data', 'router_gold.json'
    )
    gold = load_gold(path)
    assert len(gold) >= 30
    # balanced enough to be a fair test
    positives = sum(g.label for g in gold)
    assert 0.3 < positives / len(gold) < 0.7
