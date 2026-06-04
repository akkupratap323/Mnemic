"""Tests for the eval harness and the offline HashingEmbedder."""

from __future__ import annotations

import json

import pytest

from mnemic.hybrid.errors import InvalidInput
from mnemic.hybrid.eval import (
    EvalInstance,
    Turn,
    evaluate,
    evaluate_instance,
    load_longmemeval,
)
from mnemic.hybrid.hashing_embedder import HashingEmbedder
from mnemic.hybrid.memory import HybridMemory
from mnemic.hybrid.vector_store import InMemoryVectorStore

# ---- HashingEmbedder ----

async def test_hashing_embedder_is_deterministic():
    e = HashingEmbedder(dim=64)
    assert await e.embed('hello world') == await e.embed('hello world')


async def test_hashing_embedder_similar_text_closer_than_unrelated():
    e = HashingEmbedder(dim=128)

    def dot(a, b):
        return sum(x * y for x, y in zip(a, b, strict=True))

    base = await e.embed('the capital of france is paris')
    near = await e.embed('what is the capital of france')
    far = await e.embed('my favorite pizza topping is mushrooms')
    assert dot(base, near) > dot(base, far)


async def test_hashing_embedder_rejects_empty():
    with pytest.raises(InvalidInput):
        await HashingEmbedder().embed('  ')


def test_hashing_embedder_rejects_bad_dim():
    with pytest.raises(InvalidInput):
        HashingEmbedder(dim=0)


async def test_hashing_embedder_handles_token_less_text():
    # non-whitespace but no [a-z0-9] tokens -> safe unit-vector fallback (no zero norm)
    v = await HashingEmbedder(dim=4).embed('!!! ???')
    assert v == (1.0, 0.0, 0.0, 0.0)


# ---- load_longmemeval ----

def test_load_parses_instances(tmp_path):
    raw = [
        {
            'question_id': 'q1',
            'question_type': 'single-session',
            'question': 'where did I park?',
            'answer': 'level 3',
            'haystack_sessions': [
                [
                    {'role': 'user', 'content': 'I parked on level 3', 'has_answer': 'True'},
                    {'role': 'assistant', 'content': 'got it', 'has_answer': 'False'},
                    {'role': 'user', 'content': ''},  # empty -> skipped
                ]
            ],
        }
    ]
    path = tmp_path / 'lme.json'
    path.write_text(json.dumps(raw))
    instances = load_longmemeval(str(path))
    assert len(instances) == 1
    inst = instances[0]
    assert inst.question_id == 'q1'
    assert len(inst.sessions[0]) == 2  # empty turn skipped
    assert inst.sessions[0][0].is_evidence is True
    assert inst.sessions[0][1].is_evidence is False


# ---- evaluation ----

def make_memory_factory():
    embedder = HashingEmbedder(dim=256)
    return lambda: HybridMemory(embedder=embedder, vector_store=InMemoryVectorStore())


def _instance(qid='q', answer='paris') -> EvalInstance:
    return EvalInstance(
        question_id=qid,
        question='what is the capital of france',
        answer=answer,
        question_type='single-session',
        sessions=(
            (
                Turn('user', 'we talked about the weather yesterday', False),
                Turn('user', 'the capital of france is paris', True),
                Turn('user', 'i had pasta for dinner', False),
            ),
        ),
    )


async def test_evaluate_instance_recalls_evidence_and_answer():
    result = await evaluate_instance(make_memory_factory(), _instance(), k=2)
    assert result.ingested == 3
    assert result.evidence_recalled is True
    assert result.answer_recalled is True


async def test_evaluate_instance_no_answer_string():
    result = await evaluate_instance(make_memory_factory(), _instance(answer=''), k=2)
    assert result.answer_recalled is False


async def test_evaluate_aggregates_and_by_type():
    instances = [_instance('a'), _instance('b')]
    report = await evaluate(instances, make_memory_factory(), k=2)
    assert report.total == 2
    assert report.evidence_recall_at_k == 1.0
    assert report.k == 2
    assert report.by_type['single-session'] == 1.0


async def test_evaluate_respects_limit():
    instances = [_instance('a'), _instance('b'), _instance('c')]
    report = await evaluate(instances, make_memory_factory(), k=2, limit=1)
    assert report.total == 1


async def test_evaluate_empty_instances():
    report = await evaluate([], make_memory_factory(), k=2)
    assert report.total == 0
    assert report.evidence_recall_at_k == 0.0
    assert report.answer_recall_at_k == 0.0
