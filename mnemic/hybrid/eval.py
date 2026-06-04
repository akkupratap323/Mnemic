"""
Copyright 2026, Abishek (Mnemic project).

Evaluation harness for the hybrid memory layer against LongMemEval-style data.
Measures whether, after ingesting a question's conversation history, the memory
recalls the evidence turn (evidence-recall@k) and surfaces the gold answer
(answer-recall@k). Deterministic and API-free with the HashingEmbedder.
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from mnemic.hybrid.memory import HybridMemory

MemoryFactory = Callable[[], HybridMemory]


@dataclass(frozen=True)
class Turn:
    role: str
    content: str
    is_evidence: bool


@dataclass(frozen=True)
class EvalInstance:
    question_id: str
    question: str
    answer: str
    question_type: str
    sessions: tuple[tuple[Turn, ...], ...]


@dataclass(frozen=True)
class InstanceResult:
    question_id: str
    question_type: str
    ingested: int
    evidence_recalled: bool
    answer_recalled: bool


@dataclass(frozen=True)
class EvalReport:
    total: int
    k: int
    evidence_recall_at_k: float
    answer_recall_at_k: float
    by_type: dict[str, float]
    results: tuple[InstanceResult, ...]


def _is_evidence(turn: dict) -> bool:
    return str(turn.get('has_answer', 'False')).strip().lower() == 'true'


def load_longmemeval(path: str) -> list[EvalInstance]:
    """Parse a LongMemEval JSON file into EvalInstances."""
    with open(path) as fh:
        raw = json.load(fh)

    instances: list[EvalInstance] = []
    for inst in raw:
        sessions: list[tuple[Turn, ...]] = []
        for session in inst.get('haystack_sessions', []):
            turns = tuple(
                Turn(
                    role=str(turn.get('role', '')),
                    content=str(turn.get('content', '')),
                    is_evidence=_is_evidence(turn),
                )
                for turn in session
                if turn.get('content')
            )
            if turns:
                sessions.append(turns)
        instances.append(
            EvalInstance(
                question_id=str(inst.get('question_id', '')),
                question=str(inst.get('question', '')),
                answer=str(inst.get('answer', '')),
                question_type=str(inst.get('question_type', '')),
                sessions=tuple(sessions),
            )
        )
    return instances


async def evaluate_instance(
    make_memory: MemoryFactory, instance: EvalInstance, *, k: int = 10
) -> InstanceResult:
    """Ingest one instance's history into a fresh memory and probe recall."""
    memory = make_memory()
    ingested = 0
    for session_idx, session in enumerate(instance.sessions):
        for turn_idx, turn in enumerate(session):
            await memory.remember(
                turn.content,
                metadata={
                    'session': session_idx,
                    'turn': turn_idx,
                    'role': turn.role,
                    'is_evidence': turn.is_evidence,
                },
            )
            ingested += 1

    hits = await memory.recall(instance.question, k=k) if instance.question else []
    evidence_recalled = any(h.item.metadata.get('is_evidence') for h in hits)
    answer = instance.answer.lower()
    answer_recalled = bool(answer) and any(answer in h.item.content.lower() for h in hits)
    return InstanceResult(
        question_id=instance.question_id,
        question_type=instance.question_type,
        ingested=ingested,
        evidence_recalled=evidence_recalled,
        answer_recalled=answer_recalled,
    )


async def evaluate(
    instances: list[EvalInstance],
    make_memory: MemoryFactory,
    *,
    k: int = 10,
    limit: int | None = None,
) -> EvalReport:
    """Run the eval over (a slice of) the instances and aggregate metrics."""
    subset = instances[:limit] if limit is not None else instances
    results = [await evaluate_instance(make_memory, inst, k=k) for inst in subset]

    total = len(results)
    evidence = sum(r.evidence_recalled for r in results)
    answer = sum(r.answer_recalled for r in results)

    by_type: dict[str, float] = {}
    type_counts: dict[str, int] = {}
    type_hits: dict[str, int] = {}
    for r in results:
        type_counts[r.question_type] = type_counts.get(r.question_type, 0) + 1
        type_hits[r.question_type] = type_hits.get(r.question_type, 0) + int(r.evidence_recalled)
    for qtype, count in type_counts.items():
        by_type[qtype] = type_hits[qtype] / count

    return EvalReport(
        total=total,
        k=k,
        evidence_recall_at_k=evidence / total if total else 0.0,
        answer_recall_at_k=answer / total if total else 0.0,
        by_type=by_type,
        results=tuple(results),
    )
