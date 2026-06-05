"""
Copyright 2026, Abishek (Mnemic project).

Proof #3 (the real one): ANSWER QUALITY vs COST, hybrid vs full-graph.

For each question, retrieve context three ways, have DeepSeek answer FROM that
context, and have DeepSeek judge the answer against the gold answer. This is the
fair test: not "was the evidence retrieved" (saturated) but "can it answer."

  A — vector only     (cheap)
  B — full-graph      (expensive; everything in the graph)
  C — hybrid          (router decides)

Reports answer-accuracy AND graph writes ($) per config -> the quality/cost curve.

Run:
    export DEEPSEEK_API_KEY=...  OPENAI_API_KEY=sk-x
    export NEO4J_URI=bolt://localhost:7688 NEO4J_PASSWORD=mnemic_bench_pw
    python tests/evals/answer_quality.py --limit 5
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))
from live_cost_probe import MeteredDeepSeek
from live_recall import ScopedSearcher
from openai import AsyncOpenAI

from mnemic import Mnemic
from mnemic.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from mnemic.hybrid.benchmark import AlwaysGraphRouter
from mnemic.hybrid.eval import load_longmemeval
from mnemic.hybrid.memory import HybridMemory
from mnemic.hybrid.ollama_embedder import OllamaEmbedder
from mnemic.hybrid.router import WriteRouter
from mnemic.hybrid.vector_store import InMemoryVectorStore
from mnemic.llm_client.config import LLMConfig

_DATA = os.path.join(os.path.dirname(__file__), 'data', 'longmemeval_data', 'longmemeval_oracle.json')
_RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'docs', 'results')
_USD_PER_WRITE = 0.00140  # measured (live_cost_probe)
_CONFIGS = ('A_vector', 'B_full_graph', 'C_hybrid')


def _ds() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=os.environ['DEEPSEEK_API_KEY'], base_url='https://api.deepseek.com')


async def _gen_answer(ds: AsyncOpenAI, question: str, context: str) -> str:
    prompt = (
        'Answer the QUESTION using ONLY the CONTEXT. Be concise. '
        "If the context does not contain the answer, reply 'UNKNOWN'.\n\n"
        f'CONTEXT:\n{context}\n\nQUESTION: {question}\nANSWER:'
    )
    r = await ds.chat.completions.create(
        model='deepseek-chat', messages=[{'role': 'user', 'content': prompt}],
        temperature=0, max_tokens=150,
    )
    return (r.choices[0].message.content or '').strip()


async def _judge(ds: AsyncOpenAI, question: str, gold: str, candidate: str) -> bool:
    prompt = (
        'Grade the candidate answer against the gold answer.\n'
        f'QUESTION: {question}\nGOLD ANSWER: {gold}\nCANDIDATE ANSWER: {candidate}\n'
        'Does the candidate convey the same key information as the gold answer? '
        'Reply with exactly one word: CORRECT or INCORRECT.'
    )
    r = await ds.chat.completions.create(
        model='deepseek-chat', messages=[{'role': 'user', 'content': prompt}],
        temperature=0, max_tokens=4,
    )
    out = (r.choices[0].message.content or '').strip().lower()
    return out.startswith('correct')


def _build_engine() -> Mnemic:
    llm = MeteredDeepSeek(
        config=LLMConfig(
            api_key=os.environ['DEEPSEEK_API_KEY'], base_url='https://api.deepseek.com',
            model='deepseek-chat', small_model='deepseek-chat',
        ),
        max_tokens=8000,
    )
    embedder = OpenAIEmbedder(
        config=OpenAIEmbedderConfig(
            api_key='ollama', base_url='http://localhost:11434/v1',
            embedding_model='bge-m3', embedding_dim=1024,
        )
    )
    return Mnemic(
        os.environ.get('NEO4J_URI', 'bolt://localhost:7688'),
        os.environ.get('NEO4J_USER', 'neo4j'), os.environ['NEO4J_PASSWORD'],
        llm_client=llm, embedder=embedder,
    )


async def _cleanup(engine, gid):
    with contextlib.suppress(Exception):
        await engine.driver.execute_query('MATCH (n) WHERE n.group_id = $g DETACH DELETE n', g=gid)


async def _ingest(engine, hybrid_emb, inst, gid, router):
    mem = HybridMemory(
        embedder=hybrid_emb, vector_store=InMemoryVectorStore(), router=router,
        graph=engine, graph_searcher=ScopedSearcher(engine, gid),
    )
    graph_writes = 0
    for session in inst.sessions:
        for turn in session:
            res = await mem.remember(turn.content, group_id=gid)
            graph_writes += int(res.decision.writes_graph)
    return mem, graph_writes


async def run(limit: int, k: int) -> None:
    ds = _ds()
    engine = _build_engine()
    await engine.build_indices_and_constraints()
    hybrid_emb = OllamaEmbedder(model='bge-m3')
    instances = load_longmemeval(_DATA)[:limit]
    acc = {c: 0 for c in _CONFIGS}
    writes = {c: 0 for c in _CONFIGS}
    n = 0

    for qi, inst in enumerate(instances):
        try:
            gid_b = f'aq_B_{qi}'
            mem_b, gw_b = await _ingest(engine, hybrid_emb, inst, gid_b, AlwaysGraphRouter())
            a_ctx = '\n'.join(h.item.content for h in await mem_b.recall(inst.question, k=k))
            b_ctx = '\n'.join(h.content for h in await mem_b.recall_fused(inst.question, k=k))
            a_ok = await _judge(ds, inst.question, inst.answer, await _gen_answer(ds, inst.question, a_ctx))
            b_ok = await _judge(ds, inst.question, inst.answer, await _gen_answer(ds, inst.question, b_ctx))
            await _cleanup(engine, gid_b)

            gid_c = f'aq_C_{qi}'
            mem_c, gw_c = await _ingest(engine, hybrid_emb, inst, gid_c, WriteRouter())
            c_ctx = '\n'.join(h.content for h in await mem_c.recall_fused(inst.question, k=k))
            c_ok = await _judge(ds, inst.question, inst.answer, await _gen_answer(ds, inst.question, c_ctx))
            await _cleanup(engine, gid_c)

            acc['A_vector'] += int(a_ok)
            acc['B_full_graph'] += int(b_ok)
            acc['C_hybrid'] += int(c_ok)
            writes['B_full_graph'] += gw_b
            writes['C_hybrid'] += gw_c
            n += 1
            print(f'q{qi} answer-correct  A={int(a_ok)} B={int(b_ok)} C={int(c_ok)} '
                  f'(graph writes B={gw_b} C={gw_c})', flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f'q{qi} ERROR: {type(exc).__name__}: {str(exc)[:120]}', flush=True)

    print('\n=== Answer quality vs cost ===')
    print(f'{"config":<14}{"answer_acc":>12}{"graph_writes":>14}{"est_$":>10}')
    for c in _CONFIGS:
        a = acc[c] / n if n else 0.0
        print(f'{c:<14}{a:>11.0%}{writes[c]:>14}{writes[c] * _USD_PER_WRITE:>10.3f}')
    os.makedirs(_RESULTS_DIR, exist_ok=True)
    artifact = {
        'benchmark': 'longmemeval_answer_quality',
        'run_at': datetime.now(timezone.utc).isoformat(),
        'limit': n, 'k': k, 'metric': 'llm-judged answer accuracy',
        'stack': {'llm': 'deepseek-chat', 'embedder': 'bge-m3', 'graph': 'neo4j-5.26'},
        'accuracy': {c: (acc[c] / n if n else 0.0) for c in _CONFIGS},
        'graph_writes': writes,
        'usd': {c: writes[c] * _USD_PER_WRITE for c in _CONFIGS},
    }
    with open(os.path.join(_RESULTS_DIR, f'answer_quality_{n}.json'), 'w') as fh:
        json.dump(artifact, fh, indent=2)
    print(f'\nArtifact: docs/results/answer_quality_{n}.json')
    await engine.close()


def main() -> None:
    parser = argparse.ArgumentParser(description='Answer quality vs cost: A/B/C')
    parser.add_argument('--limit', type=int, default=5)
    parser.add_argument('--k', type=int, default=10)
    args = parser.parse_args()
    asyncio.run(run(args.limit, args.k))


if __name__ == '__main__':
    main()
