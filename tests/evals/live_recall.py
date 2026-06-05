"""
Copyright 2026, Abishek (Mnemic project).

Proof #3 (recall): does the hybrid keep full-graph's answer quality?

For each question, ingest its history into the live graph and measure
evidence-recall@k for three configs, by question type:
  A — vector only        (config A)   <- vector recall from the full-graph pass
  B — full-graph         (config B)   <- fused recall, everything in the graph
  C — hybrid             (config C)   <- fused recall, router decides

Two passes per question (full-graph gives A+B; hybrid gives C). Uses isolated
group_ids and cleans up after each question. Checkpoints to /tmp.

Run:
    export DEEPSEEK_API_KEY=...  OPENAI_API_KEY=sk-x
    export NEO4J_URI=bolt://localhost:7688 NEO4J_PASSWORD=...
    python tests/evals/live_recall.py --limit 100
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))
from live_cost_probe import MeteredDeepSeek  # DeepSeek adapter (json_object + schema)

from mnemic import Mnemic
from mnemic.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from mnemic.hybrid.benchmark import AlwaysGraphRouter
from mnemic.hybrid.eval import load_longmemeval
from mnemic.hybrid.fusion import GraphFact, normalize_text
from mnemic.hybrid.memory import HybridMemory
from mnemic.hybrid.ollama_embedder import OllamaEmbedder
from mnemic.hybrid.router import WriteRouter
from mnemic.hybrid.vector_store import InMemoryVectorStore
from mnemic.llm_client.config import LLMConfig

_DATA = os.path.join(os.path.dirname(__file__), 'data', 'longmemeval_data', 'longmemeval_oracle.json')
_PROGRESS = '/tmp/live_recall_progress.json'
_RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'docs', 'results')
_CONFIGS = ('A_vector', 'B_full_graph', 'C_hybrid')


def _summary(tallies: dict) -> dict:
    out: dict = {'configs': {}, 'by_type': {}}
    for c in _CONFIGS:
        t = tallies[c]
        out['configs'][c] = {
            'recall': (t['hit'] / t['total']) if t['total'] else 0.0,
            'hit': t['hit'],
            'total': t['total'],
        }
    for qt in sorted({q for c in _CONFIGS for q in tallies[c]['by_type']}):
        out['by_type'][qt] = {
            c: (lambda bt: bt[0] / max(1, bt[1]))(tallies[c]['by_type'].get(qt, [0, 1]))
            for c in _CONFIGS
        }
    return out


def _write_artifact(tallies: dict, limit: int, k: int) -> str:
    os.makedirs(_RESULTS_DIR, exist_ok=True)
    artifact = {
        'benchmark': 'longmemeval_recall_3way',
        'run_at': datetime.now(timezone.utc).isoformat(),
        'limit': limit,
        'k': k,
        'stack': {'llm': 'deepseek-chat', 'embedder': 'bge-m3', 'graph': 'neo4j-5.26'},
        'metric': 'evidence-recall@k',
        'summary': _summary(tallies),
        'raw_tallies': tallies,
    }
    path = os.path.join(_RESULTS_DIR, f'live_recall_{limit}.json')
    with open(path, 'w') as fh:
        json.dump(artifact, fh, indent=2)
    print(f'\nArtifact written: {os.path.relpath(path)}')
    return path


class ScopedSearcher:
    """Searches the graph scoped to one question's group_id (no cross-contamination)."""

    def __init__(self, engine: object, gid: str) -> None:
        self.engine = engine
        self.gid = gid

    async def search(self, query: str, *, num_results: int = 10) -> list[GraphFact]:
        try:
            edges = await self.engine.search(query, group_ids=[self.gid], num_results=num_results)  # type: ignore[attr-defined]
        except Exception:
            return []
        return [GraphFact(id=str(getattr(e, 'uuid', '')), fact=str(getattr(e, 'fact', e))) for e in edges]


def _build_engine() -> Mnemic:
    llm = MeteredDeepSeek(
        config=LLMConfig(
            api_key=os.environ['DEEPSEEK_API_KEY'],
            base_url='https://api.deepseek.com',
            model='deepseek-chat',
            small_model='deepseek-chat',
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
        os.environ.get('NEO4J_USER', 'neo4j'),
        os.environ['NEO4J_PASSWORD'],
        llm_client=llm,
        embedder=embedder,
    )


async def _cleanup(engine: Mnemic, gid: str) -> None:
    with contextlib.suppress(Exception):
        await engine.driver.execute_query('MATCH (n) WHERE n.group_id = $g DETACH DELETE n', g=gid)


async def _ingest_and_recall(engine, hybrid_emb, inst, gid, router, k):
    mem = HybridMemory(
        embedder=hybrid_emb,
        vector_store=InMemoryVectorStore(),
        router=router,
        graph=engine,
        graph_searcher=ScopedSearcher(engine, gid),
    )
    evidence = set()
    for session in inst.sessions:
        for turn in session:
            await mem.remember(turn.content, group_id=gid)
            if turn.is_evidence:
                evidence.add(normalize_text(turn.content))
    vec = await mem.recall(inst.question, k=k) if inst.question else []
    fused = await mem.recall_fused(inst.question, k=k) if inst.question else []
    vec_hit = any(normalize_text(h.item.content) in evidence for h in vec)
    fused_hit = any(normalize_text(h.content) in evidence for h in fused)
    return vec_hit, fused_hit


def _record(tallies, cfg, qtype, hit):
    t = tallies[cfg]
    t['hit'] += int(hit)
    t['total'] += 1
    bt = t['by_type'].setdefault(qtype, [0, 0])
    bt[0] += int(hit)
    bt[1] += 1


async def run(limit: int, k: int) -> None:
    engine = _build_engine()
    await engine.build_indices_and_constraints()
    hybrid_emb = OllamaEmbedder(model='bge-m3')
    instances = load_longmemeval(_DATA)[:limit]
    tallies = {c: {'hit': 0, 'total': 0, 'by_type': {}} for c in _CONFIGS}

    for qi, inst in enumerate(instances):
        t0 = time.time()
        try:
            gid_b = f'mb_B_{qi}'
            a_hit, b_hit = await _ingest_and_recall(engine, hybrid_emb, inst, gid_b, AlwaysGraphRouter(), k)
            await _cleanup(engine, gid_b)

            gid_c = f'mb_C_{qi}'
            _, c_hit = await _ingest_and_recall(engine, hybrid_emb, inst, gid_c, WriteRouter(), k)
            await _cleanup(engine, gid_c)

            _record(tallies, 'A_vector', inst.question_type, a_hit)
            _record(tallies, 'B_full_graph', inst.question_type, b_hit)
            _record(tallies, 'C_hybrid', inst.question_type, c_hit)
            print(f'q{qi} [{inst.question_type}] A={int(a_hit)} B={int(b_hit)} C={int(c_hit)} '
                  f'({time.time() - t0:.0f}s)', flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f'q{qi} ERROR: {type(exc).__name__}: {str(exc)[:120]}', flush=True)
        with open(_PROGRESS, 'w') as fh:
            json.dump(tallies, fh)

    print('\n=== Live recall by config ===')
    for c in _CONFIGS:
        t = tallies[c]
        rate = t['hit'] / t['total'] if t['total'] else 0.0
        print(f'  {c:<14} {rate:.1%}  (n={t["total"]})')
    print('\n=== recall by question type (A / B / C) ===')
    qtypes = sorted({qt for c in _CONFIGS for qt in tallies[c]['by_type']})
    for qt in qtypes:
        cells = []
        for c in _CONFIGS:
            bt = tallies[c]['by_type'].get(qt, [0, 1])
            cells.append(f'{bt[0] / max(1, bt[1]):.0%}')
        print(f'  {qt:<24} A={cells[0]:>4} B={cells[1]:>4} C={cells[2]:>4}')
    _write_artifact(tallies, limit, k)
    await engine.close()


def main() -> None:
    parser = argparse.ArgumentParser(description='Proof #3 recall: A/B/C live')
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--capture', action='store_true',
                        help='write the committed artifact from the checkpoint, no run')
    args = parser.parse_args()
    if args.capture:
        with open(_PROGRESS) as fh:
            _write_artifact(json.load(fh), args.limit, args.k)
        return
    asyncio.run(run(args.limit, args.k))


if __name__ == '__main__':
    main()
