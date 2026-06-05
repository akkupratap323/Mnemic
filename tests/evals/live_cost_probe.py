"""
Copyright 2026, Abishek (Mnemic project).

Measure the REAL per-graph-write cost: run a few real episodes through the full
Graphiti pipeline (DeepSeek LLM + bge-m3 embeddings + Neo4j), metering actual
DeepSeek calls, tokens, and wall time. Then extrapolate with the measured
routing counts. Uses an ISOLATED group_id and cleans up after — does not touch
other graphs in the database.

Run:
    export DEEPSEEK_API_KEY=sk-...  NEO4J_PASSWORD=...
    python tests/evals/live_cost_probe.py --episodes 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import typing
from datetime import datetime, timezone

from mnemic import Mnemic
from mnemic.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from mnemic.hybrid.eval import load_longmemeval
from mnemic.llm_client.config import LLMConfig
from mnemic.llm_client.openai_generic_client import OpenAIGenericClient

GROUP = 'mnemic_costprobe'  # isolated namespace; cleaned up at the end
_DATA = os.path.join(
    os.path.dirname(__file__), 'data', 'longmemeval_data', 'longmemeval_oracle.json'
)
# DeepSeek deepseek-chat pricing (approx, USD per token)
_IN_PRICE = 0.27 / 1_000_000
_OUT_PRICE = 1.10 / 1_000_000


class MeteredDeepSeek(OpenAIGenericClient):
    """DeepSeek adapter: json_object mode + schema injected into the prompt so it
    returns Graphiti's exact field names. Plus metered API calls."""

    def __init__(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        super().__init__(*args, **kwargs)
        self.calls = 0
        self.in_tok = 0
        self.out_tok = 0
        self.secs = 0.0

    async def _generate_response(self, messages, response_model=None, max_tokens=4096, model_size=None):  # type: ignore[override]
        openai_messages: list[dict[str, str]] = []
        for m in messages:
            m.content = self._clean_input(m.content)
            if m.role in ('user', 'system'):
                openai_messages.append({'role': m.role, 'content': m.content})

        # DeepSeek has no json_schema; coax exact field names via json_object + schema in prompt
        if response_model is not None:
            schema = json.dumps(response_model.model_json_schema())
            openai_messages.append({
                'role': 'system',
                'content': (
                    'Respond with ONLY a single valid JSON object that strictly conforms to '
                    'this JSON Schema. Use EXACTLY the property names it defines (including all '
                    f'required keys); do not rename or omit them. JSON Schema: {schema}'
                ),
            })
        else:
            openai_messages.append({'role': 'system', 'content': 'Respond with ONLY a valid JSON object.'})

        self.calls += 1
        t0 = time.time()
        response = await self.client.chat.completions.create(
            model=self.model or 'deepseek-chat',
            messages=openai_messages,  # type: ignore[arg-type]
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={'type': 'json_object'},
        )
        self.secs += time.time() - t0
        usage = getattr(response, 'usage', None)
        if usage:
            self.in_tok += getattr(usage, 'prompt_tokens', 0) or 0
            self.out_tok += getattr(usage, 'completion_tokens', 0) or 0
        return json.loads(response.choices[0].message.content or '{}')


def _texts(n: int) -> list[str]:
    out: list[str] = []
    for inst in load_longmemeval(_DATA):
        for session in inst.sessions:
            for turn in session:
                if turn.role == 'user' and len(turn.content) > 40:
                    out.append(turn.content)
                    if len(out) >= n:
                        return out
    return out


async def run(n_episodes: int, model: str) -> None:
    llm = MeteredDeepSeek(
        config=LLMConfig(
            api_key=os.environ['DEEPSEEK_API_KEY'],
            base_url='https://api.deepseek.com',
            model=model,
            small_model=model,
        ),
        max_tokens=8000,
    )
    embedder = OpenAIEmbedder(
        config=OpenAIEmbedderConfig(
            api_key='ollama',
            base_url='http://localhost:11434/v1',
            embedding_model='bge-m3',
            embedding_dim=1024,
        )
    )
    engine = Mnemic(
        os.environ.get('NEO4J_URI', 'bolt://localhost:7687'),
        os.environ.get('NEO4J_USER', 'neo4j'),
        os.environ['NEO4J_PASSWORD'],
        llm_client=llm,
        embedder=embedder,
    )

    print('Building indices (idempotent, IF NOT EXISTS)...')
    await engine.build_indices_and_constraints()

    texts = _texts(n_episodes)
    print(f'Ingesting {len(texts)} episodes into isolated group {GROUP!r}...\n')
    per = []
    for i, text in enumerate(texts):
        c0, i0, o0 = llm.calls, llm.in_tok, llm.out_tok
        wall0 = time.time()
        try:
            await engine.remember(
                name=f'probe-{i}',
                episode_body=text,
                source_description='cost probe',
                reference_time=datetime.now(timezone.utc),
                group_id=GROUP,
            )
            d_calls = llm.calls - c0
            d_tok = (llm.in_tok - i0) + (llm.out_tok - o0)
            d_wall = time.time() - wall0
            per.append((d_calls, llm.in_tok - i0, llm.out_tok - o0, d_wall))
            print(f'  ep{i}: {d_calls} LLM calls, {d_tok} tokens, {d_wall:.1f}s ✅')
        except Exception as exc:  # noqa: BLE001
            print(f'  ep{i} FAILED after {llm.calls - c0} calls: {type(exc).__name__}: {str(exc)[:160]}')

    if per:
        n = len(per)
        avg_calls = sum(p[0] for p in per) / n
        avg_in = sum(p[1] for p in per) / n
        avg_out = sum(p[2] for p in per) / n
        avg_wall = sum(p[3] for p in per) / n
        avg_usd = avg_in * _IN_PRICE + avg_out * _OUT_PRICE
        print(f'\n=== REAL per-graph-write cost (avg over {n} episodes) ===')
        print(f'  LLM calls/write : {avg_calls:.1f}')
        print(f'  tokens/write    : {avg_in + avg_out:.0f}')
        print(f'  wall time/write : {avg_wall:.1f}s')
        print(f'  $/write         : ${avg_usd:.5f}')
        print('\n=== Extrapolated (real per-write x measured routing counts on 100 Qs) ===')
        for name, gw in (('B_full_graph', 3094), ('C_hybrid', 1261)):
            print(f'  {name:<14}: {gw} writes -> {gw * avg_calls:.0f} calls, '
                  f'${gw * avg_usd:.2f}, {gw * avg_wall / 60:.0f} min')

    print(f'\nCleaning up group {GROUP!r} (other graphs untouched)...')
    try:
        await engine.driver.execute_query(
            'MATCH (n) WHERE n.group_id = $gid DETACH DELETE n', gid=GROUP
        )
        print('  cleaned ✅')
    except Exception as exc:  # noqa: BLE001
        print(f'  cleanup note: {exc}')
    await engine.close()


def main() -> None:
    parser = argparse.ArgumentParser(description='Measure real per-graph-write cost')
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--model', default='deepseek-chat')
    args = parser.parse_args()
    asyncio.run(run(args.episodes, args.model))


if __name__ == '__main__':
    main()
