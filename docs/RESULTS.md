# Mnemic — Measured Results

Real, measured numbers (not estimates). Reproduce with the scripts in
`tests/evals/`. Stack: DeepSeek (LLM/teacher) + BAAI/bge-m3 (embeddings, local
via Ollama) + Neo4j 5.26.

## Proof #1 — the teacher beats keyword routing
DeepSeek vs a 30-example hand-labeled gold set (`tests/evals/check_teacher.py --teacher api`):

| Labeler | Accuracy | Recall | Cohen's κ |
|---------|---------:|-------:|----------:|
| Keyword heuristic | 53.3% | 35.3% | 0.11 |
| **DeepSeek teacher** | **96.7%** | **100%** | **0.93** |

The keyword router misses 11 of 17 graph-worthy facts; the teacher misses none.

## Proof #2 — the learned router beats keyword routing
Learned router distilled from 200 real DeepSeek labels over bge-m3 embeddings,
scored on the human gold set (`tests/evals/prove_router.py --teacher api --embedder ollama`):

| Router | Accuracy | Recall | Cohen's κ |
|--------|---------:|-------:|----------:|
| Keyword | 53.3% | 35.3% | 0.11 |
| **Learned (distilled)** | **76.7%** | **100%** | **0.49** |

+23 points accuracy; recall on important facts 35% → 100%.

## Proof #3 — hybrid is cheaper (cost: measured)
Real per-graph-write cost, measured through the full live stack
(`tests/evals/live_cost_probe.py`): **2.0 LLM calls, 4,539 tokens, 5.0 s,
$0.00140 per write** (DeepSeek pricing, short single-turn episodes).

Routing measured on 100 LongMemEval questions (`tests/evals/benchmark_3way.py`):
full-graph writes everything (3,094); the hybrid router writes 1,261 (41%).

| Config | Graph writes | LLM calls | Cost | Time |
|--------|-------------:|----------:|-----:|-----:|
| B — full-graph ("before") | 3,094 | 6,188 | $4.34 | 260 min |
| **C — hybrid ("ours")** | 1,261 | 2,522 | **$1.77** | 106 min |

**Hybrid = 41% of full-graph cost — a 2.4× saving.** Ratio is model-independent
(it is the routing fraction); absolute $/time use the measured per-write cost.

## Pending
- **Proof #3 recall** (does hybrid keep full-graph's answer quality): needs the
  A/B/C search run with the live graph. Unblocked; not yet run.

## Caveats (read these)
- The 2-calls/write figure is for short conversational turns; richer documents
  cost more.
- DeepSeek has no `json_schema` mode; we coax field names via `json_object` +
  schema-in-prompt (`tests/evals/live_cost_probe.py`). Its extraction is a notch
  below OpenAI's (occasional dropped edges), but functional.
- Graphiti 0.29 requires Neo4j 5.26+ (5.20 fails on dynamic-label Cypher).
