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

## Proof #3 — recall (measured, 15 questions, live stack)
Live A/B/C evidence-recall@10 on DeepSeek + bge-m3 + Neo4j 5.26
(`tests/evals/live_recall.py`; artifact: `docs/results/live_recall_15.json`):

| Config | evidence-recall@10 |
|--------|-------------------:|
| A — vector only | 100% |
| B — full-graph | 100% |
| C — hybrid | 100% |

(All 15 are `temporal-reasoning` — LongMemEval's first instances.)

**Hybrid recall = full-graph recall (100% = 100%)** — confirming the headline:
*same recall, 41% of the cost.*

**Honest caveat — the metric is saturated.** Vector-only also scores 100%, so
evidence-recall@k does **not** isolate the graph's unique value here: with a
strong embedder (bge-m3) the cheap tier already retrieves the evidence. Two
takeaways: (1) the hybrid keeps full-graph recall at lower cost (proven); (2) on
*this* metric/question-set the expensive graph adds no measurable recall — its
real edge (multi-hop synthesis, contradiction handling) needs an LLM-judged
answer-accuracy metric to show. We report what we measured, not what we hoped.

## Proof #3 — answer accuracy (the harder, fairer test)
LLM-judged answer accuracy, 8 questions (all temporal-reasoning), live stack
(`tests/evals/answer_quality.py`; artifact `docs/results/answer_quality_8.json`):

| Config | answer accuracy | graph writes | est_$ |
|--------|----------------:|-------------:|------:|
| A — vector only | 25% | 0 | $0.00 |
| B — full-graph | 25% | 197 | $0.28 |
| C — hybrid | 12% | 90 | $0.13 |

**This does NOT support "hybrid keeps full-graph quality" — it argues against it.**
- Full-graph did not beat vector (both 25%); on some questions the graph's facts
  added noise that *hurt* the answer (e.g. q2: vector correct, graph wrong).
- Hybrid scored lowest (12%): at the default routing threshold it lost quality
  vs both A and B.
- Overall accuracy is low (12-25%) — temporal-reasoning is hard and the simple
  retrieve->answer->judge pipeline (k=10) is weak. Small, noisy sample (n=8).

**Honest standing of the claim:** the cost saving (2.4x fewer graph writes) is
real and measured; **quality parity is NOT demonstrated** and this test shows the
opposite. A quality claim would require a larger type-diverse sample, a stronger
answer pipeline, and router-threshold tuning to trace the cost/quality frontier.

## Caveats (read these)
- The 2-calls/write figure is for short conversational turns; richer documents
  cost more.
- DeepSeek has no `json_schema` mode; we coax field names via `json_object` +
  schema-in-prompt (`tests/evals/live_cost_probe.py`). Its extraction is a notch
  below OpenAI's (occasional dropped edges), but functional.
- Graphiti 0.29 requires Neo4j 5.26+ (5.20 fails on dynamic-label Cypher).
