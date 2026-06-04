# A Self-Correcting Learned Router for Cost-Aware Agent Memory

*Research protocol for the Mnemic hybrid memory layer. This document defines the
claim, method, and experiments to a standard that supports a publishable result.
Status: method implemented (`mnemic/hybrid/learned_router.py`,
`mnemic/hybrid/router_training.py`); experiments pending.*

## 1. Problem

Temporal-knowledge-graph memory (Graphiti, Zep) gives high-quality, relational,
time-aware recall — but every write costs **4–6 LLM calls** (entity extraction,
deduplication, edge resolution, invalidation, summarization). At agent scale this
is prohibitive. The practical question is **not** "graph or vector" but:

> *Which writes actually need the expensive graph, and which can be served by a
> cheap vector store with no measurable loss in downstream recall?*

A good router answers this per write, near-free, and is **right** — not merely
cheap. The naive keyword heuristic is high-precision but low-recall: it catches
"we decided" but misses semantically-equivalent, unmarked facts ("let's just go
with Postgres then"), silently leaking relational facts into the cheap tier.

## 2. Claim

A **learned, self-correcting router** routes writes between tiers such that the
hybrid system achieves **recall within ε of the all-graph system at a fraction of
its write cost**, and **improves over time** by training on its own retrieval
failures — outperforming both keyword routing and static learned routing.

## 3. Method

### 3.1 Learned router (distillation)
A linear logistic model over the embedding **already computed for the vector tier**
emits a graph-worthiness score `σ(w·x + b) ∈ [0,1]`; route to graph iff
`score ≥ τ`. Inference cost is one dot product — free.

Labels come from **distillation**: the cheap `HeuristicLabeler` weak-labels all
writes; an expensive `TeacherLabeler` (LLM judge) refines a sampled subset. The
student is trained on the union (teacher labels up-weighted). This buys
LLM-quality routing at heuristic-cost inference.

### 3.2 Self-correcting loop (the novel contribution)
The benchmark yields *labeled routing failures for free*: any query that fails
because the needed fact was cheap-tiered is a **false negative** the router should
have sent to the graph. `add_failure_labels` injects these as high-weight
positives and the router is retrained. The router thus **learns from its own
retrieval misses** — a closed loop no keyword scheme can express.

### 3.3 Salience, not frequency (consolidation fix)
Promotion must not rely on access count alone (feedback loop: graph items surface
more → counted more → kept; a once-referenced critical decision never promotes).
We derive **salience** from the graph's own machinery: an edge that was
*invalidated/superseded* (Graphiti already computes `invalid_at`) marks a memory
as important regardless of recall frequency. Salience + frequency drives promotion.

## 4. Experimental protocol

### 4.1 Datasets
- **LongMemEval** (in-repo) and **LOCOMO** for multi-session/temporal questions.
- **Critical:** report **segmented by question type** — single-hop, multi-hop,
  temporal, knowledge-update, aggregation. The graph only earns its cost on the
  relational/temporal subset; an unsegmented average hides this and mis-tunes the
  router. This is a load-bearing methodological requirement.

### 4.2 Conditions (3-way, identical data/embedder)
| Condition | Routing | Expected |
|-----------|---------|----------|
| **A — Vector-only** | everything cheap | floor recall, floor cost |
| **B — Full-graph** | everything graph (Graphiti) | ceiling recall, ceiling cost |
| **C — Hybrid** | learned router | near-B recall, near-A cost |

Ablations of C: keyword router · static learned router · **self-correcting**
learned router (the contribution).

### 4.3 Metrics
- **Quality:** evidence-recall@k and LLM-judged answer accuracy (official
  LongMemEval metric), **per query type**.
- **Cost:** LLM calls, tokens, $ and write latency (instrument the LLM client).
- **Routing quality:** precision/recall/F1/AUC of graph-worthiness vs. an
  oracle label set (`evaluate_router`).
- **Headline:** the **cost-vs-recall curve** — C should dominate the A–B line.

### 4.4 Success criterion
C reaches **≥ (B_recall − ε)** at **≤ α·B_cost** (target ε ≤ 2pts, α ≤ 0.3), and
the self-correcting ablation strictly improves routing recall over rounds.

## 5. Threats to validity
- **Embedding ceiling:** structural need (relational/contradicting) is only
  partly visible in surface form; the learned router *predicts* it. The async
  consolidation path (cheap LLM pass with cross-doc context) *detects* it and is
  the upper bound the router is distilled toward.
- **Eval vs. prod failure signal:** failure labels are ground-truth in eval; in
  production a proxy is needed (re-ask, later contradiction, user correction).
- **Benchmark composition:** a lookup-heavy eval under-credits the graph (§4.1).

## 6. Novelty
Cost-aware routing exists; distillation exists. The novel, defensible combination
is: **a router that self-corrects from retrieval failures**, **salience derived
from temporal-edge invalidation** rather than access frequency, and a
**query-type-segmented cost-vs-recall evaluation** that makes the trade-off
honest.

## 7. Reproducibility
Training is deterministic (zero-init logistic regression). The offline
`HashingEmbedder` lets the full pipeline run with no API keys; swap in a real
embedder for headline numbers. All components are unit-tested at 100% coverage.
