"""
Copyright 2026, Abishek (Mnemic project).

The 3-way proof harness: run the SAME questions through three memory
configurations and measure cost AND recall for each.

  A — vector-only   (everything cheap; the floor)
  B — full-graph    (everything expensive; "before" / Graphiti)
  C — hybrid        (the router decides; "after" / Mnemic)

Cost is modelled from routing decisions (graph writes are the expensive part);
recall is measured on the real retrieval path. Offline (no graph) this proves the
COST structure; with a live graph injected it proves quality too.
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from mnemic.hybrid.eval import EvalInstance
from mnemic.hybrid.fusion import GraphSearcher, normalize_text
from mnemic.hybrid.memory import GraphMemory, HybridMemory
from mnemic.hybrid.router import Router
from mnemic.hybrid.types import RouteDecision, Tier
from mnemic.hybrid.vector_store import InMemoryVectorStore

# offline: build a fresh (graph, searcher) per instance; None means no graph tier
GraphFactory = Callable[[], tuple[GraphMemory | None, GraphSearcher | None]]


@dataclass(frozen=True)
class AlwaysVectorRouter:
    """Config A: route everything to the cheap vector tier."""

    def route(self, content, metadata=None, *, embedding=None) -> RouteDecision:
        return RouteDecision(Tier.VECTOR_ONLY, ('config:vector_only',))


@dataclass(frozen=True)
class AlwaysGraphRouter:
    """Config B: route everything to the expensive graph tier."""

    def route(self, content, metadata=None, *, embedding=None) -> RouteDecision:
        return RouteDecision(Tier.VECTOR_AND_GRAPH, ('config:full_graph',))


@dataclass(frozen=True)
class CostBreakdown:
    llm_calls: int
    embed_calls: int
    est_usd: float
    est_seconds: float


@dataclass(frozen=True)
class CostModel:
    """Maps write counts to cost. Graph writes are the expensive part."""

    graph_llm_calls_per_write: int = 5
    usd_per_llm_call: float = 0.002
    usd_per_embed_call: float = 0.000002
    seconds_per_llm_call: float = 1.5
    seconds_per_embed_call: float = 0.02

    def cost(self, total_writes: int, graph_writes: int) -> CostBreakdown:
        llm_calls = graph_writes * self.graph_llm_calls_per_write
        embed_calls = total_writes  # every memory is embedded once
        return CostBreakdown(
            llm_calls=llm_calls,
            embed_calls=embed_calls,
            est_usd=llm_calls * self.usd_per_llm_call + embed_calls * self.usd_per_embed_call,
            est_seconds=llm_calls * self.seconds_per_llm_call
            + embed_calls * self.seconds_per_embed_call,
        )


@dataclass(frozen=True)
class ConfigResult:
    name: str
    total_writes: int
    graph_writes: int
    cost: CostBreakdown
    evidence_recall_at_k: float
    by_type: dict[str, float]

    @property
    def graph_fraction(self) -> float:
        return self.graph_writes / self.total_writes if self.total_writes else 0.0


@dataclass(frozen=True)
class BenchmarkReport:
    k: int
    results: tuple[ConfigResult, ...]

    def by_name(self, name: str) -> ConfigResult:
        for result in self.results:
            if result.name == name:
                return result
        raise KeyError(name)


async def _evaluate_config(
    instances: list[EvalInstance],
    embedder: object,
    router: Router,
    graph_factory: GraphFactory | None,
    k: int,
) -> ConfigResult:
    total_writes = 0
    graph_writes = 0
    hits = 0
    type_total: dict[str, int] = {}
    type_hits: dict[str, int] = {}

    for inst in instances:
        graph, searcher = graph_factory() if graph_factory else (None, None)
        memory = HybridMemory(
            embedder=embedder,  # type: ignore[arg-type]
            vector_store=InMemoryVectorStore(),
            router=router,
            graph=graph,
            graph_searcher=searcher,
        )
        evidence = set()
        for session in inst.sessions:
            for turn in session:
                result = await memory.remember(turn.content)
                total_writes += 1
                if result.decision.writes_graph:
                    graph_writes += 1
                if turn.is_evidence:
                    evidence.add(normalize_text(turn.content))

        recalled = await memory.recall_fused(inst.question, k=k) if inst.question else []
        got = any(normalize_text(h.content) in evidence for h in recalled)
        hits += int(got)
        type_total[inst.question_type] = type_total.get(inst.question_type, 0) + 1
        type_hits[inst.question_type] = type_hits.get(inst.question_type, 0) + int(got)

    n = len(instances)
    by_type = {t: type_hits[t] / type_total[t] for t in type_total}
    return ConfigResult(
        name='',  # set by caller
        total_writes=total_writes,
        graph_writes=graph_writes,
        cost=CostModel().cost(total_writes, graph_writes),
        evidence_recall_at_k=hits / n if n else 0.0,
        by_type=by_type,
    )


async def run_three_way(
    instances: list[EvalInstance],
    *,
    embedder: object,
    router: Router,
    graph_factory: GraphFactory | None = None,
    cost_model: CostModel | None = None,
    k: int = 10,
    limit: int | None = None,
) -> BenchmarkReport:
    """Run configs A (vector), B (full-graph), C (hybrid) and report cost + recall."""
    model = cost_model or CostModel()
    subset = instances[:limit] if limit is not None else instances
    configs: list[tuple[str, Router]] = [
        ('A_vector_only', AlwaysVectorRouter()),
        ('B_full_graph', AlwaysGraphRouter()),
        ('C_hybrid', router),
    ]
    results: list[ConfigResult] = []
    for name, rtr in configs:
        partial = await _evaluate_config(subset, embedder, rtr, graph_factory, k)
        results.append(
            ConfigResult(
                name=name,
                total_writes=partial.total_writes,
                graph_writes=partial.graph_writes,
                cost=model.cost(partial.total_writes, partial.graph_writes),
                evidence_recall_at_k=partial.evidence_recall_at_k,
                by_type=partial.by_type,
            )
        )
    return BenchmarkReport(k=k, results=tuple(results))
