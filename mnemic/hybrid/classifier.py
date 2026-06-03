"""
Copyright 2026, Abishek (Mnemic project).

Heuristic signal classifier: decides whether a memory is worth the
expensive graph-extraction tier, or should stay in the cheap vector tier.
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from mnemic.hybrid.types import RouteDecision, Tier

_TOKEN_RE = re.compile(r'[a-z0-9_]+')
_TRUE_STRINGS = frozenset({'1', 'true', 'yes', 'on'})

DEFAULT_SIGNAL_KEYWORDS: frozenset[str] = frozenset({
    'remember', 'important', 'decision', 'decided', 'decide', 'note', 'todo', 'fixme',
    'bug', 'fixed', 'broken', 'deprecated', 'breaking', 'regression',
    'config', 'credential', 'secret', 'api', 'endpoint', 'schema', 'migration',
    'requirement', 'preference', 'prefer', 'always', 'never', 'rule', 'convention',
})
DEFAULT_PHRASE_MARKERS: frozenset[str] = frozenset({
    'remember this', 'make a note', 'keep in mind', 'for the record', 'take note',
})
DEFAULT_GRAPH_TYPES: frozenset[str] = frozenset({
    'decision', 'bug', 'task', 'requirement', 'preference', 'fact', 'entity',
})
DEFAULT_IMPORTANCE_THRESHOLD = 0.7


@dataclass(frozen=True)
class SignalClassifier:
    """Pure, deterministic classifier. No I/O, no LLM calls."""

    keywords: frozenset[str] = DEFAULT_SIGNAL_KEYWORDS
    phrase_markers: frozenset[str] = DEFAULT_PHRASE_MARKERS
    graph_types: frozenset[str] = DEFAULT_GRAPH_TYPES
    importance_threshold: float = DEFAULT_IMPORTANCE_THRESHOLD

    def classify(
        self, content: str, metadata: Mapping[str, object] | None = None
    ) -> RouteDecision:
        meta = metadata or {}

        # explicit caller override always wins
        if _truthy(meta.get('force_vector_only')):
            return RouteDecision(Tier.VECTOR_ONLY, ('metadata:force_vector_only',))

        reasons: list[str] = []

        if _truthy(meta.get('force_graph')):
            reasons.append('metadata:force_graph')

        mtype = meta.get('type')
        if isinstance(mtype, str) and mtype.lower() in self.graph_types:
            reasons.append(f'type:{mtype.lower()}')

        importance = _as_float(meta.get('importance'))
        if importance is not None and importance >= self.importance_threshold:
            reasons.append('importance')

        text = (content or '').lower()
        tokens = set(_TOKEN_RE.findall(text))
        reasons.extend(f'keyword:{kw}' for kw in sorted(self.keywords & tokens))
        reasons.extend(
            f'phrase:{phrase}' for phrase in sorted(self.phrase_markers) if phrase in text
        )

        tier = Tier.VECTOR_AND_GRAPH if reasons else Tier.VECTOR_ONLY
        return RouteDecision(tier, tuple(reasons))


def _truthy(value: object) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return value.strip().lower() in _TRUE_STRINGS
    return False


def _as_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None
