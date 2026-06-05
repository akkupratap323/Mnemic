"""
Copyright 2026, Abishek (Mnemic project).

Shared prompt + response parsing for graph-worthiness teachers (local or API).
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

_PROMPT_TEMPLATE = """You label text for an AI agent's memory, which has two tiers:

- GRAPH: facts that relate entities, establish or change state over time, record \
a decision, or supersede/contradict an earlier fact. These benefit from a temporal \
knowledge graph.
- VECTOR: casual chatter, one-off trivia, greetings, questions, or standalone \
statements that do not connect to other facts or prior state.

Judge by structural need, not keywords. "let's just go with Postgres then" is a \
GRAPH decision even though it has no obvious keyword. "that's a tough decision in \
general" is VECTOR — it records no actual decision.

Classify the text. Answer with exactly one word: GRAPH or VECTOR.

Text: {text}
Answer:"""


def build_graph_worthiness_prompt(text: str) -> str:
    return _PROMPT_TEMPLATE.format(text=text.strip())


def parse_graph_worthy(response: str) -> bool:
    """Map a model response to a graph-worthiness label (conservative default)."""
    answer = response.strip().lower()
    if 'graph' in answer:
        return True
    if 'vector' in answer:
        return False
    return answer.startswith(('yes', 'true'))
