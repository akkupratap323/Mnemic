"""
Copyright 2026, Abishek (Mnemic project).

A local-LLM teacher for router distillation, via Ollama. Uses a local model
(e.g. Gemma 3 12B) to judge graph-worthiness, generating LLM-quality labels for
free at scale. Implements the TeacherLabeler protocol, so it drops into
build_training_set(teacher=...).

Setup:
    # install Ollama from https://ollama.com, then:
    ollama pull gemma3:12b

Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from mnemic.hybrid.errors import InvalidInput

_PROMPT_TEMPLATE = """You label text for an AI agent's memory, which has two tiers:

- GRAPH: facts that relate entities, establish or change state over time, record \
a decision, or supersede/contradict an earlier fact. These benefit from a temporal \
knowledge graph.
- VECTOR: casual chatter, one-off trivia, greetings, or standalone statements that \
do not connect to other facts or prior state.

Judge by structural need, not keywords. "let's just go with Postgres then" is a \
GRAPH decision even though it has no obvious keyword.

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


@dataclass
class OllamaTeacher:
    """Judges graph-worthiness with a local Ollama model (default Gemma 3 12B).

    ``generate`` is injectable so the prompt/parse logic is unit-testable without a
    running server; left None it calls Ollama over HTTP.
    """

    model: str = 'gemma3:12b'
    host: str = 'http://localhost:11434'
    temperature: float = 0.0
    timeout: float = 60.0
    generate: Callable[[str], Awaitable[str]] | None = None

    async def label(self, text: str) -> bool:
        if not isinstance(text, str) or not text.strip():
            raise InvalidInput('text to label must be a non-empty string')
        generate = self.generate or self._ollama_generate
        raw = await generate(build_graph_worthiness_prompt(text))
        return parse_graph_worthy(raw)

    async def _ollama_generate(self, prompt: str) -> str:  # pragma: no cover - network I/O
        import httpx

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f'{self.host}/api/generate',
                json={
                    'model': self.model,
                    'prompt': prompt,
                    'stream': False,
                    'options': {'temperature': self.temperature},
                },
            )
            resp.raise_for_status()
            return str(resp.json().get('response', ''))
