"""
Copyright 2026, Abishek (Mnemic project).

A local-LLM teacher for router distillation, via Ollama. Uses a local model
(e.g. Gemma 3 12B) to judge graph-worthiness, generating labels for free at
scale. Implements the TeacherLabeler protocol.

Setup:
    # install Ollama from https://ollama.com, then:
    ollama pull gemma3:12b

Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from mnemic.hybrid.errors import InvalidInput
from mnemic.hybrid.teacher_prompts import build_graph_worthiness_prompt, parse_graph_worthy

__all__ = ['OllamaTeacher', 'build_graph_worthiness_prompt', 'parse_graph_worthy']


@dataclass
class OllamaTeacher:
    """Judges graph-worthiness with a local Ollama model (default Gemma 3 12B).

    ``generate`` is injectable so the logic is unit-testable without a server.
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
