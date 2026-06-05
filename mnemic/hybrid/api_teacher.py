"""
Copyright 2026, Abishek (Mnemic project).

An API-based teacher for router distillation. Works with any OpenAI-compatible
chat endpoint — DeepSeek (default), OpenAI, Groq, or a local server. Stronger
judgments than a small local model, at low cost. Implements TeacherLabeler.

Setup (DeepSeek):
    export DEEPSEEK_API_KEY=sk-...

Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from mnemic.hybrid.errors import InvalidInput
from mnemic.hybrid.teacher_prompts import build_graph_worthiness_prompt, parse_graph_worthy


@dataclass
class ApiTeacher:
    """Judges graph-worthiness via an OpenAI-compatible chat API (default DeepSeek).

    ``complete`` is injectable so the logic is unit-testable without network.
    """

    model: str = 'deepseek-chat'
    base_url: str = 'https://api.deepseek.com'
    api_key_env: str = 'DEEPSEEK_API_KEY'
    temperature: float = 0.0
    timeout: float = 60.0
    complete: Callable[[str], Awaitable[str]] | None = None

    async def label(self, text: str) -> bool:
        if not isinstance(text, str) or not text.strip():
            raise InvalidInput('text to label must be a non-empty string')
        complete = self.complete or self._api_complete
        raw = await complete(build_graph_worthiness_prompt(text))
        return parse_graph_worthy(raw)

    async def _api_complete(self, prompt: str) -> str:  # pragma: no cover - network I/O
        import os

        import httpx

        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise InvalidInput(f'{self.api_key_env} is not set')
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f'{self.base_url}/chat/completions',
                headers={'Authorization': f'Bearer {api_key}'},
                json={
                    'model': self.model,
                    'messages': [{'role': 'user', 'content': prompt}],
                    'temperature': self.temperature,
                    'stream': False,
                },
            )
            resp.raise_for_status()
            return str(resp.json()['choices'][0]['message']['content'])
