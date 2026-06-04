"""Tests for the local-LLM (Gemma via Ollama) router teacher."""

from __future__ import annotations

import pytest

from mnemic.hybrid.errors import InvalidInput
from mnemic.hybrid.ollama_teacher import (
    OllamaTeacher,
    build_graph_worthiness_prompt,
    parse_graph_worthy,
)


def test_prompt_includes_text_and_instructions():
    prompt = build_graph_worthiness_prompt('  we chose sqlite  ')
    assert 'we chose sqlite' in prompt
    assert 'GRAPH' in prompt and 'VECTOR' in prompt


@pytest.mark.parametrize(
    'response,expected',
    [
        ('GRAPH', True),
        ('vector', False),
        ('  GRAPH\n', True),
        ('This is graph-worthy', True),
        ('VECTOR — just chatter', False),
        ('yes', True),
        ('true', True),
        ('no', False),
        ('unsure', False),  # conservative default
    ],
)
def test_parse_graph_worthy(response, expected):
    assert parse_graph_worthy(response) is expected


async def test_label_uses_injected_generate():
    captured = {}

    async def fake_generate(prompt: str) -> str:
        captured['prompt'] = prompt
        return 'GRAPH'

    teacher = OllamaTeacher(generate=fake_generate)
    assert await teacher.label('we decided to migrate') is True
    assert 'we decided to migrate' in captured['prompt']


async def test_label_maps_vector_response():
    async def fake_generate(_prompt: str) -> str:
        return 'VECTOR'

    teacher = OllamaTeacher(generate=fake_generate)
    assert await teacher.label('hello there') is False


@pytest.mark.parametrize('bad', ['', '   ', None])
async def test_label_rejects_empty(bad):
    teacher = OllamaTeacher(generate=lambda _p: _noop())
    with pytest.raises(InvalidInput):
        await teacher.label(bad)  # type: ignore[arg-type]


async def _noop() -> str:
    return ''


def test_defaults_to_gemma_12b():
    assert OllamaTeacher().model == 'gemma3:12b'
