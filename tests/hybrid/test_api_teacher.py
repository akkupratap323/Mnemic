"""Tests for the API-based (DeepSeek/OpenAI-compatible) router teacher."""

from __future__ import annotations

import pytest

from mnemic.hybrid.api_teacher import ApiTeacher
from mnemic.hybrid.errors import InvalidInput


def test_defaults_to_deepseek():
    t = ApiTeacher()
    assert t.model == 'deepseek-chat'
    assert t.base_url == 'https://api.deepseek.com'
    assert t.api_key_env == 'DEEPSEEK_API_KEY'


async def test_label_uses_injected_complete():
    captured = {}

    async def fake_complete(prompt: str) -> str:
        captured['prompt'] = prompt
        return 'GRAPH'

    teacher = ApiTeacher(complete=fake_complete)
    assert await teacher.label('we migrated to postgres') is True
    assert 'we migrated to postgres' in captured['prompt']


async def test_label_maps_vector_response():
    async def fake_complete(_prompt: str) -> str:
        return 'VECTOR'

    teacher = ApiTeacher(complete=fake_complete)
    assert await teacher.label('good morning') is False


async def test_label_can_target_openai():
    async def fake_complete(_prompt: str) -> str:
        return 'GRAPH'

    teacher = ApiTeacher(
        model='gpt-4.1-mini',
        base_url='https://api.openai.com/v1',
        api_key_env='OPENAI_API_KEY',
        complete=fake_complete,
    )
    assert teacher.model == 'gpt-4.1-mini'
    assert await teacher.label('we decided to ship') is True


@pytest.mark.parametrize('bad', ['', '   ', None])
async def test_label_rejects_empty(bad):
    async def fake_complete(_prompt: str) -> str:
        return 'GRAPH'

    teacher = ApiTeacher(complete=fake_complete)
    with pytest.raises(InvalidInput):
        await teacher.label(bad)  # type: ignore[arg-type]
