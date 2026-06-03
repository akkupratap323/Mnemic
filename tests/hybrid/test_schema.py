"""Tests for the code-aware entity/edge schema."""

from __future__ import annotations

from pydantic import BaseModel

from mnemic.hybrid.schema import (
    CODE_EDGE_TYPES,
    CODE_ENTITY_TYPES,
    Bug,
    File,
)


def test_entity_registry_contains_expected_types():
    assert set(CODE_ENTITY_TYPES) >= {'Repo', 'File', 'Function', 'Decision', 'Bug', 'Task'}
    assert all(issubclass(t, BaseModel) for t in CODE_ENTITY_TYPES.values())


def test_edge_registry_contains_expected_types():
    assert set(CODE_EDGE_TYPES) >= {'DependsOn', 'FixedBy', 'Supersedes'}
    assert all(issubclass(t, BaseModel) for t in CODE_EDGE_TYPES.values())


def test_models_default_to_none_and_validate():
    f = File()
    assert f.path is None and f.language is None
    f2 = File(path='src/main.py', language='python')
    assert f2.path == 'src/main.py'


def test_bug_fixed_flag():
    assert Bug().fixed is False
    assert Bug(description='npe', severity='high', fixed=True).fixed is True
