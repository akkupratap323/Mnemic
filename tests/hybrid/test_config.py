"""Tests for HybridConfig."""

from __future__ import annotations

import pytest

from mnemic.hybrid.config import HybridConfig
from mnemic.hybrid.errors import InvalidInput


def test_defaults_are_valid():
    cfg = HybridConfig()
    assert cfg.cache_maxsize == 10_000
    assert cfg.promote_after_accesses == 3
    assert cfg.default_recall_k == 10


def test_from_env_reads_overrides():
    env = {
        'MNEMIC_DB_PATH': '/tmp/m.db',
        'MNEMIC_CACHE_MAXSIZE': '500',
        'MNEMIC_PROMOTE_AFTER_ACCESSES': '5',
        'MNEMIC_PRUNE_AFTER_SECONDS': '120.5',
        'MNEMIC_PRUNE_MAX_ACCESSES': '2',
        'MNEMIC_DEFAULT_RECALL_K': '20',
    }
    cfg = HybridConfig.from_env(env)
    assert cfg.db_path == '/tmp/m.db'
    assert cfg.cache_maxsize == 500
    assert cfg.promote_after_accesses == 5
    assert cfg.prune_after_seconds == 120.5
    assert cfg.prune_max_accesses == 2
    assert cfg.default_recall_k == 20


def test_from_env_empty_uses_defaults():
    cfg = HybridConfig.from_env({})
    assert cfg == HybridConfig()


def test_from_env_blank_value_uses_default():
    assert HybridConfig.from_env({'MNEMIC_CACHE_MAXSIZE': ''}).cache_maxsize == 10_000


def test_from_env_invalid_int_raises():
    with pytest.raises(InvalidInput):
        HybridConfig.from_env({'MNEMIC_CACHE_MAXSIZE': 'abc'})


def test_from_env_invalid_float_raises():
    with pytest.raises(InvalidInput):
        HybridConfig.from_env({'MNEMIC_PRUNE_AFTER_SECONDS': 'soon'})


@pytest.mark.parametrize(
    'kwargs',
    [
        {'cache_maxsize': 0},
        {'promote_after_accesses': 0},
        {'prune_after_seconds': -1},
        {'prune_max_accesses': -1},
        {'default_recall_k': 0},
    ],
)
def test_validation_rejects_bad_values(kwargs):
    with pytest.raises(InvalidInput):
        HybridConfig(**kwargs)
