"""
Copyright 2026, Abishek (Mnemic project).

Environment-driven configuration for the hybrid memory layer. No external
settings dependency — a frozen dataclass with validation and an env loader.
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from mnemic.hybrid.errors import InvalidInput

_THIRTY_DAYS_SECONDS = 30 * 24 * 3600


@dataclass(frozen=True)
class HybridConfig:
    """Tunables for the hybrid memory stack (validated on construction)."""

    db_path: str = 'mnemic_memory.db'
    cache_maxsize: int = 10_000
    promote_after_accesses: int = 3
    prune_after_seconds: float = _THIRTY_DAYS_SECONDS
    prune_max_accesses: int = 0
    default_recall_k: int = 10

    def __post_init__(self) -> None:
        if self.cache_maxsize <= 0:
            raise InvalidInput('cache_maxsize must be positive')
        if self.promote_after_accesses <= 0:
            raise InvalidInput('promote_after_accesses must be positive')
        if self.prune_after_seconds < 0:
            raise InvalidInput('prune_after_seconds must be non-negative')
        if self.prune_max_accesses < 0:
            raise InvalidInput('prune_max_accesses must be non-negative')
        if self.default_recall_k <= 0:
            raise InvalidInput('default_recall_k must be positive')

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> HybridConfig:
        """Load config from environment variables (MNEMIC_* prefix)."""
        e = env if env is not None else os.environ
        return cls(
            db_path=e.get('MNEMIC_DB_PATH', cls.db_path),
            cache_maxsize=_int(e, 'MNEMIC_CACHE_MAXSIZE', cls.cache_maxsize),
            promote_after_accesses=_int(
                e, 'MNEMIC_PROMOTE_AFTER_ACCESSES', cls.promote_after_accesses
            ),
            prune_after_seconds=_float(e, 'MNEMIC_PRUNE_AFTER_SECONDS', cls.prune_after_seconds),
            prune_max_accesses=_int(e, 'MNEMIC_PRUNE_MAX_ACCESSES', cls.prune_max_accesses),
            default_recall_k=_int(e, 'MNEMIC_DEFAULT_RECALL_K', cls.default_recall_k),
        )


def _int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key)
    if raw is None or raw == '':
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise InvalidInput(f'{key} must be an integer, got {raw!r}') from exc


def _float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key)
    if raw is None or raw == '':
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise InvalidInput(f'{key} must be a number, got {raw!r}') from exc
