"""Deterministic test doubles for the hybrid memory layer (no I/O, no network)."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass, field

import pytest

from mnemic.hybrid.errors import InvalidInput

_TOKEN_RE = re.compile(r'[a-z0-9]+')


class FakeEmbedder:
    """Deterministic bag-of-words embedder, stable across processes/runs."""

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim
        self.calls: list[str] = []

    async def embed(self, text: str) -> tuple[float, ...]:
        if not text or not text.strip():
            raise InvalidInput('empty text')
        self.calls.append(text)
        vec = [0.0] * self.dim
        for token in _TOKEN_RE.findall(text.lower()):
            digest = hashlib.md5(token.encode(), usedforsecurity=False).digest()
            idx = int.from_bytes(digest[:4], 'big') % self.dim
            vec[idx] += 1.0
        if not any(vec):
            vec[0] = 1.0
        return tuple(vec)


class FakeEmbedderClient:
    """Stands in for mnemic.embedder.EmbedderClient (sync values, async create)."""

    def __init__(self, vector: list[float] | None = None) -> None:
        self.vector = vector if vector is not None else [1.0, 2.0, 3.0]
        self.calls: list[str] = []

    async def create(self, input_data: str) -> list[float]:
        self.calls.append(input_data)
        return list(self.vector)


@dataclass
class RecordingGraph:
    """Records remember() calls; can simulate a failing graph tier."""

    calls: list[dict] = field(default_factory=list)
    raise_on_call: bool = False

    async def remember(self, **kwargs: object) -> object:
        self.calls.append(dict(kwargs))
        if self.raise_on_call:
            raise RuntimeError('graph down')
        return {'ok': True}


def make_counter_ids(prefix: str = 'id') -> Callable[[], str]:
    """Returns a deterministic, monotonic id factory."""
    state = {'n': 0}

    def _next() -> str:
        state['n'] += 1
        return f'{prefix}-{state["n"]}'

    return _next


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def graph() -> RecordingGraph:
    return RecordingGraph()
