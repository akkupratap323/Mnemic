"""
Copyright 2026, Abishek (Mnemic project).

A background scheduler that runs the consolidation worker on an interval, off the
hot path. Wrap run_forever() in asyncio.create_task() to run it in the background.
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from mnemic.hybrid.consolidation import ConsolidationReport, Consolidator
from mnemic.hybrid.errors import InvalidInput


@dataclass
class ConsolidationScheduler:
    """Runs ``consolidator.run()`` every ``interval_seconds`` until stopped.

    A pass that raises (e.g. a locked or unreachable vector store) is recorded in
    ``last_error`` and ``failures`` and retried after the next interval, so one bad
    pass never stops the background loop for good.
    """

    consolidator: Consolidator
    interval_seconds: float = 3600.0
    sleeper: Callable[[float], Awaitable[None]] = asyncio.sleep
    running: bool = field(default=False, init=False)
    failures: int = field(default=0, init=False)
    last_error: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.interval_seconds <= 0:
            raise InvalidInput('interval_seconds must be positive')

    async def run_once(self) -> ConsolidationReport:
        return await self.consolidator.run()

    async def run_forever(self) -> None:
        self.running = True
        while self.running:
            try:
                await self.run_once()
                self.last_error = None
            except Exception as exc:  # retry next interval; never kill the loop
                self.failures += 1
                self.last_error = f'{type(exc).__name__}: {exc}'
            if self.running:
                await self.sleeper(self.interval_seconds)

    def stop(self) -> None:
        self.running = False
