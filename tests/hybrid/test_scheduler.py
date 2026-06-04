"""Tests for the consolidation scheduler."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from mnemic.hybrid.consolidation import ConsolidationReport
from mnemic.hybrid.errors import InvalidInput
from mnemic.hybrid.scheduler import ConsolidationScheduler


@dataclass
class SpyConsolidator:
    runs: int = 0
    reports: list[ConsolidationReport] = field(default_factory=list)

    async def run(self) -> ConsolidationReport:
        self.runs += 1
        report = ConsolidationReport(scanned=self.runs)
        self.reports.append(report)
        return report


def test_invalid_interval_rejected():
    with pytest.raises(InvalidInput):
        ConsolidationScheduler(consolidator=SpyConsolidator(), interval_seconds=0)


async def test_run_once_delegates():
    spy = SpyConsolidator()
    sched = ConsolidationScheduler(consolidator=spy, interval_seconds=1)
    report = await sched.run_once()
    assert spy.runs == 1
    assert report.scanned == 1


async def test_run_forever_stops_via_sleeper():
    spy = SpyConsolidator()
    sched = ConsolidationScheduler(consolidator=spy, interval_seconds=1)

    n = 0

    async def fake_sleep(_seconds: float) -> None:
        nonlocal n
        n += 1
        if n >= 3:
            sched.stop()

    sched.sleeper = fake_sleep
    await sched.run_forever()
    assert spy.runs == 3  # ran, slept, ran, slept, ran, slept->stop
    assert sched.running is False


async def test_stop_before_start_is_safe():
    sched = ConsolidationScheduler(consolidator=SpyConsolidator(), interval_seconds=1)
    sched.stop()
    assert sched.running is False
