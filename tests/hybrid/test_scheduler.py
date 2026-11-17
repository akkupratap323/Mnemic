"""Tests for the consolidation scheduler."""

from __future__ import annotations

import asyncio
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


@dataclass
class FlakyConsolidator:
    """Raises on the passes listed in ``fail_on`` (1-based), succeeds otherwise."""

    fail_on: set[int]
    runs: int = 0

    async def run(self) -> ConsolidationReport:
        self.runs += 1
        if self.runs in self.fail_on:
            raise OSError(f'database is locked (pass {self.runs})')
        return ConsolidationReport(scanned=self.runs)


def _stop_after(sched: ConsolidationScheduler, sleeps: int, seen: list[str | None]):
    async def fake_sleep(_seconds: float) -> None:
        seen.append(sched.last_error)
        if len(seen) >= sleeps:
            sched.stop()

    return fake_sleep


async def test_a_failed_pass_does_not_stop_the_loop():
    flaky = FlakyConsolidator(fail_on={1})
    sched = ConsolidationScheduler(consolidator=flaky, interval_seconds=1)
    seen: list[str | None] = []
    sched.sleeper = _stop_after(sched, 3, seen)

    await sched.run_forever()

    assert flaky.runs == 3  # failed, slept, ran, slept, ran, slept->stop
    assert seen == ['OSError: database is locked (pass 1)', None, None]
    assert sched.failures == 1
    assert sched.last_error is None
    assert sched.running is False


async def test_repeated_failures_are_counted_and_still_wait_the_interval():
    flaky = FlakyConsolidator(fail_on={1, 2})
    sched = ConsolidationScheduler(consolidator=flaky, interval_seconds=1)
    seen: list[str | None] = []
    sched.sleeper = _stop_after(sched, 2, seen)

    await sched.run_forever()

    assert flaky.runs == 2  # one pass per interval, no tight retry loop
    assert sched.failures == 2
    assert sched.last_error == 'OSError: database is locked (pass 2)'


async def test_cancellation_still_stops_the_loop():
    @dataclass
    class CancelledConsolidator:
        async def run(self) -> ConsolidationReport:
            raise asyncio.CancelledError

    sched = ConsolidationScheduler(consolidator=CancelledConsolidator(), interval_seconds=1)
    sched.sleeper = _stop_after(sched, 1, [])
    with pytest.raises(asyncio.CancelledError):
        await sched.run_forever()
    assert sched.failures == 0
