"""A check whose effect a busy desk can miss is tried again before it fails, and a pass on a later try says what the
earlier one found; any other check is tried once."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from agentenv_protocol.computer.conformance import checks

STARVED = "the stroke was [50, 50] to [300, 250] in 3 points"


def _probe() -> checks.Probe:
    return checks.Probe(SimpleNamespace(params=lambda uri: {}), session=None, ready_timeout_s=1)


def _answering(*answers: str | None):
    calls: list[int] = []
    pending = iter(answers)

    async def test() -> str | None:
        calls.append(len(calls) + 1)
        return next(pending)

    return test, calls


@pytest.mark.asyncio
async def test_a_check_tried_again_passes_saying_what_its_first_try_found():
    probe, (drag, calls) = _probe(), _answering(STARVED, None)

    assert await probe.check("P7", "a drag", drag, attempts=checks.BUSY_DESK_ATTEMPTS)

    [result] = probe.results
    assert (result.status, result.detail, calls) == ("passed", f"passed on try 2 of 2; before: {STARVED}", [1, 2])


@pytest.mark.asyncio
async def test_a_check_that_fails_every_try_fails_with_what_its_last_try_found():
    probe, (drag, calls) = _probe(), _answering(STARVED, "the stroke missed its waypoint (200, 100)")

    assert not await probe.check("P7", "a drag", drag, attempts=checks.BUSY_DESK_ATTEMPTS)

    [result] = probe.results
    assert (result.status, result.detail, calls) == ("failed", "the stroke missed its waypoint (200, 100)", [1, 2])


@pytest.mark.asyncio
async def test_any_other_check_is_tried_once():
    probe, (test, calls) = _probe(), _answering("the fixture shows 'abc'", None)

    assert not await probe.check("P6", "typed text arrives", test)

    assert (probe.results[0].status, calls) == ("failed", [1])
