"""The env clock's range: the gateway's clock arms and runs at any date a datetime holds, before 1970 and past 2262
(where the computers and agents that follow it now follow too), and refuses a date past 9999-12-31, or one it would run
past within a real day, naming that latest time."""
from __future__ import annotations

import pytest

from agent_env.env.gateway.clock import LATEST, Clock, ClockError
from tst.unit.env.gateway.triggers_time_test import FakeNow


@pytest.mark.parametrize("virtual_time", ["1900-01-01T00:00:00Z", "1949-12-31T23:59:59.750000Z", "2262-04-11T23:47:17Z",
                                          "3026-11-23T12:00:00Z", "9999-06-01T12:00:00Z"])
def test_the_clock_arms_and_runs_at_any_date_a_datetime_holds(virtual_time):
    now = FakeNow()
    clock = Clock(now_fn=now)

    assert clock.set_time(virtual_time, rate=60)["virtual_time"] == virtual_time
    now.tick(1)
    assert clock.read()["virtual_time"] > virtual_time


@pytest.mark.parametrize("virtual_time", ["10000-01-01T00:00:00Z", "123456-01-01T00:00:00Z"])
def test_a_date_past_9999_is_refused_naming_the_latest_the_clock_reaches(virtual_time):
    with pytest.raises(ClockError, match=f"past {LATEST}, the latest the env clock reaches"):
        Clock(now_fn=FakeNow()).set_time(virtual_time)


def test_arming_advancing_or_speeding_a_clock_that_would_run_past_9999_within_a_day_names_the_latest():
    clock = Clock(now_fn=FakeNow())
    with pytest.raises(ClockError, match=f"would pass {LATEST}"):
        clock.set_time("9999-12-31T12:00:00Z")

    clock.set_time("9999-06-01T00:00:00Z")
    with pytest.raises(ClockError, match=f"would pass {LATEST}"):
        clock.advance("9999-12-31T12:00:00Z")
    with pytest.raises(ClockError, match=f"past {LATEST}"):
        clock.set_rate(86400)
