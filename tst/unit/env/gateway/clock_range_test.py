"""The env clock's range, the times 64 bits of nanoseconds since 1970 hold, as Python and the computers that follow the
clock keep time: the gateway arms and runs anywhere in it, before 1970 too and a day per second at today's dates, and
refuses a time outside it, or one it would run past the end from within a real hour, up front, naming the range."""
from __future__ import annotations

import pytest

from agent_env.env.gateway.clock import EARLIEST, LATEST, Clock, ClockError
from tst.unit.env.gateway.triggers_time_test import FakeNow

RANGE = "the env clock's range, 1677-09-21T00:12:44Z to 2262-04-11T23:47:16Z"
END = "would pass 2262-04-11T23:47:16Z, the end of the env clock's range, within a real hour"


def test_the_range_is_the_whole_seconds_64_bits_of_nanoseconds_since_1970_hold():
    first, last = int(EARLIEST.timestamp()), int(LATEST.timestamp())
    assert (first - 1) * 10**9 < -2**63 <= first * 10**9
    assert last * 10**9 <= 2**63 - 1 < (last + 1) * 10**9


@pytest.mark.parametrize("virtual_time, rate", [
    ("1677-09-21T00:12:44Z", 1), ("1900-01-01T00:00:00Z", 60), ("1949-12-31T23:59:59.750000Z", 1),
    ("2026-11-23T12:00:00Z", 86400), ("2262-04-11T22:47:16Z", 1), ("2262-04-11T23:47:16Z", 0)])
def test_the_clock_arms_and_runs_anywhere_in_its_range(virtual_time, rate):
    now = FakeNow()
    clock = Clock(now_fn=now)

    assert clock.set_time(virtual_time, rate=rate)["virtual_time"] == virtual_time
    now.tick(1)
    assert clock.read()["virtual_time"] >= virtual_time


@pytest.mark.parametrize("virtual_time", ["0001-01-01T00:00:00Z", "1600-01-01T00:00:00Z", "1677-09-21T00:12:43Z",
                                          "2262-04-11T23:47:17Z", "3026-11-23T12:00:00Z", "9999-12-31T23:59:59Z",
                                          "10000-01-01T00:00:00Z"])
def test_a_time_outside_the_range_is_refused_naming_it(virtual_time):
    with pytest.raises(ClockError, match=f"virtual_time '{virtual_time}' is outside {RANGE}"):
        Clock(now_fn=FakeNow()).set_time(virtual_time, rate=0)


def test_arming_advancing_or_speeding_a_clock_that_would_pass_the_end_within_a_real_hour_names_it():
    clock = Clock(now_fn=FakeNow())
    with pytest.raises(ClockError, match=f"virtual_time '2262-04-11T23:00:00Z' at rate 1 {END}"):
        clock.set_time("2262-04-11T23:00:00Z")

    clock.set_time("2262-04-01T00:00:00Z")
    with pytest.raises(ClockError, match=f"virtual_time '2262-04-11T23:00:00Z' at rate 1 {END}"):
        clock.advance("2262-04-11T23:00:00Z")
    with pytest.raises(ClockError, match=f"rate 3600 {END}"):
        clock.set_rate(3600)
    assert clock.read()["virtual_time"] == "2262-04-01T00:00:00Z"
