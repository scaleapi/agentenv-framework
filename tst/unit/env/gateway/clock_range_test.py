"""The env clock's range, the times 64 bits of nanoseconds since 1970 hold, as Python and the computers that follow the
clock keep time: the gateway arms and runs anywhere in it, before 1970 too and a day per second at today's dates, and
refuses a time outside it, however it's spelled, or arming the clock where it would run to the end within a real hour,
up front, naming the range. A clock that runs to the end anyway, on a gateway that outlives it or sped up in its last
hour, stands there."""
from __future__ import annotations

import re
from datetime import datetime, timezone

import pytest

from agent_env.env.gateway.clock import EARLIEST, LATEST, Clock, ClockError, _iso, check_arm
from agent_env.env.gateway.constants import EXT_CLOCK_URI, GATEWAY_EXTENSIONS
from tst.unit.env.gateway.triggers_time_test import FakeNow

RANGE = "the env clock's range, 1677-09-21T00:12:44Z to 2262-04-11T23:47:16Z"
END = "would pass 2262-04-11T23:47:16Z, the end of the env clock's range, within a real hour"


def _refused(message: str):
    return pytest.raises(ClockError, match=re.escape(message))


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


@pytest.mark.parametrize("virtual_time", [
    "0001-01-01T00:00:00Z", "1600-01-01T00:00:00Z", "1677-09-21T00:12:43Z", "2262-04-11T23:47:17Z",
    "3026-11-23T12:00:00Z", "9999-12-31T23:59:59Z", "10000-01-01T00:00:00Z", "0000-01-01T00:00:00Z",
    "0001-01-01T00:00:00+01:00", "9999-12-31T23:59:59-01:00", "1677-09-21T00:12:44+00:01"])
def test_a_time_outside_the_range_is_refused_naming_it_however_its_spelled(virtual_time):
    with _refused(f"virtual_time '{virtual_time}' is outside {RANGE}"):
        Clock(now_fn=FakeNow()).set_time(virtual_time, rate=0)


@pytest.mark.parametrize("virtual_time", ["10000-junk", "999-01-01T00:00:00Z", "2026-13-01T00:00:00Z"])
def test_a_string_that_isnt_a_time_is_refused_as_one(virtual_time):
    with _refused(f"invalid RFC3339 virtual_time '{virtual_time}'"):
        Clock(now_fn=FakeNow()).set_time(virtual_time, rate=0)


def test_arming_a_clock_that_would_pass_the_end_within_a_real_hour_names_it():
    with _refused(f"virtual_time '2262-04-11T23:00:00Z' at rate 1 {END}"):
        Clock(now_fn=FakeNow()).set_time("2262-04-11T23:00:00Z")


def test_however_slow_a_clock_runs_from_the_end_it_would_pass_it():
    with _refused(f"virtual_time '2262-04-11T23:47:16Z' at rate 1e-10 {END}"):
        Clock(now_fn=FakeNow()).set_time("2262-04-11T23:47:16Z", rate=1e-10)


def test_a_clock_frozen_in_its_last_real_hour_starts_again_and_runs_to_the_end():
    """As always_on runs one: frozen between turns, advanced while frozen to a due event, then run at the task's rate
    for the next turn."""
    now = FakeNow()
    clock = Clock(now_fn=now)
    clock.set_time("2262-04-11T22:40:00Z")
    now.tick(60)
    clock.set_rate(0)
    clock.advance("2262-04-11T23:00:00Z")

    assert clock.set_rate(1)["virtual_seconds_per_real_second"] == 1
    now.tick(40 * 60)
    assert clock.read() == {"virtual_time": "2262-04-11T23:40:00Z"}
    now.tick(10 * 60)
    state = clock.state()
    assert (state["virtual_time"], state["virtual_seconds_per_real_second"]) == ("2262-04-11T23:47:16Z", 0.0)


def test_in_its_last_real_hour_a_running_clock_takes_any_advance_or_rate_and_stands_at_the_end():
    now = FakeNow()
    clock = Clock(now_fn=now)
    clock.set_time("2262-04-11T22:47:16Z")

    assert clock.advance("2262-04-11T23:30:00Z")["virtual_time"] == "2262-04-11T23:30:00Z"
    assert clock.set_rate(86400)["virtual_seconds_per_real_second"] == 86400
    now.tick(1)
    assert clock.now() == LATEST
    with _refused(f"virtual_time '2262-04-11T23:47:17Z' is outside {RANGE}"):
        clock.advance("2262-04-11T23:47:17Z")


def test_a_clock_armed_frozen_at_the_end_takes_a_rate_as_one_that_ran_there_does():
    clock = Clock(now_fn=FakeNow())
    clock.set_time("2262-04-11T23:47:16Z", rate=0)
    state = clock.set_rate(1e-10)
    assert (state["virtual_time"], state["virtual_seconds_per_real_second"]) == ("2262-04-11T23:47:16Z", 0.0)


def test_a_clock_that_outlives_its_range_stands_at_the_end_and_can_still_be_frozen_or_set_back():
    """Its reads and state stop at the end, at rate 0, so the computers that follow it stop there too rather than run
    past what their clock page holds."""
    now = FakeNow()
    clock = Clock(now_fn=now)
    clock.set_time("2262-04-11T22:47:16Z")
    now.tick(3601)
    assert clock.read() == {"virtual_time": "2262-04-11T23:47:16Z"}
    state = clock.state()
    assert (state["virtual_time"], state["virtual_seconds_per_real_second"]) == ("2262-04-11T23:47:16Z", 0.0)
    now.tick(10**9)
    assert clock.now() == LATEST

    assert clock.advance("2262-04-11T23:47:16Z")["virtual_time"] == "2262-04-11T23:47:16Z"
    assert clock.set_rate(1)["virtual_time"] == "2262-04-11T23:47:16Z"
    assert clock.set_rate(0)["virtual_time"] == "2262-04-11T23:47:16Z"
    with _refused("advance only moves forward"):
        clock.advance("2262-04-11T23:00:00Z")
    assert clock.set_time("2026-11-23T12:00:00Z")["virtual_time"] == "2026-11-23T12:00:00Z"


def test_a_check_names_the_time_as_its_caller_wrote_it():
    with _refused(f"virtual_time '1677-09-21T05:00:00' in Asia/Tokyo is outside {RANGE}"):
        check_arm("1677-09-20T19:41:01Z", 0, what="virtual_time '1677-09-21T05:00:00' in Asia/Tokyo")
    assert check_arm("2026-11-23T12:00:00Z", 60) == (datetime(2026, 11, 23, 12, tzinfo=timezone.utc), 60.0)


def test_the_clock_card_states_the_range():
    clock = next(ext for ext in GATEWAY_EXTENSIONS if ext["uri"] == EXT_CLOCK_URI)
    assert f"{_iso(EARLIEST)} to {_iso(LATEST)}" in clock["description"]
