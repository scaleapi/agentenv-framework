"""The clock's followers that hold only a range of times (a computer's and a clock-following agent's card declare
clock/v1's ``range``, what 64 bits of nanoseconds since 1970 hold): while the clock has one, it refuses a set_time or an
advance outside that range, naming the follower, and a set_time it would run past the range's end from within a real
hour; a declaration the clock is outside now, or would run past the end of from within a real hour, is refused. Without
such a follower the clock is as it always was."""
from __future__ import annotations

import asyncio

import pytest
from agentenv_protocol.computer.clock import RANGE

from agent_env.env.gateway.clock import Clock, ClockError, OutsideRange, parse_follower_range
from agent_env.env.gateway.clock_followers import ClockFollowers
from agent_env.env.gateway.entrypoint import parse_clock_follower_ranges
from tst.unit.env.gateway.triggers_time_test import FakeNow

HELD = parse_follower_range(RANGE)
DESKTOP = "clock follower 'desktop' holds, 1677-09-21T00:12:44Z to 2262-04-11T23:47:16Z"


def _clock(held: dict | None = None, tokens: dict | None = None) -> tuple[Clock, ClockFollowers, FakeNow]:
    now = FakeNow()
    followers: ClockFollowers | None = None
    clock = Clock(now_fn=now, follower_ranges=lambda: followers.held())
    followers = ClockFollowers(lambda: clock, tokens if tokens is not None else dict.fromkeys(held or {}, "token"), held)
    return clock, followers, now


def test_the_range_computers_and_agents_declare_is_what_64_bits_of_nanoseconds_since_1970_hold():
    first, last = int(HELD.earliest.timestamp()), int(HELD.latest.timestamp())
    assert (first - 1) * 10**9 < -2**63 <= first * 10**9
    assert last * 10**9 <= 2**63 - 1 < (last + 1) * 10**9


@pytest.mark.parametrize("virtual_time", ["1600-01-01T00:00:00Z", "2262-04-11T23:47:17Z", "3026-11-23T12:00:00Z",
                                          "9999-06-01T12:00:00Z"])
def test_without_a_follower_that_holds_a_range_the_clock_arms_anywhere_a_datetime_holds(virtual_time):
    for clock in (_clock()[0], _clock(tokens={"desktop": "token"})[0]):
        assert clock.set_time(virtual_time)["virtual_time"] == virtual_time


def test_without_one_the_clock_still_refuses_what_it_always_did():
    with pytest.raises(ClockError, match="would overflow datetime near the max"):
        _clock()[0].set_time("9999-12-31T12:00:00Z")
    with pytest.raises(ClockError, match="unknown time zone 'Nowhere/Zone'"):
        _clock()[0].set_time("9999-12-31T12:00:00Z", 1, "Nowhere/Zone")


@pytest.mark.parametrize("virtual_time", ["0001-01-01T00:00:00Z", "1677-09-21T00:12:43Z", "2262-04-11T23:47:17Z",
                                          "3026-11-23T12:00:00Z", "9999-06-01T12:00:00Z"])
def test_a_time_outside_a_followers_range_is_refused_naming_it(virtual_time):
    clock, _, _ = _clock({"desktop": HELD})
    with pytest.raises(ClockError, match=f"virtual_time '{virtual_time}' is outside the times {DESKTOP}"):
        clock.set_time(virtual_time, rate=0)
    assert not clock.state()["armed"]


@pytest.mark.parametrize("virtual_time, rate", [("1677-09-21T00:12:44Z", 1), ("1949-12-31T23:59:59.750000Z", 1),
                                                ("2026-11-23T12:00:00Z", 86400), ("2262-04-11T22:47:16Z", 1),
                                                ("2262-04-11T23:47:16Z", 0)])
def test_the_clock_arms_anywhere_in_a_followers_range(virtual_time, rate):
    clock, _, now = _clock({"desktop": HELD})
    assert clock.set_time(virtual_time, rate=rate)["virtual_time"] == virtual_time
    now.tick(1)
    assert clock.read()["virtual_time"] >= virtual_time


def test_arming_a_clock_that_would_run_past_a_followers_range_within_a_real_hour_is_refused():
    clock, _, _ = _clock({"desktop": HELD})
    with pytest.raises(ClockError, match=f"at rate 1 would pass the end of the times {DESKTOP}, within a real hour"):
        clock.set_time("2262-04-11T23:00:00Z")
    with pytest.raises(ClockError, match="at rate 86400 would pass the end"):
        clock.set_time("2255-01-01T00:00:00Z", rate=86400)


def test_an_advance_past_a_followers_range_is_refused_and_one_into_its_last_hour_or_a_rate_change_is_not():
    clock, _, _ = _clock({"desktop": HELD})
    clock.set_time("2262-04-01T00:00:00Z")
    with pytest.raises(ClockError, match=f"virtual_time '2262-04-12T00:00:00Z' is outside the times {DESKTOP}"):
        clock.advance("2262-04-12T00:00:00Z")
    assert clock.read()["virtual_time"] == "2262-04-01T00:00:00Z"
    assert clock.advance("2262-04-11T23:00:00Z")["virtual_time"] == "2262-04-11T23:00:00Z"
    assert clock.set_rate(3600)["virtual_seconds_per_real_second"] == 3600


def test_a_declared_agent_brings_its_range_and_takes_it_away_when_retired():
    clock, followers, _ = _clock()
    clock.set_time("2026-11-23T12:00:00Z")
    followers.declare("agent-solver", "token", HELD)
    assert followers.state()["agent-solver"]["range"] == RANGE
    with pytest.raises(ClockError, match="is outside the times clock follower 'agent-solver' holds"):
        clock.set_time("3026-01-01T00:00:00Z")
    assert asyncio.run(followers.retire("agent-solver"))
    assert clock.set_time("3026-01-01T00:00:00Z")["virtual_time"] == "3026-01-01T00:00:00Z"


@pytest.mark.parametrize("virtual_time, rate", [("2262-04-11T23:30:00Z", 1), ("2262-04-11T00:00:00Z", 86400)])
def test_an_agent_the_clock_would_run_past_the_end_of_from_within_a_real_hour_is_refused_as_an_arm_with_it_is(
        virtual_time, rate):
    clock, followers, _ = _clock()
    clock.set_time(virtual_time, rate=rate)
    with pytest.raises(OutsideRange, match=f"the clock, at {virtual_time}, at rate {rate:g} would pass the end of the "
                                           "times clock follower 'agent-solver' holds, .*, within a real hour"):
        followers.declare("agent-solver", "token", HELD)
    assert "agent-solver" not in followers.state()
    with pytest.raises(OutsideRange, match="would pass the end"):
        _clock({"agent-solver": HELD})[0].set_time(virtual_time, rate=rate)
    clock.set_rate(0)
    followers.declare("agent-solver", "token", HELD)
    assert followers.state()["agent-solver"]["range"] == RANGE


def test_an_agent_whose_range_the_clock_is_outside_now_is_refused_and_one_that_declares_none_is_not():
    clock, followers, _ = _clock()
    clock.set_time("3026-01-01T00:00:00Z")
    with pytest.raises(ClockError, match="the clock, at 3026-01-01T00:00:00Z, is outside the times clock follower "
                                         "'agent-solver' holds, 1677-09-21T00:12:44Z to 2262-04-11T23:47:16Z"):
        followers.declare("agent-solver", "token", HELD)
    assert "agent-solver" not in followers.state()
    followers.declare("agent-old", "token")
    assert "range" not in followers.state()["agent-old"]
    assert clock.set_time("9000-01-01T00:00:00Z")["virtual_time"] == "9000-01-01T00:00:00Z"


def test_the_gateway_reads_each_computers_range_from_its_start_environment():
    assert parse_clock_follower_ranges(
        "desktop=1677-09-21T00:12:44Z/2262-04-11T23:47:16Z, browser=1950-01-01T00:00:00Z/2100-01-01T00:00:00Z") == {
        "desktop": HELD, "browser": parse_follower_range({"earliest": "1950-01-01T00:00:00Z",
                                                          "latest": "2100-01-01T00:00:00Z"})}
    assert parse_clock_follower_ranges("") == {}
    for spec in ("desktop", "desktop=1677-09-21T00:12:44Z", "desktop=2262-04-11T23:47:16Z/1677-09-21T00:12:44Z",
                 "desktop=yesterday/today"):
        with pytest.raises(ValueError, match="CLOCK_FOLLOWER_RANGES"):
            parse_clock_follower_ranges(spec)
    with pytest.raises(ValueError, match="invalid CLOCK_FOLLOWER_RANGES entry for 'desktop': invalid RFC3339 range "
                                         "earliest '1677-09-21'"):
        parse_clock_follower_ranges("desktop=1677-09-21/2262-04-11")


@pytest.mark.parametrize("held, said", [
    ({"earliest": "x", "latest": "y"}, "invalid RFC3339 range earliest 'x'"),
    ({"earliest": "1677-09-21T00:12:44Z", "latest": "2262-04-11"}, "invalid RFC3339 range latest '2262-04-11'"),
    ({}, "range earliest must be an RFC3339 string, got NoneType"),
    ({"earliest": "1677-09-21T00:12:44Z", "latest": 1}, "range latest must be an RFC3339 string, got int")])
def test_a_range_that_doesnt_parse_is_refused_naming_the_end_of_it_that_doesnt(held, said):
    with pytest.raises(ClockError, match=said) as refused:
        parse_follower_range(held)
    assert "virtual_time" not in str(refused.value)
