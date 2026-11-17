"""The clock's opaque version, which every write changes, and set_rate, which changes its rate the way advance moves it:
from its time at the write, keeping t0 and generation."""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from agent_env.env.gateway.clock import Clock, ClockError
from agent_env.env.gateway.triggers import TriggerEngine
from tst.unit.env.gateway.triggers_time_test import FakeNow, _FakeGateway, _perm, _settle, _tick_call

T0 = "2026-01-01T00:00:00Z"


def _utc(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def test_every_write_changes_the_version_and_reads_leave_it():
    clock = Clock(now_fn=FakeNow())
    seen = [clock.version]
    clock.set_time(T0, rate=1.0)
    seen.append(clock.version)
    clock.read(), clock.state(), clock.now()
    assert clock.version == seen[-1]
    clock.advance("2026-01-02T00:00:00Z")
    seen.append(clock.version)
    clock.set_rate(60)
    seen.append(clock.version)
    clock.clear()
    seen.append(clock.version)
    clock.clear()
    seen.append(clock.version)
    assert len(set(seen)) == len(seen)
    assert [clock.count_of(version) for version in seen] == [0, 1, 2, 3, 4, 5]


def test_a_version_never_repeats_across_clocks_and_names_only_what_its_clock_had():
    first, second = Clock(now_fn=FakeNow()), Clock(now_fn=FakeNow())
    assert first.version != second.version
    first.set_time(T0)
    boot = first.version.split(".")[0]
    assert first.count_of(first.version) == 1
    assert first.count_of(second.version) is None
    assert [first.count_of(v) for v in (f"{boot}.2", f"{boot}.-1", f"{boot}.²", f"{boot}", "", None, 3)] == [None] * 7


def test_a_rate_change_continues_from_the_clock_as_of_the_write_keeping_its_t0_and_generation():
    """Taking the new time from a state read a moment earlier stepped the universe back by the gap, which a computer,
    never stepping back, then ran ahead of; and a re-arm moved every time trigger's relative marks to the new t0."""
    now = FakeNow()
    clock = Clock(now_fn=now)
    clock.set_time(T0, rate=1.0)
    generation = clock.generation
    now.tick(10)
    read = clock.state()
    now.tick(0.07)
    state = clock.set_rate(60)
    assert _utc(state["virtual_time"]) - _utc(read["virtual_time"]) == timedelta(milliseconds=70)
    assert (state["virtual_seconds_per_real_second"], state["t0"], clock.generation) == (60, T0, generation)
    now.tick(1)
    assert clock.read()["virtual_time"] == "2026-01-01T00:01:10.070000Z"
    assert clock.elapsed() == pytest.approx(70.07)
    assert clock.at_elapsed(130.07) == _utc("2026-01-01T00:02:10.070000Z")
    clock.advance("2026-01-01T01:00:00Z")
    now.tick(1)
    assert clock.read()["virtual_time"] == "2026-01-01T01:01:00Z"


def test_a_rate_change_hands_followers_a_clock_continuous_with_the_one_before():
    """A follower extrapolating the snapshot before the change to the moment of the write is where the one after starts."""
    now = FakeNow()
    clock = Clock(now_fn=now)
    clock.set_time(T0, rate=1.0)
    before, at = clock.state(), now()
    now.tick(3.25)
    after = clock.set_rate(2)
    extrapolated = _utc(before["virtual_time"]) + timedelta(seconds=(now() - at) * before["virtual_seconds_per_real_second"])
    assert abs((_utc(after["virtual_time"]) - extrapolated).total_seconds()) < 0.001


class SteppingNow:
    """A real clock's reading moves between any two calls: each call is 100 µs after the one before it."""

    def __init__(self):
        self.t, self.calls = 1000.0, []

    def __call__(self) -> float:
        self.t += 0.0001
        self.calls.append(self.t)
        return self.t


@pytest.mark.parametrize("before,after", [(1.0, 86400.0), (86400.0, 1.0)])
def test_a_rate_change_starts_its_new_rate_at_the_instant_it_read_the_time(before, after):
    now = SteppingNow()
    clock = Clock(now_fn=now)
    mark = len(now.calls)
    clock.set_time(T0, rate=before)
    armed_at = now.calls[mark]
    mark = len(now.calls)
    clock.set_rate(after)
    changed_at = now.calls[mark]
    read = clock.now()
    expected = _utc(T0) + timedelta(seconds=(changed_at - armed_at) * before + (now.calls[-1] - changed_at) * after)
    assert abs((read - expected).total_seconds()) < 0.001


def test_a_rate_change_needs_an_armed_clock_and_a_rate():
    with pytest.raises(ClockError, match="not armed"):
        Clock(now_fn=FakeNow()).set_rate(2)
    clock = Clock(now_fn=FakeNow())
    clock.set_time(T0)
    for rate in (None, -1, "2"):
        with pytest.raises(ClockError, match="rate must be"):
            clock.set_rate(rate)


@pytest.mark.asyncio
async def test_a_relative_time_trigger_keeps_its_mark_across_a_rate_change():
    now = FakeNow()
    clock = Clock(now_fn=now)
    engine = TriggerEngine(_FakeGateway(clock))
    clock.set_time(T0, rate=1)
    engine.register({"watch_roles": ["default"], "triggers": [
        {"id": "rel", "when": {"type": "time", "at": "PT2H"}, "actions": [_perm()]}]})
    _tick_call(engine, now, 3600)
    clock.set_rate(3600)
    assert engine.clock_advanced() == []
    _tick_call(engine, now, 1.5)
    await _settle(engine)
    trig = engine.state()["triggers"][0]
    assert (trig["status"], trig["fire_count"]) == ("fired", 1)
    assert "reanchored" not in {e["kind"] for e in engine.state()["events"]}


def test_a_count_too_long_to_be_one_the_clock_had_names_nothing():
    clock = Clock(now_fn=FakeNow())
    clock.set_time(T0)
    boot = clock.version.split(".")[0]
    assert [clock.count_of(f"{boot}.{'9' * 5000}"), clock.count_of(f"{boot}.01"), clock.count_of(f"{boot}.1")] == [None, None, 1]
