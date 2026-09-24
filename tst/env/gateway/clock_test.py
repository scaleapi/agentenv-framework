"""Unit tests for the wall-clock gateway virtual clock (urn:agentenv:clock/v1).

A fake monotonic clock (FakeNow) drives advancement deterministically, without real sleeps.
"""
from datetime import datetime, timezone

import pytest

from agent_env.env.gateway.clock import Clock, ClockError, _parse_rate

T0 = "2026-01-01T00:00:00Z"


class FakeNow:
    def __init__(self, start=1000.0):
        self.t = start

    def __call__(self):
        return self.t

    def tick(self, seconds):
        self.t += seconds


def _dt(iso):
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone(timezone.utc)


def test_unarmed_holds_no_time():
    c = Clock(now_fn=FakeNow())
    assert c.armed is False
    assert c.now() is None
    with pytest.raises(ClockError):
        c.read()


def test_baseline_at_t0():
    c = Clock(now_fn=FakeNow())
    c.set_time(T0, rate=1.0)
    assert c.read()["virtual_time"] == T0


def test_advances_at_rate_1():
    now = FakeNow()
    c = Clock(now_fn=now)
    c.set_time(T0, rate=1.0)
    now.tick(60)
    assert c.read()["virtual_time"] == "2026-01-01T00:01:00Z"


def test_rate_scaling():
    now = FakeNow()
    c = Clock(now_fn=now)
    c.set_time(T0, rate=3600)  # 1 real s -> 1 virtual h
    now.tick(2)
    assert c.read()["virtual_time"] == "2026-01-01T02:00:00Z"


def test_rate_zero_frozen():
    now = FakeNow()
    c = Clock(now_fn=now)
    c.set_time(T0, rate=0)
    now.tick(10_000)
    assert c.read()["virtual_time"] == T0


def test_default_rate_is_one():
    now = FakeNow()
    c = Clock(now_fn=now)
    c.set_time(T0)
    now.tick(30)
    assert c.read()["virtual_time"] == "2026-01-01T00:00:30Z"


def test_read_does_not_mutate():
    now = FakeNow()
    c = Clock(now_fn=now)
    c.set_time(T0, rate=1.0)
    now.tick(5)
    a = c.read()["virtual_time"]
    b = c.read()["virtual_time"]
    assert a == b == "2026-01-01T00:00:05Z"


def test_rearm_reanchors_and_applies_new_rate():
    now = FakeNow()
    c = Clock(now_fn=now)
    c.set_time(T0, rate=1.0)
    now.tick(10)
    assert c.read()["virtual_time"] == "2026-01-01T00:00:10Z"
    c.set_time("2026-06-01T00:00:00Z", rate=2.0)
    now.tick(5)
    assert c.read()["virtual_time"] == "2026-06-01T00:00:10Z"


def test_clear_disarms():
    now = FakeNow()
    c = Clock(now_fn=now)
    c.set_time(T0, rate=1.0)
    now.tick(5)
    c.clear()
    assert c.armed is False
    with pytest.raises(ClockError):
        c.read()


def test_monotonic_non_decreasing():
    now = FakeNow()
    c = Clock(now_fn=now)
    c.set_time(T0, rate=1.5)
    prev = _dt(c.read()["virtual_time"])
    for _ in range(5):
        now.tick(3)
        cur = _dt(c.read()["virtual_time"])
        assert cur >= prev
        prev = cur


def test_state_reports_fields():
    now = FakeNow()
    c = Clock(now_fn=now)
    assert c.state() == {"armed": False}
    st = c.set_time(T0, rate=3600)
    assert st["armed"] is True and st["virtual_seconds_per_real_second"] == 3600.0 and st["t0"] == T0 and st["virtual_time"] == T0
    now.tick(1)
    assert c.state()["virtual_time"] == "2026-01-01T01:00:00Z"


def test_offset_parsing():
    c = Clock(now_fn=FakeNow())
    c.set_time("2026-01-01T05:00:00+05:00", rate=1.0)  # +05:00 -> 00:00:00Z
    assert c.read()["virtual_time"] == T0


@pytest.mark.parametrize("bad", ["fast", True, [], {}, None, -1, -0.5, 86401, 1e9])
def test_parse_rate_rejects(bad):
    with pytest.raises(ClockError):
        _parse_rate(bad)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_parse_rate_rejects_non_finite(bad):
    """NaN slips past both ordered bounds (every comparison against it is False)."""
    with pytest.raises(ClockError):
        _parse_rate(bad)


def test_nan_rate_never_arms_the_clock():
    """A NaN rate must be refused at set_time, not stored and re-raised on every read (json.loads accepts bare NaN)."""
    c = Clock(now_fn=FakeNow())
    c.set_time(T0, rate=2.0)
    with pytest.raises(ClockError):
        c.set_time(T0, rate=float("nan"))
    assert c.armed is True
    assert c.state()["virtual_seconds_per_real_second"] == 2.0
    assert c.read()["virtual_time"] == T0


def test_parse_rate_accepts():
    assert _parse_rate(0) == 0.0
    assert _parse_rate(1) == 1.0
    assert _parse_rate(3600.5) == 3600.5
    assert _parse_rate(86400) == 86400.0


def test_set_time_rejects_bad_inputs():
    c = Clock(now_fn=FakeNow())
    with pytest.raises(ClockError):
        c.set_time(None)
    with pytest.raises(ClockError):
        c.set_time("not-a-date")
    with pytest.raises(ClockError):
        c.set_time(T0, rate=-2)
    with pytest.raises(ClockError):
        c.set_time(T0, rate=1e9)


@pytest.mark.parametrize("bad_time", ["2026-01-01T00:00:00", "2026-01-01", "2026-01-01 00:00:00"])
def test_set_time_rejects_offset_free(bad_time):
    c = Clock(now_fn=FakeNow())
    with pytest.raises(ClockError):
        c.set_time(bad_time, rate=1.0)


@pytest.mark.parametrize(
    "bad_time",
    [
        "2026-01-01T00:00:00+05:30:00",  # offset with seconds — fromisoformat takes it, RFC3339 doesn't
        "2026-01-01T00:00:00+05:30:00.000000",  # ...nor one with microseconds
        "2026-01-01T00:00+00:00",  # seconds are mandatory in partial-time
        "20260101T000000Z",  # basic format (fromisoformat accepts on 3.11+)
        "2026-W01-1T00:00:00Z",  # ISO week date
        "2026-001T00:00:00Z",  # ISO ordinal date
        " 2026-01-01T00:00:00Z",  # leading whitespace is not part of the grammar
        "2026-01-01T00:00:00Z ",  # ...nor trailing
        "\t2026-01-01T00:00:00Z\n",  # ...nor any other whitespace wrapping
    ],
)
def test_set_time_rejects_non_rfc3339_shapes(bad_time):
    c = Clock(now_fn=FakeNow())
    with pytest.raises(ClockError):
        c.set_time(bad_time, rate=1.0)


@pytest.mark.parametrize(
    "good_time,expected",
    [
        ("2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"),
        ("2026-01-01t00:00:00z", "2026-01-01T00:00:00Z"),  # RFC3339 allows lowercase T/Z
        ("2026-01-01 00:00:00+00:00", "2026-01-01T00:00:00Z"),  # RFC3339 NOTE: space separator
        ("2026-01-01T05:30:00+05:30", "2026-01-01T00:00:00Z"),  # offset normalized to UTC
        ("2026-01-01T00:00:00.500000Z", "2026-01-01T00:00:00.500000Z"),  # fractional seconds
    ],
)
def test_set_time_accepts_rfc3339_shapes(good_time, expected):
    c = Clock(now_fn=FakeNow())
    c.set_time(good_time, rate=0)
    assert c.read()["virtual_time"] == expected


def test_set_time_rejection_leaves_clock_untouched():
    """A refused virtual_time must not disturb an already-armed clock."""
    c = Clock(now_fn=FakeNow())
    c.set_time(T0, rate=2.0)
    with pytest.raises(ClockError):
        c.set_time("2026-06-01T00:00:00+05:30:00", rate=1.0)
    st = c.state()
    assert st["armed"] is True and st["virtual_seconds_per_real_second"] == 2.0 and st["t0"] == T0


def test_set_time_rejects_overflowing_t0():
    c = Clock(now_fn=FakeNow())
    with pytest.raises(ClockError):
        c.set_time("9999-12-31T00:00:00Z", rate=1.0)


def test_set_time_allows_max_t0_when_frozen():
    c = Clock(now_fn=FakeNow())
    c.set_time("9999-12-31T00:00:00Z", rate=0)  # frozen never advances
    assert c.read()["virtual_time"] == "9999-12-31T00:00:00Z"


def test_read_saturates_past_the_arm_time_horizon():
    """A t0 accepted by the horizon check still overflows if the gateway outlives it — reads saturate, not 500."""
    now = FakeNow()
    c = Clock(now_fn=now)
    c.set_time("9999-12-30T00:00:00Z", rate=1.0)  # accepted: >1 real day of headroom
    now.tick(86400.0 * 2)  # ...but still up two real days later
    assert c.read()["virtual_time"] == "9999-12-31T23:59:59.999999Z"
    assert c.state()["virtual_time"] == "9999-12-31T23:59:59.999999Z"


def test_saturated_clock_stays_monotonic():
    now = FakeNow()
    c = Clock(now_fn=now)
    c.set_time("9999-12-30T00:00:00Z", rate=1.0)
    seen = []
    for _ in range(4):
        now.tick(86400.0)
        seen.append(c.read()["virtual_time"])
    assert seen == sorted(seen)
    assert seen[-1] == "9999-12-31T23:59:59.999999Z"
