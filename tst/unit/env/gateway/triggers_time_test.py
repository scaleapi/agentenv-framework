"""Unit tests for clock time-triggers (when.type=='time') on the gateway trigger engine.

A FakeNow drives the injected Clock deterministically; async firing is awaited via _settle.
"""
from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest
from mcp.types import CallToolResult, TextContent

from agent_env.env.gateway import triggers as trig_mod
from agent_env.env.gateway.clock import Clock, ClockError, _parse_duration
from agent_env.env.gateway.triggers import TriggerEngine, TriggerError

T0 = "2026-01-01T00:00:00Z"


class FakeNow:
    def __init__(self, start: float = 1000.0):
        self.t = start

    def __call__(self) -> float:
        return self.t

    def tick(self, seconds: float) -> None:
        self.t += seconds


class _CountingClock(Clock):
    def __init__(self, now_fn):
        super().__init__(now_fn=now_fn)
        self.now_calls = 0

    def now(self):
        self.now_calls += 1
        return super().now()


class _FakeGateway:
    def __init__(self, clock: Clock):
        self._clock = clock
        self._server_tools: dict = {}
        self._tool_server_urls: dict = {}
        self._role_rules_lock = asyncio.Lock()
        self.rules: list = []
        self.TOOL_CALL_TIMEOUT_S = 30

    def _apply_rule(self, role, tool, value):
        self.rules.append((role, tool, value))

    def _query_changelog_id(self):
        return 0

    async def _log_event(self, event):
        return "e"

    async def _ensure_tools_discovered(self):
        pass


def _result(is_error: bool = False) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text="{}")], isError=is_error)


def _perm():
    """A no-tool permission action: reaches action_ok with no external dependency."""
    return {"type": "permission", "action": "disable", "role": "default", "tools": []}


def _make(counting: bool = False):
    now = FakeNow()
    clock = _CountingClock(now) if counting else Clock(now_fn=now)
    gw = _FakeGateway(clock)
    return TriggerEngine(gw), gw, clock, now


async def _settle(engine: TriggerEngine) -> None:
    for _ in range(200):
        if not engine._tasks:
            return
        await asyncio.sleep(0.01)


def _tick_call(engine, now, seconds, tool="read_tool", is_error=False):
    now.tick(seconds)
    engine.on_tool_call("default", tool, {}, _result(is_error))


# --------------------------------------------------------------------------- parsing


@pytest.mark.parametrize("text,expected", [
    ("PT24H", timedelta(hours=24)),
    ("P1DT2H30M", timedelta(days=1, hours=2, minutes=30)),
    ("PT0.5S", timedelta(seconds=0.5)),
    ("P1W", timedelta(weeks=1)),
    ("PT0S", timedelta(0)),
])
def test_parse_duration_ok(text, expected):
    assert _parse_duration(text) == expected


@pytest.mark.parametrize("bad", ["P1Y", "P1M", "", "P", "PT", "10m", "24H", "PT1H30", None, 5,
                                 "P1000000000D", "P999999999999W"])  # last two overflow timedelta
def test_parse_duration_rejects(bad):
    with pytest.raises(ClockError):
        _parse_duration(bad)


# --------------------------------------------------------------------------- fail-loud validation


@pytest.mark.parametrize("when,fragment", [
    ({"type": "time"}, "at least one of"),
    ({"type": "time", "at": T0, "after": "x", "offset": "PT1H"}, "at most one of"),
    ({"type": "time", "after": "x"}, "requires a non-empty 'offset'"),
    ({"type": "time", "at": "PT1H", "offset": "PT1H"}, "only valid with 'after'"),
    ({"type": "time", "every": "PT0S"}, "positive duration"),
    ({"type": "time", "every": {"dist": "exp", "mean": "PT10M"}}, "requires an integer 'seed'"),
    ({"type": "time", "every": {"dist": "poisson", "mean": "PT10M"}, "seed": 1}, "every.dist must be 'exp'"),
    ({"type": "time", "every": "PT10M", "seed": 1}, "only valid with a stochastic"),
    ({"type": "time", "at": T0, "count": 5}, "'count' requires a recurring"),
    ({"type": "time", "at": T0, "until": T0}, "'until' requires a recurring"),
    ({"type": "time", "every": "PT10M", "count": 0}, "count must be a positive integer"),
    ({"type": "time", "at": "not-a-date"}, "at"),
    ({"type": "time", "at": "P1Y"}, "at"),
    ({"type": "time", "every": "P1000000000D"}, "out of representable range"),  # 500->400 (review)
])
def test_time_validation_fails_loud(when, fragment):
    engine, *_ = _make()
    with pytest.raises(TriggerError) as ei:
        engine.register({"watch_roles": ["default"],
                         "triggers": [{"id": "t", "when": when, "actions": []}]})
    assert fragment in str(ei.value)


def test_validation_is_all_or_nothing():
    """A bad time trigger co-submitted with a valid action trigger adds neither (registration is atomic)."""
    engine, *_ = _make()
    with pytest.raises(TriggerError):
        engine.register({"watch_roles": ["default"], "triggers": [
            {"id": "ok", "when": {"type": "action", "tool": "t"}, "actions": []},
            {"id": "bad", "when": {"type": "time", "every": "PT0S"}, "actions": []},
        ]})
    assert engine.state()["triggers"] == []
    assert engine._has_time_triggers is False


# --------------------------------------------------------------------------- absolute one-shot


@pytest.mark.asyncio
async def test_absolute_one_shot_fires_once_when_crossed():
    engine, gw, clock, now = _make()
    clock.set_time(T0, rate=1.0)
    engine.register({"watch_roles": ["default"], "triggers": [
        {"id": "abs", "when": {"type": "time", "at": "2026-01-01T00:10:00Z"}, "actions": [_perm()]}]})
    _tick_call(engine, now, 300)  # +5 min: not yet due
    await _settle(engine)
    assert engine._triggers["abs"]["status"] == "armed" and engine._triggers["abs"]["fire_count"] == 0
    _tick_call(engine, now, 400)  # +8:20 more -> past 00:10
    await _settle(engine)
    assert engine._triggers["abs"]["status"] == "fired" and engine._triggers["abs"]["fire_count"] == 1
    _tick_call(engine, now, 1000)  # further calls never re-fire a one-shot
    await _settle(engine)
    assert engine._triggers["abs"]["fire_count"] == 1


@pytest.mark.asyncio
async def test_a_failing_action_still_retires_a_time_trigger():
    """An action/state trigger re-arms on failure; a time trigger cannot — failure clears next_mark and
    there is no "next call" to re-arm for — so it stays terminal. The tally records it either way."""
    engine, gw, clock, now = _make()
    clock.set_time(T0, rate=1.0)
    engine.register({"watch_roles": ["default"], "triggers": [
        {"id": "sched", "when": {"type": "time", "at": "2026-01-01T00:10:00Z"},
         "actions": [{"type": "tool", "tool": "unrouted_tool", "args": {}}]}]})
    _tick_call(engine, now, 700)  # past 00:10 -> due
    await _settle(engine)
    trig = engine._triggers["sched"]
    assert trig["status"] == "failed" and trig["next_mark"] is None and trig["fire_count"] == 0
    assert trig["failure_count"] == 1 and trig["last_failure_at"]
    assert [e["kind"] for e in engine._events if e["kind"] in ("failed", "fired")] == ["failed"]


@pytest.mark.asyncio
async def test_past_at_fires_once_immediately():
    engine, gw, clock, now = _make()
    clock.set_time(T0, rate=1.0)
    engine.register({"watch_roles": ["default"], "triggers": [
        {"id": "abs", "when": {"type": "time", "at": "2025-06-01T00:00:00Z"}, "actions": [_perm()]}]})
    _tick_call(engine, now, 1)  # first call: mark is already in the past -> fires
    await _settle(engine)
    assert engine._triggers["abs"]["status"] == "fired" and engine._triggers["abs"]["fire_count"] == 1


@pytest.mark.asyncio
async def test_relative_at_is_t0_plus_duration():
    engine, gw, clock, now = _make()
    clock.set_time(T0, rate=1.0)
    engine.register({"watch_roles": ["default"], "triggers": [
        {"id": "rel", "when": {"type": "time", "at": "PT1H"}, "actions": [_perm()]}]})
    _tick_call(engine, now, 3599)  # +59:59: not yet
    await _settle(engine)
    assert engine._triggers["rel"]["fire_count"] == 0
    _tick_call(engine, now, 2)  # cross t0+1h
    await _settle(engine)
    assert engine._triggers["rel"]["fire_count"] == 1


# --------------------------------------------------------------------------- event-anchored (flagship)


@pytest.mark.asyncio
async def test_event_anchored_flagship():
    engine, gw, clock, now = _make()
    clock.set_time(T0, rate=1.0)
    engine.register({"watch_roles": ["default"], "triggers": [
        {"id": "submit", "when": {"type": "action", "tool": "submit"}, "actions": []},
        {"id": "corr", "when": {"type": "time", "after": "submit", "offset": "PT24H"}, "actions": [_perm()]}]})
    # agent submits at +10 min -> stamps corr mark = virtual_now + 24h
    _tick_call(engine, now, 600, tool="submit")
    await _settle(engine)
    assert engine._triggers["submit"]["status"] == "fired"
    assert engine._triggers["corr"]["resolved"] is True and engine._triggers["corr"]["fire_count"] == 0
    anchored = [e for e in engine._events if e["kind"] == "anchored" and e["trigger_id"] == "corr"]
    assert anchored and anchored[0]["mark"] == "2026-01-02T00:10:00Z"
    # not yet due 23h later
    _tick_call(engine, now, 23 * 3600)
    await _settle(engine)
    assert engine._triggers["corr"]["fire_count"] == 0
    # cross +24h after submit
    _tick_call(engine, now, 3600)
    await _settle(engine)
    assert engine._triggers["corr"]["status"] == "fired" and engine._triggers["corr"]["fire_count"] == 1


@pytest.mark.asyncio
async def test_event_anchored_never_fires_without_anchor():
    engine, gw, clock, now = _make()
    clock.set_time(T0, rate=1.0)
    engine.register({"watch_roles": ["default"], "triggers": [
        {"id": "corr", "when": {"type": "time", "after": "ghost", "offset": "PT1H"}, "actions": [_perm()]}]})
    for _ in range(5):
        _tick_call(engine, now, 3600)
        await _settle(engine)
    assert engine._triggers["corr"]["status"] == "armed"
    assert engine._triggers["corr"]["resolved"] is False and engine._triggers["corr"]["fire_count"] == 0


@pytest.mark.parametrize("anchor_when", [
    {"type": "time", "at": "PT1M"},                        # one-shot absolute
    {"type": "time", "every": "PT1M", "count": 1},         # bounded recurrence, exhausted on its first fire
], ids=["one-shot", "bounded-recurrence"])
@pytest.mark.asyncio
async def test_terminating_time_anchor_does_not_strand_dependent(anchor_when):
    """Greptile #765: a TIME anchor that retires must still resolve its dependents, even though
    `_fire_time` recomputes `_has_time_triggers` (in `finally`) *before* calling `_resolve_dependents`,
    which early-returns on a cleared flag.

    It cannot be cleared here, and that is the invariant this pins: an unresolved `after` dependent is
    itself a time trigger stuck in `armed` (`_resolve_first_mark` returns early for `after`, so it never
    gets a mark and can never be retired by the eval loop). So `_recompute_time_flag` always counts it and
    the flag stays set for exactly as long as a strandable dependent exists. If a future change lets an
    unresolved dependent leave `armed` — or narrows the flag to *resolved* marks — this goes red instead
    of silently dropping the dependent's actions."""
    engine, gw, clock, now = _make()
    clock.set_time(T0, rate=1.0)
    engine.register({"watch_roles": ["default"], "triggers": [
        {"id": "anchor", "when": anchor_when, "actions": []},
        {"id": "dep", "when": {"type": "time", "after": "anchor", "offset": "PT1H"}, "actions": [_perm()]}]})
    _tick_call(engine, now, 120)  # anchor's mark (t0+1m) is due -> fires and retires
    await _settle(engine)
    assert engine._triggers["anchor"]["status"] == "fired"  # terminated: cannot fire again
    assert engine._has_time_triggers is True, "the armed dependent must keep the flag alive"
    assert engine._triggers["dep"]["resolved"] is True, "terminating anchor stranded its dependent"
    anchored = [e for e in engine._events if e["kind"] == "anchored" and e["trigger_id"] == "dep"]
    assert len(anchored) == 1 and anchored[0]["anchor"] == "anchor"
    _tick_call(engine, now, 3600)  # cross the stamped mark (+1h after the anchor fired)
    await _settle(engine)
    assert engine._triggers["dep"]["status"] == "fired" and engine._triggers["dep"]["fire_count"] == 1


# --------------------------------------------------------------------------- fixed recurring + catch-up + cap


@pytest.mark.asyncio
async def test_fixed_recurring_catch_up():
    engine, gw, clock, now = _make()
    clock.set_time(T0, rate=1.0)
    engine.register({"watch_roles": ["default"], "triggers": [
        {"id": "rec", "when": {"type": "time", "every": "PT10M"}, "actions": [_perm()]}]})
    # first mark is t0+10m; jump +35m so marks at 10/20/30 are all due at once
    _tick_call(engine, now, 35 * 60)
    await _settle(engine)
    assert engine._triggers["rec"]["fire_count"] == 3 and engine._triggers["rec"]["status"] == "armed"
    # +10m more -> one more
    _tick_call(engine, now, 10 * 60)
    await _settle(engine)
    assert engine._triggers["rec"]["fire_count"] == 4


@pytest.mark.asyncio
async def test_recurring_count_terminates():
    engine, gw, clock, now = _make()
    clock.set_time(T0, rate=1.0)
    engine.register({"watch_roles": ["default"], "triggers": [
        {"id": "rec", "when": {"type": "time", "every": "PT1M", "count": 2}, "actions": [_perm()]}]})
    _tick_call(engine, now, 3600)  # way past many intervals, but count caps arrivals at 2
    await _settle(engine)
    assert engine._triggers["rec"]["fire_count"] == 2 and engine._triggers["rec"]["status"] == "fired"


@pytest.mark.asyncio
async def test_recurring_until_terminates():
    engine, gw, clock, now = _make()
    clock.set_time(T0, rate=1.0)
    engine.register({"watch_roles": ["default"], "triggers": [
        {"id": "rec", "when": {"type": "time", "every": "PT10M", "until": "2026-01-01T00:25:00Z"},
         "actions": [_perm()]}]})
    _tick_call(engine, now, 3600)  # marks 10,20 <= until(25); 30 > until -> stop
    await _settle(engine)
    assert engine._triggers["rec"]["fire_count"] == 2 and engine._triggers["rec"]["status"] == "fired"


@pytest.mark.asyncio
async def test_catch_up_capped(monkeypatch):
    monkeypatch.setattr(trig_mod, "_MAX_CATCHUP", 5)
    engine, gw, clock, now = _make()
    clock.set_time(T0, rate=1.0)
    engine.register({"watch_roles": ["default"], "triggers": [
        {"id": "rec", "when": {"type": "time", "every": "PT1S"}, "actions": [_perm()]}]})
    _tick_call(engine, now, 3600)  # 3600 marks due, cap to 5 this call
    await _settle(engine)
    assert engine._triggers["rec"]["fire_count"] == 5
    assert engine._triggers["rec"]["status"] == "armed"  # backlog remains -> re-armed
    assert any(e["kind"] == "fired" and e.get("capped") for e in engine._events)
    # the backlog drains on the next call (another <=5)
    _tick_call(engine, now, 0)
    await _settle(engine)
    assert engine._triggers["rec"]["fire_count"] == 10


# --------------------------------------------------------------------------- stochastic reproducibility


def _stochastic_trig(engine, seed):
    engine.register({"watch_roles": ["default"], "triggers": [
        {"id": "s", "when": {"type": "time", "every": {"dist": "exp", "mean": "PT10M"}, "seed": seed},
         "actions": [_perm()]}]})
    return engine._triggers["s"]


def test_seeded_interval_sequence_is_reproducible():
    e1, *_ = _make()
    e2, *_ = _make()
    t1, t2 = _stochastic_trig(e1, 42), _stochastic_trig(e2, 42)
    seq1 = [e1._next_interval_seconds(t1) for _ in range(20)]
    seq2 = [e2._next_interval_seconds(t2) for _ in range(20)]
    assert seq1 == seq2
    assert all(x > 0 for x in seq1)
    # mean of the sequence is roughly the configured 600s (sanity, wide tolerance)
    assert 300 < (sum(seq1) / len(seq1)) < 1200


def test_seeded_interval_sequence_differs_by_seed():
    e1, *_ = _make()
    e2, *_ = _make()
    t1, t2 = _stochastic_trig(e1, 42), _stochastic_trig(e2, 43)
    seq1 = [e1._next_interval_seconds(t1) for _ in range(20)]
    seq2 = [e2._next_interval_seconds(t2) for _ in range(20)]
    assert seq1 != seq2


@pytest.mark.asyncio
async def test_stochastic_fires_reproducibly_across_runs():
    """Same seed + same virtual-time schedule -> identical arrival count (rate-independent)."""
    async def run(seed, rate):
        engine, gw, clock, now = _make()
        clock.set_time(T0, rate=rate)
        engine.register({"watch_roles": ["default"], "triggers": [
            {"id": "s", "when": {"type": "time", "every": {"dist": "exp", "mean": "PT5M"}, "seed": seed},
             "actions": [_perm()]}]})
        # advance ~1 virtual hour of monotonic time (scaled by rate) in one shot
        now.tick((3600 / rate))
        engine.on_tool_call("default", "read_tool", {}, _result())
        await _settle(engine)
        return engine._triggers["s"]["fire_count"]
    a = await run(7, rate=1.0)
    b = await run(7, rate=60.0)   # different real->virtual rate, same seed
    c = await run(8, rate=1.0)
    assert a == b and a > 0 and c != a  # rate-independent; seed determines the count


# --------------------------------------------------------------------------- off-by-default / no-background-timer


def test_off_by_default_never_touches_clock():
    engine, gw, clock, now = _make(counting=True)
    engine.register({"watch_roles": ["default"], "triggers": [
        {"id": "a", "when": {"type": "action", "tool": "grant_tool"}, "actions": []}]})
    assert engine._has_time_triggers is False
    clock.now_calls = 0  # registration stamps its own events; measure only the tool-call path
    for _ in range(5):  # a non-matching watched call: no fire, and the clock is never consulted
        engine.on_tool_call("default", "other_tool", {}, _result())
    assert clock.now_calls == 0


@pytest.mark.asyncio
async def test_unarmed_clock_defers_then_fires_once_armed():
    engine, gw, clock, now = _make()
    engine.register({"watch_roles": ["default"], "triggers": [
        {"id": "abs", "when": {"type": "time", "at": "PT10M"}, "actions": [_perm()]}]})
    _tick_call(engine, now, 10_000)  # clock unarmed -> now() None -> no fire
    await _settle(engine)
    assert engine._triggers["abs"]["fire_count"] == 0 and engine._triggers["abs"]["resolved"] is False
    clock.set_time(T0, rate=1.0)
    _tick_call(engine, now, 3600)  # now armed and past t0+10m
    await _settle(engine)
    assert engine._triggers["abs"]["fire_count"] == 1


@pytest.mark.asyncio
async def test_time_advance_alone_is_inert_without_driver_or_calls():
    """No hidden timer: with no watched call and no driver, advancing virtual time fires nothing."""
    engine, gw, clock, now = _make()
    clock.set_time(T0, rate=1.0)
    engine.register({"watch_roles": ["default"], "triggers": [
        {"id": "rec", "when": {"type": "time", "every": "PT1M"}, "actions": [_perm()]}]})
    now.tick(3600)  # an hour of virtual time passes with no tool call and no driver started
    await _settle(engine)
    assert engine._triggers["rec"]["status"] == "armed" and engine._triggers["rec"]["fire_count"] == 0


@pytest.mark.asyncio
async def test_rearm_reanchors_relative_marks():
    """A resolved relative mark re-anchors to the new t0 on re-arm, not the old one."""
    engine, gw, clock, now = _make()
    clock.set_time("2026-01-01T00:00:00Z", rate=1.0)  # t0a
    engine.register({"watch_roles": ["default"], "triggers": [
        {"id": "rel", "when": {"type": "time", "at": "PT1H"}, "actions": [_perm()]}]})
    _tick_call(engine, now, 60)  # resolves mark = t0a + 1h; not due
    await _settle(engine)
    assert engine._triggers["rel"]["fire_count"] == 0
    assert engine._triggers["rel"]["next_mark"].year == 2026
    clock.set_time("2030-01-01T00:00:00Z", rate=1.0)  # re-arm at a far-future t0b
    _tick_call(engine, now, 60)  # generation changed -> re-anchor; must NOT fire against the stale 2026 mark
    await _settle(engine)
    assert engine._triggers["rel"]["fire_count"] == 0, "stale 2026 mark fired against the 2030 clock"
    assert engine._triggers["rel"]["next_mark"].year == 2030, "mark did not re-anchor to the new t0"
    _tick_call(engine, now, 3600)  # cross t0b + 1h
    await _settle(engine)
    assert engine._triggers["rel"]["fire_count"] == 1


@pytest.mark.asyncio
async def test_rearm_does_not_strand_anchored_trigger():
    """Re-arm must not clear a resolved anchored trigger: its fire-once anchor can't re-fire."""
    engine, gw, clock, now = _make()
    clock.set_time("2026-01-01T00:00:00Z", rate=1.0)
    engine.register({"watch_roles": ["default"], "triggers": [
        {"id": "anchor", "when": {"type": "action", "tool": "submit"}, "actions": []},
        {"id": "dep", "when": {"type": "time", "after": "anchor", "offset": "PT1H"}, "actions": [_perm()]}]})
    _tick_call(engine, now, 60, tool="submit")  # anchor fires -> dep resolves (mark = now + 1h)
    await _settle(engine)
    assert engine._triggers["dep"]["resolved"] is True and engine._triggers["dep"]["fire_count"] == 0
    clock.set_time("2026-03-01T00:00:00Z", rate=1.0)  # re-arm (generation bumps) before dep's mark
    _tick_call(engine, now, 60)  # reset loop must skip the anchored dep (not strand it)
    await _settle(engine)
    assert engine._triggers["dep"]["status"] == "fired" and engine._triggers["dep"]["fire_count"] == 1, \
        "re-arm stranded the resolved anchored trigger"


@pytest.mark.asyncio
async def test_relative_overflow_saturates_and_does_not_wedge():
    """A huge but valid duration saturates instead of raising and wedging the eval loop."""
    engine, gw, clock, now = _make()
    clock.set_time("2026-01-01T00:00:00Z", rate=1.0)
    engine.register({"watch_roles": ["default"], "triggers": [
        {"id": "overflow", "when": {"type": "time", "at": "P999999999D"}, "actions": [_perm()]},  # registered first
        {"id": "normal", "when": {"type": "time", "at": "PT1M"}, "actions": [_perm()]}]})
    _tick_call(engine, now, 120)  # normal (t0+1m) is due; overflow saturates far in the future
    await _settle(engine)
    assert engine._triggers["normal"]["fire_count"] == 1, "overflow trigger wedged the eval loop"
    assert engine._triggers["overflow"]["status"] == "armed"
    assert engine._triggers["overflow"]["resolved"] is True  # resolved (saturated), no raise
    assert engine._triggers["overflow"]["fire_count"] == 0


@pytest.mark.asyncio
async def test_rearm_reseeds_stochastic_recurrence():
    """A re-armed stochastic recurrence restarts its RNG, so seed+t0 reproduces a fresh trigger."""
    # reference: a fresh trigger (seed 42) armed at t0b
    e_ref, _, clock_ref, now_ref = _make()
    clock_ref.set_time("2030-01-01T00:00:00Z", rate=1.0)
    _stochastic_trig(e_ref, 42)
    _tick_call(e_ref, now_ref, 1)
    ref_mark = e_ref._triggers["s"]["next_mark"]
    # subject: register at t0a, consume several arrivals (advancing the RNG), then re-arm at t0b
    e, _, clock, now = _make()
    clock.set_time("2026-01-01T00:00:00Z", rate=1.0)
    _stochastic_trig(e, 42)
    _tick_call(e, now, 100000)  # ~27 virtual h: several ~10-min arrivals -> RNG advanced
    await _settle(e)
    assert e._triggers["s"]["fire_count"] > 0
    clock.set_time("2030-01-01T00:00:00Z", rate=1.0)  # re-arm at the same t0b as the reference
    _tick_call(e, now, 1)
    assert e._triggers["s"]["next_mark"] == ref_mark, "re-arm did not reproduce a fresh trigger's schedule"


@pytest.mark.asyncio
async def test_rearm_reanchors_trigger_that_was_firing():
    """A trigger mid-fire during a re-arm still re-anchors once it returns to armed."""
    engine, gw, clock, now = _make()
    clock.set_time("2026-01-01T00:00:00Z", rate=1.0)
    engine.register({"watch_roles": ["default"], "triggers": [
        {"id": "rec", "when": {"type": "time", "every": "PT1H"}, "actions": [_perm()]}]})
    _tick_call(engine, now, 60)  # resolve first mark (t0a + 1h), mark_gen = 1
    await _settle(engine)
    t = engine._triggers["rec"]
    assert t["mark_gen"] == 1
    t["status"] = "firing"                              # simulate an in-flight async fire...
    clock.set_time("2030-01-01T00:00:00Z", rate=1.0)    # ...during which the clock is re-armed
    _tick_call(engine, now, 60)                         # eval skips it (firing) -> not yet re-anchored
    await _settle(engine)
    assert t["mark_gen"] == 1 and t["next_mark"].year == 2026
    t["status"] = "armed"                               # fire completes -> back to armed
    _tick_call(engine, now, 60)
    await _settle(engine)
    assert t["mark_gen"] == 2 and t["next_mark"].year == 2030, "firing-then-armed trigger kept a stale mark"


@pytest.mark.asyncio
async def test_rearm_drops_capped_backlog_and_says_so(monkeypatch):
    """Greptile P1c: a re-arm that lands while a capped catch-up batch is firing drops the leftover
    backlog — by design (those marks are on a timeline the new clock never reached) — but the drop must
    NOT be silent: it emits a `reanchored` event naming the dropped mark, and the trigger re-anchors to
    the new t0 rather than replaying the old axis."""
    monkeypatch.setattr(trig_mod, "_MAX_CATCHUP", 5)
    engine, gw, clock, now = _make()
    clock.set_time(T0, rate=1.0)
    engine.register({"watch_roles": ["default"], "triggers": [
        {"id": "rec", "when": {"type": "time", "every": "PT1S"}, "actions": [_perm()]}]})
    _tick_call(engine, now, 3600)  # 3600 marks due -> capped at 5, backlog of ~3595 left on next_mark
    await _settle(engine)
    t = engine._triggers["rec"]
    assert t["fire_count"] == 5 and t["status"] == "armed"
    backlog_mark = t["next_mark"]
    assert backlog_mark is not None and backlog_mark.year == 2026  # old axis

    t["status"] = "firing"                              # the capped batch is still draining...
    clock.set_time("2030-01-01T00:00:00Z", rate=1.0)    # ...when the clock is re-armed at a new t0
    t["status"] = "armed"                               # fire completes -> back to armed
    _tick_call(engine, now, 0)
    await _settle(engine)

    drops = [e for e in engine._events if e["kind"] == "reanchored"]
    assert len(drops) == 1, "the dropped backlog was silent"
    assert drops[0]["dropped_mark"] == backlog_mark.isoformat().replace("+00:00", "Z")
    # re-anchored to the new t0 (2030 + PT1S), not replaying the 2026 backlog
    assert t["mark_gen"] == 2 and t["next_mark"].year == 2030
    assert t["fire_count"] == 5, "old-axis backlog fired against the new clock"


def test_advance_marks_saturated_retires_not_infinite():
    """A mark saturated at datetime.max retires the trigger instead of re-bursting every call."""
    from datetime import datetime, timezone
    engine, *_ = _make()
    MAX = datetime.max.replace(tzinfo=timezone.utc)
    trig = {"spec": {"when": {"type": "time", "every": "PT30M"}}, "status": "armed",
            "next_mark": MAX, "resolved": True, "fire_count": 0, "rng": None}
    marks, capped, terminated = engine._advance_marks(trig, MAX)  # now == next_mark == datetime.max
    assert len(marks) == 1 and terminated is True and capped is False  # fires once at the ceiling, then retires
    assert trig["next_mark"] is None


@pytest.mark.asyncio
async def test_errored_call_still_ticks_time_triggers():
    """An errored tool call is still a legitimate clock tick (regression: time eval before the isError guard)."""
    engine, gw, clock, now = _make()
    clock.set_time(T0, rate=1.0)
    engine.register({"watch_roles": ["default"], "triggers": [
        {"id": "abs", "when": {"type": "time", "at": "PT5M"}, "actions": [_perm()]}]})
    _tick_call(engine, now, 3600, is_error=True)  # errored result, but time trigger is due
    await _settle(engine)
    assert engine._triggers["abs"]["fire_count"] == 1


# --------------------------------------------------------------------------- autonomous background driver
# A fast interval + a hand-ticked FakeNow, then a few real poll cycles.


async def _spin(seconds: float = 0.06) -> None:
    """Let the driver run a few poll cycles."""
    await asyncio.sleep(seconds)


@pytest.mark.asyncio
async def test_driver_fires_recurring_with_zero_tool_calls():
    """An `every` trigger fires on virtual-time passage alone — no on_tool_call at all."""
    engine, gw, clock, now = _make()
    clock.set_time(T0, rate=1.0)
    engine._driver_interval = 0.01
    engine.register({"watch_roles": ["default"], "triggers": [
        {"id": "rec", "when": {"type": "time", "every": "PT10M"}, "actions": [_perm()]}]})
    engine.start_driver()
    now.tick(35 * 60)  # marks at 10/20/30 min all become due; NO tool call is ever made
    await _spin()
    await _settle(engine)
    await engine.stop_driver()
    assert engine._triggers["rec"]["fire_count"] == 3 and engine._triggers["rec"]["status"] == "armed"


@pytest.mark.asyncio
async def test_driver_fires_absolute_with_zero_tool_calls():
    engine, gw, clock, now = _make()
    clock.set_time(T0, rate=1.0)
    engine._driver_interval = 0.01
    engine.register({"watch_roles": ["default"], "triggers": [
        {"id": "abs", "when": {"type": "time", "at": "PT10M"}, "actions": [_perm()]}]})
    engine.start_driver()
    now.tick(15 * 60)  # past t0+10m
    await _spin()
    await _settle(engine)
    await engine.stop_driver()
    assert engine._triggers["abs"]["status"] == "fired" and engine._triggers["abs"]["fire_count"] == 1


@pytest.mark.asyncio
async def test_driver_fires_event_anchored_offset_autonomously():
    """The anchor is one agent action; the +offset mark then fires with no further calls."""
    engine, gw, clock, now = _make()
    clock.set_time(T0, rate=1.0)
    engine._driver_interval = 0.01
    engine.register({"watch_roles": ["default"], "triggers": [
        {"id": "submit", "when": {"type": "action", "tool": "submit"}, "actions": []},
        {"id": "corr", "when": {"type": "time", "after": "submit", "offset": "PT24H"}, "actions": [_perm()]}]})
    engine.start_driver()
    _tick_call(engine, now, 600, tool="submit")  # the one agent action: stamps corr mark = now + 24h
    await _settle(engine)
    assert engine._triggers["corr"]["resolved"] is True and engine._triggers["corr"]["fire_count"] == 0
    now.tick(24 * 3600 + 60)  # 24 virtual hours pass with NO further tool calls
    await _spin()
    await _settle(engine)
    await engine.stop_driver()
    assert engine._triggers["corr"]["status"] == "fired" and engine._triggers["corr"]["fire_count"] == 1


@pytest.mark.asyncio
async def test_start_driver_idempotent():
    engine, *_ = _make()
    engine.start_driver()
    first = engine._driver_task
    engine.start_driver()  # a second start while the first is alive is a no-op
    assert engine._driver_task is first
    await engine.stop_driver()
    assert engine._driver_task is None


@pytest.mark.asyncio
async def test_stop_driver_cancels_and_is_idempotent():
    engine, *_ = _make()
    engine.start_driver()
    task = engine._driver_task
    await engine.stop_driver()
    assert task.done() and engine._driver_task is None
    await engine.stop_driver()  # stopping an already-stopped driver is a no-op (no leak, no raise)


@pytest.mark.asyncio
async def test_driver_off_by_default_never_reads_clock():
    """With no time-trigger armed the driver never consults the clock."""
    engine, gw, clock, now = _make(counting=True)
    clock.set_time(T0, rate=1.0)
    engine._driver_interval = 0.01
    engine.register({"watch_roles": ["default"], "triggers": [
        {"id": "a", "when": {"type": "action", "tool": "grant_tool"}, "actions": []}]})
    assert engine._has_time_triggers is False
    clock.now_calls = 0  # registration stamps its own events; measure only the driver ticks
    engine.start_driver()
    await _spin()
    await engine.stop_driver()
    assert clock.now_calls == 0


@pytest.mark.asyncio
async def test_driver_and_tool_call_no_double_fire():
    """A mark seen by both the driver and a concurrent tool call fires exactly once (status guard)."""
    engine, gw, clock, now = _make()
    clock.set_time(T0, rate=1.0)
    engine.register({"watch_roles": ["default"], "triggers": [
        {"id": "abs", "when": {"type": "time", "at": "PT10M"}, "actions": [_perm()]}]})
    now.tick(15 * 60)  # due
    engine._eval_time_triggers()  # driver evaluation -> status 'firing'
    engine.on_tool_call("default", "read_tool", {}, _result())  # tool-call evaluation sees 'firing' -> skips
    await _settle(engine)
    assert engine._triggers["abs"]["fire_count"] == 1


@pytest.mark.asyncio
async def test_driver_started_before_triggers_registered():
    """The lifespan starts the driver before any trigger exists; a later one must still be picked up."""
    engine, gw, clock, now = _make()
    engine._driver_interval = 0.01
    engine.start_driver()  # started against an empty engine, exactly as the lifespan does
    await _spin(0.03)      # idle ticks: nothing armed -> nothing fires
    clock.set_time(T0, rate=1.0)
    engine.register({"watch_roles": ["default"], "triggers": [
        {"id": "rec", "when": {"type": "time", "every": "PT10M"}, "actions": [_perm()]}]})
    now.tick(35 * 60)
    await _spin()
    await _settle(engine)
    assert engine._driver_task is not None and not engine._driver_task.done()
    assert engine._triggers["rec"]["fire_count"] == 3
    await engine.stop_driver()


@pytest.mark.asyncio
async def test_driver_tick_exception_does_not_kill_poller(monkeypatch):
    """Fail-open: a tick that raises is logged and swallowed; the poller survives and resumes firing."""
    engine, gw, clock, now = _make()
    clock.set_time(T0, rate=1.0)
    engine.register({"watch_roles": ["default"], "triggers": [
        {"id": "rec", "when": {"type": "time", "every": "PT10M"}, "actions": [_perm()]}]})
    real = engine._eval_time_triggers
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")  # the first tick raises
        return real()

    monkeypatch.setattr(engine, "_eval_time_triggers", flaky)
    engine._driver_interval = 0.01
    engine.start_driver()
    now.tick(35 * 60)
    await _spin(0.1)  # several ticks: the first raises, the rest work
    await _settle(engine)
    assert not engine._driver_task.done(), "poller died on a tick exception"
    assert engine._triggers["rec"]["fire_count"] >= 1, "poller did not resume firing after the exception"
    await engine.stop_driver()


@pytest.mark.asyncio
async def test_driver_restart_after_stop():
    """start -> stop -> start yields a fresh, working poller (not a silent no-op)."""
    engine, gw, clock, now = _make()
    engine._driver_interval = 0.01
    engine.start_driver()
    first = engine._driver_task
    await engine.stop_driver()
    engine.start_driver()  # restart
    assert engine._driver_task is not None and engine._driver_task is not first and not engine._driver_task.done()
    clock.set_time(T0, rate=1.0)
    engine.register({"watch_roles": ["default"], "triggers": [
        {"id": "abs", "when": {"type": "time", "at": "PT5M"}, "actions": [_perm()]}]})
    now.tick(10 * 60)
    await _spin()
    await _settle(engine)
    assert engine._triggers["abs"]["fire_count"] == 1
    await engine.stop_driver()


# --------------------------------------------------------------------------- observability
# `mark` is the instant an arrival was DUE (reproducible); `virtual_time` is when it was emitted.


@pytest.mark.asyncio
async def test_fired_events_carry_mark_and_virtual_time():
    engine, gw, clock, now = _make()
    clock.set_time(T0, rate=1.0)
    engine.register({"watch_roles": ["default"], "triggers": [
        {"id": "rec", "when": {"type": "time", "every": "PT10M"}, "actions": [_perm()]}]})
    _tick_call(engine, now, 35 * 60)  # marks at +10/+20/+30 all due in one evaluation
    await _settle(engine)
    fired = [e for e in engine._events if e["kind"] == "fired"]
    assert [e["mark"] for e in fired] == ["2026-01-01T00:10:00Z", "2026-01-01T00:20:00Z", "2026-01-01T00:30:00Z"], \
        "each arrival must record the distinct virtual instant it was scheduled for"
    # all three drained in the same evaluation -> identical real ts, so `mark` is the only separator
    assert all(e["virtual_time"] == "2026-01-01T00:35:00Z" for e in fired)
    detected = next(e for e in engine._events if e["kind"] == "detected")
    assert detected["due"] == 3
    assert detected["first_mark"] == "2026-01-01T00:10:00Z" and detected["last_mark"] == "2026-01-01T00:30:00Z"


@pytest.mark.asyncio
async def test_time_trigger_provoking_is_clock_on_both_eval_paths():
    """Attribution must not depend on which path observed the mark."""
    engine, gw, clock, now = _make()
    clock.set_time(T0, rate=1.0)
    engine.register({"watch_roles": ["default"], "triggers": [
        {"id": "a", "when": {"type": "time", "at": "PT5M"}, "actions": [_perm()]},
        {"id": "b", "when": {"type": "time", "at": "PT15M"}, "actions": [_perm()]}]})
    _tick_call(engine, now, 10 * 60)          # observed by a tool call
    await _settle(engine)
    now.tick(10 * 60); engine._eval_time_triggers()  # observed by the driver
    await _settle(engine)
    provoking = [e["provoking"] for e in engine._events if e["kind"] == "detected"]
    assert provoking == [{"source": "clock"}, {"source": "clock"}]


def test_events_carry_no_virtual_time_when_clock_unarmed():
    engine, *_ = _make()
    engine.register({"watch_roles": ["default"], "triggers": [
        {"id": "a", "when": {"type": "action", "tool": "t"}, "actions": []}]})
    assert all("virtual_time" not in e for e in engine._events)


def test_state_exposes_type_and_schedule():
    engine, *_ = _make()
    engine.register({"watch_roles": ["default"], "triggers": [
        {"id": "act", "when": {"type": "action", "tool": "t"}, "actions": []},
        {"id": "beat", "when": {"type": "time", "every": "PT1H", "count": 6}, "actions": [_perm()]}]})
    rows = {t["id"]: t for t in engine.state()["triggers"]}
    assert rows["act"]["type"] == "action"
    assert rows["beat"]["type"] == "time"
    assert rows["beat"]["when"] == {"type": "time", "every": "PT1H", "count": 6}


@pytest.mark.asyncio
async def test_action_trigger_increments_fire_count():
    """status=='fired' with fire_count==0 made a cross-type fire tally silently wrong."""
    engine, gw, clock, now = _make()
    engine.register({"watch_roles": ["default"], "triggers": [
        {"id": "act", "when": {"type": "action", "tool": "grant"}, "actions": [_perm()]}]})
    engine.on_tool_call("default", "grant", {}, _result())
    await _settle(engine)
    assert engine._triggers["act"]["fire_count"] == 1
    assert engine.state()["triggers"][0]["fire_count"] == 1


def test_event_log_is_bounded_and_reports_drops(monkeypatch):
    monkeypatch.setattr(trig_mod, "_EVENTS_HEAD", 3)
    engine, *_ = _make()
    engine._events_tail = trig_mod.deque(maxlen=4)  # tiny head+tail for the test
    for i in range(20):
        engine._emit("added", f"t{i}")
    state = engine.state()
    assert len(state["events"]) == 7                      # 3 head + 4 tail, middle elided
    assert state["events_dropped"] == 13
    assert [e["trigger_id"] for e in state["events"][:3]] == ["t0", "t1", "t2"]     # earliest kept
    assert [e["trigger_id"] for e in state["events"][-2:]] == ["t18", "t19"]        # newest kept


@pytest.mark.asyncio
async def test_driver_parks_again_once_all_time_triggers_terminate():
    """A finished trigger lets the poller park again."""
    engine, gw, clock, now = _make()
    clock.set_time(T0, rate=1.0)
    engine.register({"watch_roles": ["default"], "triggers": [
        {"id": "once", "when": {"type": "time", "at": "PT5M"}, "actions": [_perm()]}]})
    assert engine._has_time_triggers is True and engine._time_trigger_gate.is_set()
    _tick_call(engine, now, 10 * 60)
    await _settle(engine)
    assert engine._triggers["once"]["status"] == "fired"
    assert engine._has_time_triggers is False and not engine._time_trigger_gate.is_set()


@pytest.mark.asyncio
async def test_gate_survives_a_terminating_peer_while_another_is_mid_fire():
    """The gate must count "firing": a peer terminating in the same evaluation recomputes it."""
    engine, gw, clock, now = _make()
    clock.set_time(T0, rate=1.0)
    engine.register({"watch_roles": ["default"], "triggers": [
        {"id": "live", "when": {"type": "time", "every": "PT10M"}, "actions": [_perm()]},
        # `until` already in the past at evaluation -> retires without firing, recomputing the flag
        {"id": "expired", "when": {"type": "time", "every": "PT10M", "until": "2026-01-01T00:05:00Z"},
         "actions": [_perm()]}]})
    now.tick(20 * 60)
    engine._eval_time_triggers()          # `live` -> firing (no recompute), `expired` -> recompute
    assert engine._triggers["live"]["status"] == "firing"
    assert engine._triggers["expired"]["status"] == "fired"
    assert engine._has_time_triggers is True, "a mid-fire trigger must keep the driver awake"
    assert engine._time_trigger_gate.is_set()
    await _settle(engine)
    assert engine._triggers["live"]["fire_count"] == 2 and engine._has_time_triggers is True


def test_state_when_echo_excludes_unbounded_selectors():
    """`when` is projected: a state trigger's `check` must not enter the log."""
    engine, *_ = _make()
    engine.register({"watch_roles": ["default"], "triggers": [
        {"id": "st", "when": {"type": "state", "check": {"tool": "t", "predicate": {"exists": True}}},
         "actions": []}]})
    row = engine.state()["triggers"][0]
    assert row["type"] == "state" and row["when"] == {"type": "state"}


def test_time_trigger_gate_mirrors_has_time_triggers():
    """The park gate is set iff a time-trigger can still fire."""
    engine, *_ = _make()
    engine.register({"watch_roles": ["default"], "triggers": [
        {"id": "a", "when": {"type": "action", "tool": "t"}, "actions": []}]})
    assert engine._has_time_triggers is False and not engine._time_trigger_gate.is_set()
    engine.register({"watch_roles": ["default"], "triggers": [
        {"id": "tt", "when": {"type": "time", "every": "PT10M"}, "actions": [_perm()]}]})
    assert engine._has_time_triggers is True and engine._time_trigger_gate.is_set()
    engine.remove({"ids": ["tt"]})
    assert engine._has_time_triggers is False and not engine._time_trigger_gate.is_set()
    engine.register({"watch_roles": ["default"], "triggers": [
        {"id": "tt2", "when": {"type": "time", "at": "PT1H"}, "actions": [_perm()]}]})
    assert engine._time_trigger_gate.is_set()
    engine.clear()
    assert not engine._time_trigger_gate.is_set()


@pytest.mark.asyncio
async def test_driver_parks_until_time_trigger_registered():
    """A started driver parks until a time-trigger exists, then wakes and fires."""
    engine, gw, clock, now = _make(counting=True)
    clock.set_time(T0, rate=1.0)
    engine._driver_interval = 0.01
    engine.start_driver()
    await _spin(0.05)  # no time-trigger yet -> parked -> the clock is never consulted
    assert clock.now_calls == 0 and not engine._driver_task.done()
    engine.register({"watch_roles": ["default"], "triggers": [
        {"id": "rec", "when": {"type": "time", "every": "PT10M"}, "actions": [_perm()]}]})
    now.tick(35 * 60)
    await _spin()
    await _settle(engine)
    await engine.stop_driver()
    assert clock.now_calls > 0 and engine._triggers["rec"]["fire_count"] == 3
