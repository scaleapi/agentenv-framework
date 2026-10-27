"""Clock.advance: a forward jump on the current arm, so time triggers keep their marks, and its PUT /clock/advance route."""
from __future__ import annotations

import asyncio

import httpx
import pytest

from agent_env.env.gateway import gateway as gw_mod
from agent_env.env.gateway import triggers as trig_mod
from agent_env.env.gateway.clock import Clock, ClockError
from agent_env.env.gateway.constants import EXT_CLOCK_URI, GATEWAY_EXTENSIONS
from agent_env.env.gateway.gateway import Gateway
from agent_env.env.gateway.triggers import TriggerEngine
from tst.unit.env.gateway.triggers_time_test import FakeNow, _FakeGateway, _perm, _settle, _tick_call

T0 = "2026-01-01T00:00:00Z"


def _armed(rate: float = 1.0) -> tuple[Clock, FakeNow]:
    now = FakeNow()
    clock = Clock(now_fn=now)
    clock.set_time(T0, rate=rate)
    return clock, now


def test_advance_jumps_forward_and_keeps_t0_rate_and_generation():
    clock, now = _armed(rate=2.0)
    gen = clock.generation
    state = clock.advance("2026-01-03T00:00:00Z")
    assert state["virtual_time"] == "2026-01-03T00:00:00Z"
    assert (state["t0"], state["virtual_seconds_per_real_second"], clock.generation) == (T0, 2.0, gen)
    now.tick(30)
    assert clock.read()["virtual_time"] == "2026-01-03T00:01:00Z"


def test_advance_on_a_frozen_clock_holds_there():
    clock, now = _armed(rate=0)
    clock.advance("2026-01-02T12:00:00Z")
    now.tick(3600)
    assert clock.read()["virtual_time"] == "2026-01-02T12:00:00Z"


def test_elapsed_counts_the_jump():
    clock, now = _armed()
    now.tick(10)
    clock.advance("2026-01-01T01:00:10Z")
    assert clock.elapsed() == pytest.approx(3610)


def test_advance_refuses_backwards_and_unarmed():
    clock, now = _armed()
    now.tick(60)
    with pytest.raises(ClockError, match="only moves forward"):
        clock.advance("2026-01-01T00:00:30Z")
    with pytest.raises(ClockError, match="not armed"):
        Clock(now_fn=FakeNow()).advance(T0)
    with pytest.raises(ClockError, match="RFC3339"):
        clock.advance("tomorrow")


def test_a_rearm_after_an_advance_starts_from_its_own_t0_and_banks_from_the_advanced_time():
    clock, now = _armed()
    clock.advance("2026-01-02T00:00:00Z")
    lived = clock.elapsed()
    clock.set_time("2026-01-02T06:00:00Z", rate=1.0)
    assert clock.read()["virtual_time"] == "2026-01-02T06:00:00Z"
    assert clock.elapsed() == pytest.approx(lived + 6 * 3600)
    clock.set_time("2026-01-01T00:00:00Z", rate=1.0)
    assert clock.read()["virtual_time"] == T0


def test_clear_drops_the_advance():
    clock, _ = _armed()
    clock.advance("2026-01-05T00:00:00Z")
    clock.clear()
    clock.set_time(T0, rate=0)
    assert clock.read()["virtual_time"] == T0


@pytest.mark.asyncio
async def test_a_relative_mark_comes_due_after_an_advance_with_no_reanchor():
    now = FakeNow()
    clock = Clock(now_fn=now)
    engine = TriggerEngine(_FakeGateway(clock))
    clock.set_time(T0, rate=0)
    engine.register({"watch_roles": ["default"], "triggers": [
        {"id": "rel", "when": {"type": "time", "at": "P2D"}, "actions": [_perm()]}]})
    _tick_call(engine, now, 1)
    clock.advance("2026-01-03T00:00:00Z")
    _tick_call(engine, now, 1)
    await _settle(engine)
    trig = engine.state()["triggers"][0]
    assert (trig["status"], trig["fire_count"]) == ("fired", 1)
    assert "reanchored" not in {e["kind"] for e in engine.state()["events"]}


@pytest.mark.asyncio
async def test_a_delay_counts_the_jumped_time(monkeypatch):
    monkeypatch.setattr(trig_mod, "_VIRTUAL_RECHECK_S", 0.01)
    now = FakeNow()
    clock = Clock(now_fn=now)
    engine = TriggerEngine(_FakeGateway(clock))
    clock.set_time(T0, rate=0)
    engine.register({"watch_roles": ["default"], "triggers": [
        {"id": "slow", "when": {"type": "time", "at": "PT1M"}, "actions": [{**_perm(), "delay": "PT6H"}]}]})
    clock.advance("2026-01-01T00:01:00Z")
    _tick_call(engine, now, 1)
    await asyncio.sleep(0.05)
    assert engine.state()["triggers"][0]["waiting"] is True
    clock.advance("2026-01-01T06:01:00Z")
    for _ in range(100):
        if engine.state()["triggers"][0]["status"] == "fired":
            break
        await asyncio.sleep(0.05)
    assert engine.state()["triggers"][0]["status"] == "fired"


def test_the_card_offers_advance():
    clock_ext = next(e for e in GATEWAY_EXTENSIONS if e["uri"] == EXT_CLOCK_URI)
    assert clock_ext["params"]["methods"]["advance"] == {
        "method": "PUT", "endpoint": "/clock/advance",
        "request": {"type": "object", "properties": {"virtual_time": {"type": "string"}}, "required": ["virtual_time"]}}


@pytest.fixture
def gateway(tmp_path, monkeypatch) -> Gateway:
    monkeypatch.setattr(gw_mod, "GATEWAY_TRAJECTORY_FILE", str(tmp_path / "trajectory.jsonl"))
    gw = Gateway(host="127.0.0.1", port=0, server_name="AgentEnvGateway", internal_mcp_servers=[])
    gw._clock = Clock(now_fn=FakeNow())
    yield gw
    gw._close_trajectory_file()


def _put(gw: Gateway, path: str, body) -> httpx.Response:
    async def _do() -> httpx.Response:
        transport = httpx.ASGITransport(app=gw._mcp.streamable_http_app())
        async with httpx.AsyncClient(transport=transport, base_url="http://gateway") as client:
            return await client.put(path, content=body if isinstance(body, str) else None,
                                    json=None if isinstance(body, str) else body)

    return asyncio.run(_do())


def test_route_advances_an_armed_clock(gateway):
    gateway._clock.set_time(T0, rate=0)
    resp = _put(gateway, "/clock/advance", {"virtual_time": "2026-01-04T08:00:00Z"})
    assert resp.status_code == 200
    assert resp.json()["virtual_time"] == "2026-01-04T08:00:00Z"


@pytest.mark.parametrize("body,fragment", [
    ({"virtual_time": "2026-01-04T08:00:00Z"}, "not armed"),
    ("{not json", "Invalid JSON body"),
    ("null", "must be a JSON object"),
    ("[]", "must be a JSON object"),
    ('"2026-01-04T08:00:00Z"', "must be a JSON object"),
])
def test_route_refuses_with_400(gateway, body, fragment):
    resp = _put(gateway, "/clock/advance", body)
    assert resp.status_code == 400 and fragment in resp.json()["error"]


def test_route_refuses_a_backwards_advance(gateway):
    gateway._clock.set_time("2026-01-04T08:00:00Z", rate=0)
    resp = _put(gateway, "/clock/advance", {"virtual_time": T0})
    assert resp.status_code == 400 and "only moves forward" in resp.json()["error"]
