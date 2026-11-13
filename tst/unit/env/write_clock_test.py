"""``DeployedEnv.write_clock``, against a gateway in this process: each write waits for the computers that follow the
clock, one they didn't show raises ``ClockFollowersBehind`` naming them with the write applied, a clock no computer
follows answers at once, and a caller's own wait stands."""

from __future__ import annotations

import json
import time

import httpx
import pytest
from agentenv_protocol.client import ExtensionCallError

from agent_env.env import env as env_mod
from agent_env.env.env import CLOCK_WAIT_FOLLOWERS_S, ClockFollowersBehind, DeployedGatewayEnv
from agent_env.env.gateway import gateway as gw_mod
from agent_env.env.gateway.constants import EXT_CLOCK_URI, GATEWAY_EXTENSIONS, WELL_KNOWN_PATH
from agent_env.env.gateway.gateway import Gateway

GATEWAY = "http://gateway"
T0 = "2026-01-01T00:00:00Z"
LATER = "2026-01-02T00:00:00Z"
#: Short enough to keep the tests quick, long enough that answering "at once" can't be a wait running out.
WAIT_S = 0.5


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A gateway whose clock ``laptop`` follows (``followers``), reached through a deployed env record; every request
    body the record sends is kept in ``sent``."""
    monkeypatch.setattr(gw_mod, "GATEWAY_TRAJECTORY_FILE", str(tmp_path / "trajectory.jsonl"))
    monkeypatch.setattr(env_mod, "CLOCK_WAIT_FOLLOWERS_S", WAIT_S)
    made, sent = [], []
    real = httpx.AsyncClient

    def make(followers: dict | None = None) -> DeployedGatewayEnv:
        gw = Gateway(host="127.0.0.1", port=0, server_name="unit", internal_mcp_servers=[], clock_followers=followers)
        made.append(gw)
        app = httpx.ASGITransport(app=gw._asgi_app())

        async def handle(request: httpx.Request) -> httpx.Response:
            body = await request.aread()
            sent.append((request.url.path, json.loads(body) if body else None))
            return await app.handle_async_request(request)

        monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: real(*a, **{**k, "transport": httpx.MockTransport(handle)}))
        card = {"capabilities": {"extensions": list(GATEWAY_EXTENSIONS)}}
        return DeployedGatewayEnv(env_id="office", env_version=1, gateway_url=GATEWAY, mcp_url="", sandbox_id="sb",
                                  environment_card_url=f"{GATEWAY}{WELL_KNOWN_PATH}", environment_card=card)

    yield make, sent
    for gw in made:
        gw._close_trajectory_file()


@pytest.mark.asyncio
async def test_a_clock_no_computer_follows_answers_at_once_and_each_write_asked_to_wait(world):
    make, sent = world
    office = make()

    started = time.monotonic()
    await office.write_clock("set_time", {"virtual_time": T0, "virtual_seconds_per_real_second": 0})
    await office.write_clock("advance", {"virtual_time": LATER})
    await office.write_clock("set_rate", {"virtual_seconds_per_real_second": 1})
    took = time.monotonic() - started

    assert took < WAIT_S, took
    assert [(path, body["wait_followers_s"]) for path, body in sent] == [
        ("/clock/set-time", WAIT_S), ("/clock/advance", WAIT_S), ("/clock/set-rate", WAIT_S)]
    assert CLOCK_WAIT_FOLLOWERS_S == 2.0


@pytest.mark.asyncio
async def test_a_write_a_computer_doesnt_show_raises_naming_it_and_the_write_stands(world):
    make, _ = world
    office = make({"laptop": "token-laptop"})

    started = time.monotonic()
    with pytest.raises(ClockFollowersBehind) as behind:
        await office.write_clock("set_time", {"virtual_time": T0, "virtual_seconds_per_real_second": 0})
    waited = time.monotonic() - started
    state = await office.invoke(EXT_CLOCK_URI, "state")

    assert (behind.value.env_id, behind.value.method, behind.value.stale) == ("office", "set_time", ["laptop"])
    assert "env 'office': clock set_time: " in str(behind.value) and "laptop didn't show it" in str(behind.value)
    assert waited >= WAIT_S
    assert state["armed"] and state["virtual_time"] == T0


@pytest.mark.asyncio
async def test_a_callers_own_wait_stands(world):
    make, sent = world
    office = make({"laptop": "token-laptop"})

    await office.write_clock("set_time", {"virtual_time": T0, "virtual_seconds_per_real_second": 0, "wait_followers_s": 0})

    assert sent[0][1]["wait_followers_s"] == 0


@pytest.mark.asyncio
async def test_a_write_the_clock_refuses_is_the_gateways_error_and_a_read_isnt_a_write(world):
    make, _ = world
    office = make({"laptop": "token-laptop"})

    with pytest.raises(ExtensionCallError) as refused:
        await office.write_clock("advance", {"virtual_time": LATER})
    with pytest.raises(ValueError, match="'state' isn't a clock write"):
        await office.write_clock("state")

    assert refused.value.response.status_code == 400 and "not armed" in refused.value.response.text
