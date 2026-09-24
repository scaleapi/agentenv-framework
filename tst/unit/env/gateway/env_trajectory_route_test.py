"""GET /trajectory route tests: the real Gateway's Starlette app driven
through httpx's ASGITransport — no sockets, no lifespan needed for custom routes."""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from agent_env.env.gateway import gateway as gw_mod
from agent_env.env.gateway.clock import Clock
from agent_env.env.gateway.constants import EXT_TRAJECTORY_URI
from agent_env.env.gateway.gateway import Gateway

T0 = "2026-01-01T00:00:00Z"
RATE = 86400


class FakeNow:
    def __init__(self, start: float = 1000.0):
        self.t = start

    def __call__(self) -> float:
        return self.t

    def tick(self, seconds: float) -> None:
        self.t += seconds


@pytest.fixture
def gateway(tmp_path, monkeypatch) -> Gateway:
    """A Gateway logging to tmp_path (module attr patched before construction)."""
    monkeypatch.setattr(gw_mod, "GATEWAY_TRAJECTORY_FILE", str(tmp_path / "trajectory.jsonl"))
    gw = Gateway(
        host="127.0.0.1",
        port=0,
        server_name="AgentEnvGateway",
        internal_mcp_servers=[],
    )
    gw._clock = Clock(now_fn=FakeNow())
    yield gw
    gw._close_trajectory_file()


def _get(gw: Gateway, path: str) -> httpx.Response:
    async def _do() -> httpx.Response:
        transport = httpx.ASGITransport(app=gw._mcp.streamable_http_app())
        async with httpx.AsyncClient(transport=transport, base_url="http://gateway") as client:
            return await client.get(path)

    return asyncio.run(_do())


def _log(gw: Gateway, n: int, event_type: str = "tool_call") -> None:
    async def _do() -> None:
        for _ in range(n):
            await gw._log_event({"event_type": event_type})

    asyncio.run(_do())


def test_empty_trajectory_streams_an_empty_body(gateway):
    response = _get(gateway, "/trajectory")

    assert response.status_code == 200
    assert response.content == b""
    assert response.headers["content-type"].startswith("application/jsonl")
    assert response.headers["X-Trajectory-Total-Bytes"] == "0"


def test_events_round_trip_as_jsonl(gateway):
    _log(gateway, 3)

    response = _get(gateway, "/trajectory")

    events = [json.loads(line) for line in response.text.splitlines()]
    assert [e["event_id"] for e in events] == ["event_1", "event_2", "event_3"]
    assert all(e["event_type"] == "tool_call" for e in events)
    assert int(response.headers["X-Trajectory-Total-Bytes"]) == len(response.content)


def test_query_params_are_ignored(gateway):
    """The route has no parameters — the full trajectory always streams."""
    _log(gateway, 2)

    response = _get(gateway, "/trajectory?tail_bytes=1")

    events = [json.loads(line) for line in response.text.splitlines()]
    assert [e["event_id"] for e in events] == ["event_1", "event_2"]


def test_virtual_time_flows_through_the_route(gateway):
    """The slice-1 stamp is visible to route consumers when the clock is armed."""
    gateway._clock.set_time(T0, RATE)
    _log(gateway, 1, event_type="trigger_fired")

    response = _get(gateway, "/trajectory")

    (event,) = [json.loads(line) for line in response.text.splitlines()]
    assert event["virtual_time"] == T0


def test_env_card_advertises_the_extension(gateway):
    response = _get(gateway, "/.well-known/agent-env.json")

    card = response.json()
    by_uri = {e["uri"]: e for e in card["capabilities"]["extensions"]}
    ext = by_uri[EXT_TRAJECTORY_URI]
    assert ext["params"]["endpoint"] == "/trajectory"
    assert ext["params"]["methods"]["get"]["method"] == "GET"
