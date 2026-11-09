"""The gateway's control routes take the deploy's control token when it has one; every other route stays open."""
from __future__ import annotations

import httpx
import pytest
from starlette.responses import JSONResponse

from agent_env.env.gateway.constants import CONTROL_ROUTES
from agent_env.env.gateway.gateway import Gateway

TOKEN = "s3cret"
CONTROL = {
    "/tools/disable": "POST", "/tools/enable": "POST",
    "/triggers/register": "POST", "/triggers/remove": "POST", "/triggers/clear": "POST",
    "/notifications/config": "POST", "/notifications/take": "POST", "/notifications/clear": "POST",
}
OPEN = {"/state": "GET", "/step": "POST", "/clock/time": "GET", "/clock/state": "GET", "/triggers/state": "GET",
        "/notifications/state": "GET", "/trajectory": "GET", "/changes": "GET", "/.well-known/agent-env.json": "GET",
        "/clock/set-time": "PUT", "/clock/advance": "PUT", "/clock/clear": "POST"}


def _client(token: str | None) -> httpx.AsyncClient:
    gw = Gateway(host="127.0.0.1", port=0, server_name="unit", internal_mcp_servers=[], control_token=token)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=gw._asgi_app()), base_url="http://gateway")


def test_the_routes_under_test_are_the_ones_the_gateway_covers():
    assert set(CONTROL) == CONTROL_ROUTES


@pytest.mark.asyncio
@pytest.mark.parametrize("path", sorted(CONTROL))
@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}, {"Authorization": TOKEN}],
                         ids=["none", "wrong", "not-bearer"])
async def test_a_control_route_refuses_a_call_without_the_token(path, headers):
    async with _client(TOKEN) as client:
        before = (await client.get("/triggers/state")).json(), (await client.get("/notifications/state")).json()
        response = await client.request(CONTROL[path], path, json={}, headers=headers)
        after = (await client.get("/triggers/state")).json(), (await client.get("/notifications/state")).json()
    assert response.status_code == 401
    assert response.json() == {"ok": False, "error": "this route takes the env's control token"}
    assert after == before


@pytest.mark.asyncio
@pytest.mark.parametrize("path", sorted(CONTROL))
async def test_a_control_route_answers_a_call_with_the_token(path):
    async with _client(TOKEN) as client:
        response = await client.request(CONTROL[path], path, json={}, headers={"Authorization": f"Bearer {TOKEN}"})
    assert response.status_code != 401


@pytest.mark.asyncio
async def test_the_clock_is_armed_and_read_without_the_token():
    async with _client(TOKEN) as client:
        armed = await client.put("/clock/set-time", json={"virtual_time": "2019-06-15T12:00:00Z"})
        read = await client.get("/clock/time")
    assert (armed.status_code, read.status_code) == (200, 200)
    assert read.json()["virtual_time"].startswith("2019-06-15T12:00")


@pytest.mark.asyncio
@pytest.mark.parametrize("path", sorted(OPEN))
async def test_every_other_route_stays_open(path):
    async with _client(TOKEN) as client:
        response = await client.request(OPEN[path], path, json={} if OPEN[path] == "POST" else None)
    assert response.status_code != 401


@pytest.mark.asyncio
@pytest.mark.parametrize("value, seen", [(f"Bearer {TOKEN}", None), ("Bearer a-childs-own", "Bearer a-childs-own")],
                         ids=["the-gateways", "another"])
async def test_no_route_behind_the_gateway_reads_its_token(value, seen):
    gw = Gateway(host="127.0.0.1", port=0, server_name="unit", internal_mcp_servers=[], control_token=TOKEN)

    @gw._mcp.custom_route("/echo", methods=["GET"])
    async def echo(request):
        return JSONResponse({"authorization": request.headers.get("authorization")})

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gw._asgi_app()), base_url="http://gateway") as client:
        response = await client.get("/echo", headers={"Authorization": value})
    assert response.json() == {"authorization": seen}


@pytest.mark.asyncio
@pytest.mark.parametrize("path", sorted(CONTROL))
async def test_a_gateway_started_without_a_token_checks_none(path):
    async with _client(None) as client:
        response = await client.request(CONTROL[path], path, json={})
    assert response.status_code != 401
