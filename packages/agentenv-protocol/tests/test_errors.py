"""Structured errors: an AgentEnvError raised in an extension answers with its status and fields, raised in a tool
it answers isError with the fields as structured content on both FastMCPs, and the client parses them back."""
from __future__ import annotations

from unittest.mock import patch

import anyio
import httpx
import pytest

pytest.importorskip("mcp.server.fastmcp")
fastmcp = pytest.importorskip("fastmcp")

from mcp.shared.memory import create_connected_server_and_client_session  # noqa: E402

from agentenv_protocol import (  # noqa: E402
    AgentEnvEnvironment,
    AgentEnvError,
    ExtensionCallError,
    ToolContext,
    client,
    environment_card,
    extension,
    tool,
)

_URI = "urn:agentenv:frames/v1"


@environment_card(name="frames")
class _Env(AgentEnvEnvironment):
    def __init__(self) -> None:
        self.refuse_hook = False

    def on_tool_call(self, context: ToolContext):
        if self.refuse_hook:
            raise AgentEnvError("device_unavailable", "not ready", retryable=True, status=503)
        return None

    @tool()
    def click(self, x: int) -> str:
        if x > 100:
            raise AgentEnvError("out_of_bounds", f"({x}, 0) is outside frame f1 (100×100)", hint="x < 100")
        return "ok"

    @tool()
    def crash(self) -> str:
        raise RuntimeError("kaboom")

    @extension(_URI)
    async def stale(self, frame: str) -> dict:
        raise AgentEnvError("stale_frame", f"frame {frame} is stale", retryable=True, status=409)

    @extension("urn:agentenv:partial/v1")
    async def partial(self) -> dict:
        raise AgentEnvError("delivery_failed", "typed 3 of 5 characters", applied="unknown", status=502)

    @extension("urn:agentenv:broken/v1")
    async def broken(self) -> dict:
        raise RuntimeError("kaboom")


def test_fields_carry_the_defaults_and_a_hint_only_when_given():
    assert AgentEnvError("invalid_argument", "bad").fields() == {
        "code": "invalid_argument", "message": "bad", "retryable": False, "applied": "no"}
    error = AgentEnvError("invalid_argument", "unknown key", hint="did you mean Return?")
    assert error.fields()["hint"] == "did you mean Return?" and error.status == 400 and str(error) == "unknown key"


def _http(env: _Env) -> httpx.AsyncClient:
    app = env.create_app().streamable_http_app()
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


@pytest.mark.asyncio
async def test_an_extension_answers_with_the_errors_status_and_fields():
    async with _http(_Env()) as http:
        stale = await http.post("/agentenv/ext/stale", json={"frame": "f3"})
        partial = await http.post("/agentenv/ext/partial", json={})
    assert stale.status_code == 409
    assert stale.json() == {"ok": False, "error": {
        "code": "stale_frame", "message": "frame f3 is stale", "retryable": True, "applied": "no"}}
    assert partial.status_code == 502 and partial.json()["error"]["applied"] == "unknown"


@pytest.mark.asyncio
async def test_any_other_extension_failure_is_extension_failed_with_applied_unknown():
    async with _http(_Env()) as http:
        r = await http.post("/agentenv/ext/broken", json={})
    assert r.status_code == 500
    assert r.json()["error"] == {"code": "extension_failed", "message": "kaboom", "retryable": False, "applied": "unknown"}


@pytest.mark.asyncio
async def test_invalid_params_never_ran_so_applied_is_no():
    async with _http(_Env()) as http:
        r = await http.post("/agentenv/ext/stale", json={"frame": "f1", "extra": 1})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_params" and r.json()["error"]["applied"] == "no"


@pytest.mark.asyncio
async def test_a_tool_error_is_an_error_result_with_structured_fields_on_mcps_fastmcp():
    env = _Env()
    with anyio.fail_after(30):
        async with create_connected_server_and_client_session(env.create_app()) as session:
            refused = await session.call_tool("click", {"x": 140})
            served = await session.call_tool("click", {"x": 4})
            crashed = await session.call_tool("crash", {})
            env.refuse_hook = True
            hooked = await session.call_tool("click", {"x": 4})
    assert refused.isError is True
    assert refused.content[0].text == "out_of_bounds: (140, 0) is outside frame f1 (100×100)"
    assert refused.structuredContent == {"code": "out_of_bounds", "message": "(140, 0) is outside frame f1 (100×100)",
                                         "retryable": False, "applied": "no", "hint": "x < 100"}
    assert served.isError is False and served.content[0].text == "ok"
    assert crashed.isError is True and "kaboom" in crashed.content[0].text and crashed.structuredContent is None
    assert hooked.isError is True and hooked.structuredContent["code"] == "device_unavailable"
    assert hooked.structuredContent["retryable"] is True


@pytest.mark.asyncio
async def test_a_tool_error_is_an_error_result_with_structured_fields_on_fastmcp():
    app = fastmcp.FastMCP("frames")
    env = _Env()
    env.mount(app)
    with anyio.fail_after(30):
        async with fastmcp.Client(app) as session:
            refused = await session.call_tool("click", {"x": 140}, raise_on_error=False)
            crashed = await session.call_tool("crash", {}, raise_on_error=False)
            env.refuse_hook = True
            hooked = await session.call_tool("click", {"x": 4}, raise_on_error=False)
    assert refused.is_error is True
    assert refused.content[0].text == "out_of_bounds: (140, 0) is outside frame f1 (100×100)"
    assert refused.structured_content["code"] == "out_of_bounds" and refused.structured_content["hint"] == "x < 100"
    assert crashed.is_error is True and "kaboom" in crashed.content[0].text and not crashed.structured_content
    assert hooked.is_error is True and hooked.structured_content["code"] == "device_unavailable"


@pytest.mark.asyncio
async def test_the_client_raises_the_servers_fields_as_an_extension_call_error():
    env = _Env()
    app = env.create_app().streamable_http_app()
    card = await _card(app)
    asgi = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")
    with patch("agentenv_protocol.client.httpx.AsyncClient", return_value=asgi):
        with pytest.raises(ExtensionCallError) as raised:
            await client.invoke_extension("http://t", card, _URI, {"frame": "f9"})
    error = raised.value
    assert isinstance(error, httpx.HTTPStatusError) and error.response.status_code == 409
    assert (error.code, error.detail, error.retryable, error.applied, error.hint) == (
        "stale_frame", "frame f9 is stale", True, "no", None)
    assert str(error) == "409 stale_frame: frame f9 is stale"


async def _card(app) -> dict:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as http:
        return (await http.get("/.well-known/agent-env.json")).json()
