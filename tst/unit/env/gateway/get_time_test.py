"""Unit tests for the agent-facing get_time tool (urn:agentenv:clock/v1)."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from mcp.types import Tool as MCPTool

from agent_env.env.gateway import InternalMCPServer
from agent_env.env.gateway.clock import ClockError
from agent_env.env.gateway.get_time import GET_TIME_TOOL_DESCRIPTION, GET_TIME_TOOL_NAME
from agent_env.env.gateway.gateway import Gateway

T = "2019-06-15T12:00:00Z"


class _FakeSession:
    """Minimal stand-in for an MCP ClientSession — _discover_and_register_tools only calls
    list_tools() on it, so a real session (and a live server) isn't needed to exercise the path."""

    def __init__(self, tool_names: list[str]):
        self._tools = [MCPTool(name=n, description="", inputSchema={"type": "object", "properties": {}})
                       for n in tool_names]

    async def list_tools(self):
        return SimpleNamespace(tools=self._tools)


@pytest.fixture
def gw() -> Gateway:
    return Gateway(host="127.0.0.1", port=0, server_name="unit", internal_mcp_servers=[])


def _tools(gw: Gateway) -> dict:
    return gw._mcp._tool_manager._tools


def _call(gw: Gateway) -> dict:
    return asyncio.run(_tools(gw)[GET_TIME_TOOL_NAME].fn())


def _discovered(gw: Gateway) -> Gateway:
    """Simulate a gateway that completed tool discovery: the flag is what marks the pass done,
    and a backing server's entry is what a pass over a non-empty server list leaves behind."""
    gw._server_tools["mcp-svc"] = [MCPTool(name="svc_thing", description="",
                                           inputSchema={"type": "object", "properties": {}})]
    gw._tools_discovered = True
    return gw


def test_absent_until_armed_and_removed_on_clear(gw):
    _discovered(gw)
    assert GET_TIME_TOOL_NAME not in _tools(gw)
    assert "gateway" not in gw._server_tools

    gw._clock.set_time(T, 1)
    gw._register_get_time_tool()
    assert GET_TIME_TOOL_NAME in _tools(gw)
    assert [t.name for t in gw._server_tools["gateway"]] == [GET_TIME_TOOL_NAME]

    gw._clock.clear()
    gw._unregister_get_time_tool()
    assert GET_TIME_TOOL_NAME not in _tools(gw)
    # the unarmed shape round-trips exactly, key included
    assert "gateway" not in gw._server_tools
    assert "mcp-svc" in gw._server_tools


def test_does_not_populate_server_tools_when_discovery_has_not_run(gw):
    """Mirroring before a successful discovery pass would publish a descriptor into a listing that
    is about to be rebuilt, so the mirror waits for the flag. The tool is still live over MCP."""
    assert gw._tools_discovered is False
    gw._clock.set_time(T, 1)
    gw._register_get_time_tool()

    assert gw._server_tools == {}, "nothing to mirror into until discovery has run"
    assert GET_TIME_TOOL_NAME in _tools(gw), "the agent still sees it over MCP"


def test_gateway_with_no_backing_servers_still_mirrors_the_clock(gw):
    """A clock-only gateway discovers successfully and leaves _server_tools empty, so gating the
    mirror on emptiness would skip it forever."""
    assert gw.internal_mcp_servers == []
    gw._clock.set_time(T, 1)
    gw._register_get_time_tool()

    asyncio.run(gw._discover_and_register_tools({}))

    assert [t.name for t in gw._server_tools["gateway"]] == [GET_TIME_TOOL_NAME]
    assert gw._trigger_engine._is_readonly(GET_TIME_TOOL_NAME) is True


def test_a_completed_pass_over_zero_servers_is_not_rediscovered(gw):
    """The same emptiness overload on the read side: _ensure_tools_discovered must short-circuit
    on the flag, or a clock-only gateway re-opens sessions and re-discovers on every /step call."""
    asyncio.run(gw._discover_and_register_tools({}))
    assert gw._server_tools == {} and gw._tools_discovered is True

    def _explode(*a, **kw):
        raise AssertionError("discovery must not run again after a successful pass")

    gw._open_persistent_sessions = _explode
    asyncio.run(gw._ensure_tools_discovered())


def test_arm_during_failed_discovery_is_recovered_by_the_next_discovery(gw):
    """Counterpart to the test above: discovery only writes backing-server entries, so without a
    re-mirror get_time stays callable over MCP yet invisible to /step and /state."""
    gw._clock.set_time(T, 1)
    gw._register_get_time_tool()
    assert gw._server_tools == {}

    _discovered(gw)  # the lazy retry finally succeeds, writing backing-server entries only
    gw._mirror_get_time_descriptor()

    assert [t.name for t in gw._server_tools["gateway"]] == [GET_TIME_TOOL_NAME]
    assert "mcp-svc" in gw._server_tools
    # readOnlyHint has to survive the recovery, or every clock read re-evaluates state triggers
    assert gw._server_tools["gateway"][0].annotations.readOnlyHint is True


def test_real_discovery_pass_restores_the_descriptor():
    """The production wiring, not just the helper: a clock armed while discovery was failed/pending
    must come back into /step list_tools and /state once _discover_and_register_tools succeeds."""
    server = InternalMCPServer(name="mcp-svc", mcp_url="http://svc/mcp")
    gw = Gateway(host="127.0.0.1", port=0, server_name="unit", internal_mcp_servers=[server])

    gw._clock.set_time(T, 1)
    gw._register_get_time_tool()
    assert gw._server_tools == {}, "the failed-discovery sentinel must survive the arm"

    asyncio.run(gw._discover_and_register_tools({server.mcp_url: _FakeSession(["svc_thing"])}))

    assert [t.name for t in gw._server_tools["gateway"]] == [GET_TIME_TOOL_NAME]
    assert [t.name for t in gw._server_tools["mcp-svc"]] == ["svc_thing"]
    assert gw._trigger_engine._is_readonly(GET_TIME_TOOL_NAME) is True


def test_remirror_is_inert_when_the_clock_was_never_armed(gw):
    """Discovery calls the re-mirror unconditionally, so an unarmed gateway must come out of it
    with the exact same tool inventory — no empty "gateway" key, no phantom descriptor."""
    _discovered(gw)
    assert gw._mirror_get_time_descriptor() is False
    assert "gateway" not in gw._server_tools


def test_remirror_does_not_duplicate_an_already_mirrored_descriptor(gw):
    _discovered(gw)
    gw._clock.set_time(T, 1)
    gw._register_get_time_tool()
    assert gw._mirror_get_time_descriptor() is False
    assert [t.name for t in gw._server_tools["gateway"]] == [GET_TIME_TOOL_NAME]


def test_registration_is_idempotent(gw):
    _discovered(gw)
    gw._clock.set_time(T, 1)
    gw._register_get_time_tool()
    gw._register_get_time_tool()
    assert len(gw._server_tools["gateway"]) == 1
    gw._unregister_get_time_tool()
    gw._unregister_get_time_tool()
    assert GET_TIME_TOOL_NAME not in _tools(gw)


def test_advertised_contract(gw):
    _discovered(gw)
    gw._clock.set_time(T, 1)
    gw._register_get_time_tool()
    tool = _tools(gw)[GET_TIME_TOOL_NAME]
    assert tool.description == GET_TIME_TOOL_DESCRIPTION
    assert tool.parameters["properties"] == {}
    # readOnlyHint keeps a clock read from re-evaluating every armed state trigger
    assert tool.annotations.readOnlyHint is True
    assert gw._trigger_engine._is_readonly(GET_TIME_TOOL_NAME) is True


def test_registration_drops_a_readonly_memo_built_before_the_clock_was_armed(gw):
    """The memo is derived from _server_tools, so mirroring into it has to invalidate — the other
    contract assertions build the memo fresh after arming and would pass without it."""
    _discovered(gw)
    assert gw._trigger_engine._is_readonly("svc_thing") is False
    gw._clock.set_time(T, 1)
    gw._register_get_time_tool()
    assert gw._trigger_engine._is_readonly(GET_TIME_TOOL_NAME) is True


def test_registering_beside_list_website_urls_does_not_clobber_it(gw):
    gw._tools_discovered = True
    gw._server_tools["gateway"] = [MCPTool(name="list_website_urls", description="",
                                           inputSchema={"type": "object", "properties": {}})]
    gw._clock.set_time(T, 1)
    gw._register_get_time_tool()
    assert [t.name for t in gw._server_tools["gateway"]] == ["list_website_urls", GET_TIME_TOOL_NAME]
    gw._unregister_get_time_tool()
    assert [t.name for t in gw._server_tools["gateway"]] == ["list_website_urls"]


@pytest.mark.parametrize("rate", [0, 1, 3600])
def test_answers_on_the_armed_clock_at_second_precision(gw, rate):
    gw._clock.set_time(T, rate)
    gw._register_get_time_tool()
    payload = _call(gw)
    assert set(payload) == {"current_time"}
    assert payload["current_time"].startswith("2019-06-15T12:")
    # every synthetic server formats to second precision; a bare isoformat leaks microseconds at rate>0
    assert "." not in payload["current_time"]


def test_unarmed_clock_fails_loud_without_disclosing_why(gw):
    """Unreachable in normal operation, but a wall-time answer would look real. The message reaches
    the agent as MCP error content, so it must not repeat Clock.read()'s "not armed" tell."""
    gw._register_get_time_tool()
    with pytest.raises(ClockError) as excinfo:
        _call(gw)
    assert "armed" not in str(excinfo.value)


def test_backing_server_name_collision_refuses_to_register(gw):
    """A backing `get_time` already sits in the tool manager, so the guard has to run before the
    idempotence check — otherwise arming would silently expose the wrong tool."""
    gw._tool_server_urls[GET_TIME_TOOL_NAME] = "http://backing"
    gw._create_and_register_proxy_tool("http://backing", MCPTool(
        name=GET_TIME_TOOL_NAME, description="backing", inputSchema={"type": "object", "properties": {}}))

    with pytest.raises(ValueError, match="already registered by a backing server"):
        gw._register_get_time_tool()
    assert GET_TIME_TOOL_NAME in _tools(gw)


def test_clear_never_deletes_a_backing_servers_tool(gw):
    """Removal must leave a colliding backing tool alone rather than deleting it — and must not
    raise either, or /clock/clear would 500 on an env it never registered anything in."""
    gw._tool_server_urls[GET_TIME_TOOL_NAME] = "http://backing"
    gw._create_and_register_proxy_tool("http://backing", MCPTool(
        name=GET_TIME_TOOL_NAME, description="backing", inputSchema={"type": "object", "properties": {}}))

    gw._unregister_get_time_tool()
    assert GET_TIME_TOOL_NAME in _tools(gw)
    assert _tools(gw)[GET_TIME_TOOL_NAME].description == "backing"


def _trajectory(gw: Gateway) -> list[dict]:
    from agent_env.env.gateway.gateway import GATEWAY_TRAJECTORY_FILE
    gw._trajectory_file.flush()
    return [json.loads(line) for line in open(GATEWAY_TRAJECTORY_FILE)]


def test_mcp_call_lands_in_the_trajectory_exactly_once(gw):
    """A gateway-native tool has no proxy closure to log for it, and /step bypasses the MCP wrapper
    entirely — so without logging in filtered_call_tool an agent's clock reads leave no record."""
    _discovered(gw)
    gw._clock.set_time(T, 0)
    gw._register_get_time_tool()
    before = len(_trajectory(gw))

    asyncio.run(gw._mcp._tool_manager.call_tool(GET_TIME_TOOL_NAME, {}))

    new = _trajectory(gw)[before:]
    calls = [e for e in new if e["event_type"] == "tool_call"]
    results = [e for e in new if e["event_type"] == "tool_call_result"]
    assert len(calls) == 1 and len(results) == 1, new
    assert calls[0]["tool_call"]["function_name"] == GET_TIME_TOOL_NAME
    assert results[0]["tool_call_event_id"] == calls[0]["event_id"]
    # Every event carries the gateway clock while it is armed
    assert calls[0]["virtual_time"] == T, calls[0]


def test_trajectory_logging_never_breaks_the_call(gw, monkeypatch):
    """Observability is fail-open: a broken trajectory write must not take the tool down with it."""
    _discovered(gw)
    gw._clock.set_time(T, 0)
    gw._register_get_time_tool()

    async def boom(_event):
        raise OSError("disk full")

    monkeypatch.setattr(gw, "_log_event", boom)
    assert asyncio.run(gw._mcp._tool_manager.call_tool(GET_TIME_TOOL_NAME, {})) is not None


def test_backing_tools_are_not_logged_by_the_wrapper(gw):
    """Proxied tools already log inside their own closure, so the wrapper must skip them or every
    backing tool call would be counted twice."""
    from mcp.types import Tool as _MCPTool

    gw._create_and_register_proxy_tool("http://backing", _MCPTool(
        name="svc_thing", description="", inputSchema={"type": "object", "properties": {}}))
    gw._tool_server_urls["svc_thing"] = "http://backing"
    before = len(_trajectory(gw))
    try:
        asyncio.run(gw._mcp._tool_manager.call_tool("svc_thing", {}))
    except Exception:
        pass  # no live session behind it; we only care what the wrapper logged
    logged = [e for e in _trajectory(gw)[before:]
              if e.get("tool_call", {}).get("function_name") == "svc_thing"]
    # exactly one — the proxy closure's own. Two would mean the wrapper logged it as well.
    assert len(logged) == 1, f"expected the proxy closure's single record, got {logged}"
