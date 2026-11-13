"""Unit tests for the user chat (urn:agentenv:user-chat/v1): message_user and stay_quiet, built in for the roles a
config names."""
from __future__ import annotations

import asyncio

import httpx
import pytest
from mcp.types import Tool as MCPTool

from agent_env.env.gateway import gateway as gw_mod
from agent_env.env.gateway.clock import Clock
from agent_env.env.gateway.constants import AGENT_ENV_ROLE_HEADER, EXT_USER_CHAT_URI, GATEWAY_EXTENSIONS
from agent_env.env.gateway.gateway import Gateway

T0 = "2027-02-01T15:00:00Z"


@pytest.fixture
def gateway(tmp_path, monkeypatch):
    monkeypatch.setattr(gw_mod, "GATEWAY_TRAJECTORY_FILE", str(tmp_path / "trajectory.jsonl"))
    gw = Gateway(host="127.0.0.1", port=0, server_name="AgentEnvGateway", internal_mcp_servers=[])
    clock = Clock()
    clock.set_time(T0, 0)
    gw._clock = clock
    gw._server_tools["mcp-svc"] = [MCPTool(name="svc_thing", description="", inputSchema={"type": "object"})]
    gw._tools_discovered = True
    yield gw
    gw._close_trajectory_file()


def _req(gw, method, path, body=None, role=None):
    async def _do():
        transport = httpx.ASGITransport(app=gw._mcp.streamable_http_app())
        async with httpx.AsyncClient(transport=transport, base_url="http://gateway") as client:
            return await client.request(method, path, json=body,
                                        headers={AGENT_ENV_ROLE_HEADER: role} if role else None)
    return asyncio.run(_do())


def _names(gw, role):
    return {t["name"] for t in _req(gw, "POST", "/step", {"action": "list_tools"}, role=role).json()["tools"]}


def _call(gw, tool, arguments, role):
    return _req(gw, "POST", "/step", {"action": "call_tool", "tool_name": tool, "arguments": arguments}, role=role)


def test_the_card_offers_config_state_take_and_clear():
    [ext] = [e for e in GATEWAY_EXTENSIONS if e["uri"] == EXT_USER_CHAT_URI]
    assert set(ext["params"]["methods"]) == {"config", "state", "take", "clear"}
    assert "stay_quiet" in ext["params"]["methods"]["config"]["request"]["properties"]


def test_message_user_reaches_a_configured_role_through_a_deny_all(gateway):
    _req(gateway, "POST", "/tools/disable", {"role": "psi_agent", "tools": ["*"]})
    assert "message_user" not in _names(gateway, "psi_agent")
    assert _req(gateway, "POST", "/user_chat/config", {"roles": ["psi_agent"], "user_name": "Tess"}).status_code == 200

    names = _names(gateway, "psi_agent")
    assert "message_user" in names and "svc_thing" not in names
    assert "message_user" not in _names(gateway, "default")
    assert _call(gateway, "message_user", {"text": "hi Tess, two questions"}, "psi_agent").status_code == 200
    assert _call(gateway, "message_user", {"text": "x"}, "default").status_code == 403
    taken = _req(gateway, "POST", "/user_chat/take", {"take_id": "a"}).json()["items"]
    assert [(m["from"], m["at"], m["text"]) for m in taken] == [("psi_agent", T0, "hi Tess, two questions")]
    assert _req(gateway, "POST", "/user_chat/take", {"take_id": "a"}).json()["items"] == taken
    assert _req(gateway, "POST", "/user_chat/take", {}).json()["items"] == []


def test_stay_quiet_is_off_unless_the_config_turns_it_on(gateway):
    _req(gateway, "POST", "/user_chat/config", {"roles": ["psi_agent"], "user_name": "Tess"})
    assert "message_user" in _names(gateway, "psi_agent") and "stay_quiet" not in _names(gateway, "psi_agent")
    called = _call(gateway, "stay_quiet", {}, "psi_agent")
    assert called.status_code != 200 or called.json().get("isError")
    assert _req(gateway, "POST", "/user_chat/take", {"take_id": "a"}).json() == {"items": []}


def test_stay_quiet_sends_nothing_and_is_recorded_for_the_next_take(gateway):
    _req(gateway, "POST", "/tools/disable", {"role": "psi_agent", "tools": ["*"]})
    st = _req(gateway, "POST", "/user_chat/config", {"roles": ["psi_agent"], "user_name": "Tess", "stay_quiet": True})
    assert st.status_code == 200 and st.json()["stay_quiet"] is True
    assert {"message_user", "stay_quiet"} <= _names(gateway, "psi_agent")
    assert "stay_quiet" not in _names(gateway, "default")

    assert _call(gateway, "stay_quiet", {}, "psi_agent").status_code == 200
    assert _call(gateway, "stay_quiet", {}, "default").status_code == 403
    taken = _req(gateway, "POST", "/user_chat/take", {"take_id": "b"}).json()
    assert taken == {"items": [], "quiet": [{"from": "psi_agent", "at": T0}]}
    assert _req(gateway, "POST", "/user_chat/take", {"take_id": "b"}).json() == taken
    assert _req(gateway, "POST", "/user_chat/take", {}).json() == {"items": [], "quiet": []}
    [desc] = [t["description"] for t in
              _req(gateway, "POST", "/step", {"action": "list_tools"}, role="psi_agent").json()["tools"]
              if t["name"] == "stay_quiet"]
    assert "Tess" in desc


def test_a_take_for_one_role_leaves_the_others_messages_and_quiet_calls_queued(gateway):
    _req(gateway, "POST", "/user_chat/config", {"roles": ["psi_agent", "other"], "stay_quiet": True})
    _call(gateway, "message_user", {"text": "from psi"}, "psi_agent")
    _call(gateway, "message_user", {"text": "from other"}, "other")
    _call(gateway, "stay_quiet", {}, "other")

    taken = _req(gateway, "POST", "/user_chat/take", {"from": "psi_agent"}).json()
    assert [(m["from"], m["text"]) for m in taken["items"]] == [("psi_agent", "from psi")] and taken["quiet"] == []
    assert [m["text"] for m in _req(gateway, "GET", "/user_chat/state").json()["pending"]] == ["from other"]
    rest = _req(gateway, "POST", "/user_chat/take", {}).json()
    assert [(m["from"], m["text"]) for m in rest["items"]] == [("other", "from other")]
    assert [q["from"] for q in rest["quiet"]] == ["other"]


@pytest.mark.parametrize("who", ["", 3])
def test_a_take_from_no_role_is_refused(gateway, who):
    assert _req(gateway, "POST", "/user_chat/take", {"from": who}).status_code == 400


def test_a_new_config_or_clear_removes_stay_quiet(gateway):
    _req(gateway, "POST", "/user_chat/config", {"roles": ["psi_agent"], "stay_quiet": True})
    _req(gateway, "POST", "/user_chat/config", {"roles": ["psi_agent"]})
    assert "stay_quiet" not in _names(gateway, "psi_agent") and "message_user" in _names(gateway, "psi_agent")
    _req(gateway, "POST", "/user_chat/config", {"roles": ["psi_agent"], "stay_quiet": True})
    _req(gateway, "POST", "/user_chat/clear", {})
    assert not {"stay_quiet", "message_user"} & _names(gateway, "psi_agent")


def test_turning_the_chat_off_or_clearing_it_leaves_a_backing_server_s_stay_quiet(gateway):
    gateway._tool_server_urls["stay_quiet"] = "http://backing"
    gateway._create_and_register_proxy_tool("http://backing", MCPTool(
        name="stay_quiet", description="backing", inputSchema={"type": "object", "properties": {}}))
    tools = gateway._mcp._tool_manager._tools

    _req(gateway, "POST", "/user_chat/config", {"roles": ["psi_agent"], "stay_quiet": True})
    _req(gateway, "POST", "/user_chat/config", {"roles": []})
    assert tools["stay_quiet"].description == "backing" and "message_user" not in tools
    _req(gateway, "POST", "/user_chat/config", {"roles": ["psi_agent"]})
    _req(gateway, "POST", "/user_chat/clear", {})
    assert tools["stay_quiet"].description == "backing" and "message_user" not in tools


@pytest.mark.parametrize("body", [{"roles": ["psi_agent"], "stay_quiet": "yes"}, {"roles": "psi_agent"},
                                  {"roles": ["psi_agent"], "colour": "red"}])
def test_a_bad_config_is_refused(gateway, body):
    assert _req(gateway, "POST", "/user_chat/config", body).status_code == 400


def test_an_unconfigured_gateway_lists_neither_tool(gateway):
    assert not {"message_user", "stay_quiet"} & _names(gateway, "default")


def test_get_time_reaches_a_configured_role_through_a_deny_all_while_the_clock_is_set(gateway):
    _req(gateway, "POST", "/tools/disable", {"role": "psi_agent", "tools": ["*"]})
    _req(gateway, "POST", "/tools/disable", {"role": "other", "tools": ["*"]})
    assert "get_time" not in _names(gateway, "psi_agent")
    _req(gateway, "POST", "/user_chat/config", {"roles": ["psi_agent"]})
    assert "get_time" not in _names(gateway, "psi_agent")  # no clock set, so no get_time at all

    assert _req(gateway, "PUT", "/clock/set-time", {"virtual_time": T0}).status_code == 200
    assert "get_time" in _names(gateway, "psi_agent") and "get_time" not in _names(gateway, "other")
    called = _call(gateway, "get_time", {}, "psi_agent")
    assert called.status_code == 200 and T0 in called.text
    _req(gateway, "POST", "/user_chat/clear", {})
    assert "get_time" not in _names(gateway, "psi_agent")


def test_the_check_tools_reach_a_configured_role_through_a_deny_all_once_notifications_name_it(gateway):
    checks = {"schedule_check", "list_scheduled_checks", "cancel_scheduled_check"}
    _req(gateway, "POST", "/tools/disable", {"role": "psi_agent", "tools": ["*"]})
    _req(gateway, "POST", "/notifications/config", {"roles": ["psi_agent"]})
    assert not checks & _names(gateway, "psi_agent")  # the task's deny-all, as before

    _req(gateway, "POST", "/user_chat/config", {"roles": ["psi_agent"]})
    assert checks <= _names(gateway, "psi_agent")
    _req(gateway, "POST", "/notifications/config", {"roles": []})
    assert not checks & _names(gateway, "psi_agent")


def test_message_user_and_stay_quiet_are_read_only_for_the_world_so_they_don_t_re_evaluate_state_triggers(gateway):
    _req(gateway, "POST", "/user_chat/config", {"roles": ["psi_agent"], "stay_quiet": True})
    engine = gateway._trigger_engine
    assert engine._is_readonly("message_user") and engine._is_readonly("stay_quiet")
    assert not engine._is_readonly("svc_thing")
