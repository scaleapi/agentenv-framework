"""Role-rule precedence in Gateway._is_disabled / _apply_rule: a role's own rules (explicit, then its
wildcard) fully decide before the global "*" role is consulted, which is only a default for roles that
say nothing. A hide that must survive any later rule is an internal_only_tool, not a "*"-role rule."""
from __future__ import annotations

import asyncio

import pytest
from agentenv_protocol import ROLE_HEADER

from agent_env.env.gateway import AGENT_ENV_ROLE_HEADER
from agent_env.env.gateway.gateway import Gateway


def _gw() -> Gateway:
    return Gateway(host="127.0.0.1", port=0, server_name="t", internal_mcp_servers=[])


def test_the_gateway_reads_the_role_header_the_protocol_tool_client_sends():
    assert AGENT_ENV_ROLE_HEADER == ROLE_HEADER


def test_default_is_enabled_and_role_rules_are_scoped():
    gw = _gw()
    assert gw._is_disabled("default", "x") is False
    gw._apply_rule("default", "x", True)
    assert gw._is_disabled("default", "x") is True
    assert gw._is_disabled("other", "x") is False


def test_deny_all_then_allow_one_for_a_role():
    gw = _gw()
    gw._apply_rule("executor", "*", True)
    gw._apply_rule("executor", "slack_send_message", False)
    assert gw._is_disabled("executor", "slack_send_message") is False
    assert gw._is_disabled("executor", "gdrive_search") is True
    assert gw._is_disabled("default", "gdrive_search") is False


def test_a_global_hide_is_only_a_default_for_roles_that_say_nothing():
    """A "*"-role rule covers every role that has no rule of its own, but it does not outrank a role's
    own wildcard: `enable role=default tools="*"` means what it says. A hide that must survive that is
    an internal_only_tool (see below), which is guarded unconditionally."""
    gw = _gw()
    gw._apply_rule("*", "gdrive_touch_file", True)
    assert gw._is_disabled("default", "gdrive_touch_file") is True
    assert gw._is_disabled("brand-new-role", "gdrive_touch_file") is True
    gw._apply_rule("default", "*", False)                    # enable role=default tools="*"
    assert gw._is_disabled("default", "gdrive_touch_file") is False
    assert gw._is_disabled("brand-new-role", "gdrive_touch_file") is True   # untouched roles keep the default


def test_a_role_deny_all_is_not_undone_by_a_global_explicit_enable():
    """The converse leak: a globally enabled tool must not stay reachable for a role that was denied
    everything. The role's own wildcard is consulted before any "*"-role rule."""
    gw = _gw()
    gw._apply_rule("*", "slack_send_message", False)   # globally enabled, explicitly
    assert gw._is_disabled("locked-down", "slack_send_message") is False
    gw._apply_rule("locked-down", "*", True)           # disable role=locked-down tools="*"
    assert gw._is_disabled("locked-down", "slack_send_message") is True
    assert gw._is_disabled("other", "slack_send_message") is False
    gw._apply_rule("locked-down", "slack_send_message", False)  # an explicit role rule still wins
    assert gw._is_disabled("locked-down", "slack_send_message") is False


def test_global_deny_all_with_role_wildcard_enable():
    gw = _gw()
    gw._apply_rule("*", "*", True)
    assert gw._is_disabled("default", "anything") is True
    gw._apply_rule("default", "*", False)
    assert gw._is_disabled("default", "anything") is False
    assert gw._is_disabled("other", "anything") is True


def test_fire_triggers_forwards_the_engine_s_tasks_and_bound(monkeypatch):
    """What the call sites hand to _await_barriers is whatever the engine computed — and a broken
    engine yields nothing to wait on rather than breaking the agent's tool call."""
    gw = _gw()
    monkeypatch.setattr(gw._trigger_engine, "on_tool_call", lambda *a: (["task"], 7.5))
    assert gw._fire_triggers("default", "t", {}, None) == (["task"], 7.5)

    def boom(*a):
        raise RuntimeError("engine broke")
    monkeypatch.setattr(gw._trigger_engine, "on_tool_call", boom)
    assert gw._fire_triggers("default", "t", {}, None) == ([], None)


@pytest.mark.asyncio
async def test_a_barrier_holds_the_call_until_the_fire_settles(monkeypatch):
    """The guarantee itself: the call is not answered until the fire it barriers on has finished."""
    gw = _gw()
    landed = []

    async def fire():
        await asyncio.sleep(0.05)
        landed.append("mirror")

    task = asyncio.get_running_loop().create_task(fire())
    task.set_name("gsuite-mx-create-drive-1")
    await gw._await_barriers([task])
    assert landed == ["mirror"] and task.done()


@pytest.mark.asyncio
async def test_a_barrier_uses_the_bound_the_trigger_asked_for(monkeypatch):
    gw = _gw()
    events = []

    async def record(event):
        events.append(event)
        return "e1"
    monkeypatch.setattr(gw, "_log_event", record)
    gw.TRIGGER_BARRIER_TIMEOUT_S = 30.0  # would hang the test if the per-trigger ask were ignored
    slow = asyncio.get_running_loop().create_task(asyncio.sleep(5))
    slow.set_name("gsuite-m1-create")
    await gw._await_barriers([slow], timeout_s=0.05)
    assert events and events[0]["event_type"] == "trigger_barrier_timeout"
    assert events[0]["trigger_ids"] == ["gsuite-m1-create"] and events[0]["timeout_s"] == 0.05
    assert events[0]["at"] == "provoking_call"
    assert not slow.done()  # fail-open: the fire keeps running, the call was answered
    slow.cancel()
    await gw._await_barriers([])  # no-op
