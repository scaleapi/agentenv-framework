"""Unit tests for an agent's notifications (urn:agentenv:notifications/v1): scheduled checks and the notify action."""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from mcp.types import CallToolResult, TextContent
from mcp.types import Tool as MCPTool

from agent_env.env.gateway import gateway as gw_mod
from agent_env.env.gateway.clock import Clock
from agent_env.env.gateway.constants import GATEWAY_EXTENSIONS
from agent_env.env.gateway.gateway import Gateway
from agent_env.env.gateway.triggers import TriggerEngine, TriggerError
from agent_env.env.gateway.notifications import CHECK_TOOL_NAMES, NotificationError, Notifications

T0 = "2027-02-01T15:00:00Z"  # Mon 7:00 AM Pacific


class FakeNow:
    def __init__(self, start: float = 1000.0):
        self.t = start

    def __call__(self) -> float:
        return self.t

    def tick(self, seconds: float) -> None:
        self.t += seconds


def _frozen_clock(at: str = T0) -> Clock:
    clock = Clock(now_fn=FakeNow())
    clock.set_time(at, 0)
    return clock


def _notifications(clock: Clock | None = None, **config) -> Notifications:
    clock = clock or _frozen_clock()
    w = Notifications(lambda: clock)
    w.configure({"scheduled_checks": True, **config})
    return w


# ---- Notifications ----

def test_scheduled_checks_are_off_until_a_task_turns_them_on():
    w = Notifications(lambda: _frozen_clock())
    assert w.state()["scheduled_checks"] is False
    with pytest.raises(NotificationError, match="not available"):
        w.schedule("look again", at="PT6H")


def test_a_check_takes_a_duration_from_now_or_a_time_with_its_offset():
    w = _notifications()
    assert w.schedule("in six hours", at="PT6H")["next"] == "2027-02-01T21:00:00Z"
    assert w.schedule("tonight", at="2027-02-01T18:00:00-08:00")["next"] == "2027-02-02T02:00:00Z"
    assert [c["note"] for c in w.list()["checks"]] == ["in six hours", "tonight"]


@pytest.mark.parametrize("kwargs, error", [
    ({"note": "x"}, "give at"),
    ({"note": " ", "at": "PT1H"}, "note must say"),
    ({"note": "x", "at": "2027-02-01T06:00:00-08:00"}, "already passed"),
    ({"note": "x", "at": "2027-02-01T18:00:00"}, "RFC 3339|RFC3339"),
    ({"note": "x", "every": "PT10M"}, "at least"),
    ({"note": "x", "at": "PT1H", "until": "P1D"}, "until only applies"),
    ({"note": "x", "every": "PT6H", "until": "PT1H"}, "before the first"),
    ({"note": "x", "at": "PT1H", "every": "P3000000D"}, "too long"),
    ({"note": "x", "at": "P3000000D"}, "too far away"),
])
def test_bad_checks_are_refused_with_the_reason(kwargs, error):
    with pytest.raises(NotificationError, match=error):
        _notifications().schedule(**kwargs)


@pytest.mark.parametrize("text", ["PT3H", "P1D", "P1DT6H30M", "PT0.001S", "PT1M0.5S"])
def test_durations_are_echoed_in_the_form_the_tools_take(text):
    w = _notifications(min_interval="PT0.001S")
    assert w.schedule("x", every=text)["every"] == text
    assert _notifications(min_interval=text).state()["min_interval"] == text


def test_the_too_short_error_names_the_minimum_as_a_duration_to_send_back():
    with pytest.raises(NotificationError, match=r"at least PT3H apart"):
        _notifications(min_interval="PT3H").schedule("x", every="PT1H")


def test_the_active_cap_refuses_one_more_until_one_is_cancelled():
    w = _notifications(max_active=2)
    first = w.schedule("a", at="PT1H")["check_id"]
    w.schedule("b", at="PT2H")
    with pytest.raises(NotificationError, match="already have 2"):
        w.schedule("c", at="PT3H")
    w.cancel(first)
    w.schedule("c", at="PT3H")
    with pytest.raises(NotificationError, match="no active scheduled check"):
        w.cancel(first)


def test_take_hands_out_each_due_check_once_and_a_one_shot_is_done():
    clock = _frozen_clock()
    w = _notifications(clock)
    w.schedule("look at 9", at="PT2H")
    w.schedule("look at noon", at="PT5H")
    assert w.state()["next_due_at"] == "2027-02-01T17:00:00Z"
    clock.set_time("2027-02-01T17:00:00Z", 0)
    taken = w.take({})["items"]
    assert [(i["kind"], i["note"], i["at"]) for i in taken] == [("scheduled_check", "look at 9", "2027-02-01T17:00:00Z")]
    assert w.take({})["items"] == []
    assert [c["note"] for c in w.list()["checks"]] == ["look at noon"]
    assert w.state()["next_due_at"] == "2027-02-01T20:00:00Z"


def test_next_due_at_is_the_earliest_of_the_checks_and_the_pending_notifications():
    clock = _frozen_clock()
    n = _notifications(clock)
    n.schedule("look again", at="PT6H")
    clock.set_time("2027-02-01T16:00:00Z", 0)
    n.notify("Messages", "Jamie: still on for tonight?")  # queued at 16:00, before the check at 21:00
    assert n.state()["next_due_at"] == "2027-02-01T16:00:00Z"
    n.take({})
    assert n.state()["next_due_at"] == "2027-02-01T21:00:00Z"


def test_a_notification_queued_before_the_clock_was_set_is_due_once_it_is():
    clock = Clock(now_fn=FakeNow())
    n = Notifications(lambda: clock)
    n.notify("Gmail", "New email")
    assert n.state()["next_due_at"] is None
    clock.set_time(T0, 0)
    assert n.state()["next_due_at"] == T0


def test_a_repeating_check_moves_past_a_long_jump_and_says_how_often_it_came_due():
    clock = _frozen_clock()
    w = _notifications(clock)
    w.schedule("check rugs", at="PT5H", every="PT6H", until="P4D")
    clock.set_time("2027-02-02T15:00:00Z", 0)  # a day later: due at 20:00, 02:00, 08:00, 14:00
    [item] = w.take({})["items"]
    assert (item["at"], item["times"]) == ("2027-02-01T20:00:00Z", 4)
    assert w.list()["checks"][0]["next"] == "2027-02-02T20:00:00Z"


def test_a_jump_over_a_short_every_is_counted_not_stepped():
    clock = _frozen_clock()
    w = _notifications(clock, min_interval="PT0.001S")
    w.schedule("tick", every="PT0.001S")
    clock.set_time("2027-02-02T15:00:00Z", 0)
    [item] = w.take({})["items"]
    assert item["times"] == 86_400_000
    assert w.list()["checks"][0]["next"] == "2027-02-02T15:00:00.001000Z"


def test_a_repeating_check_ends_at_until():
    clock = _frozen_clock()
    w = _notifications(clock)
    w.schedule("check", every="P1D", until="P2D")
    clock.set_time("2027-02-05T15:00:00Z", 0)
    assert w.take({})["items"][0]["times"] == 2
    assert w.list()["checks"] == [] and w.state()["next_due_at"] is None


def test_take_until_looks_ahead_without_moving_the_clock():
    w = _notifications()
    w.schedule("later", at="PT3H")
    assert w.take({"until": "PT1H"})["items"] == []
    assert [i["note"] for i in w.take({"until": "PT3H"})["items"]] == ["later"]


def test_notifications_queue_at_the_virtual_time_and_are_taken_once():
    clock = _frozen_clock()
    w = Notifications(lambda: clock)  # notifications need no switch
    w.notify("Messages", "Jamie: did the rug thing get sorted?", trigger_id="t1")
    pending = w.state()["pending_notifications"]
    assert [(n["channel"], n["at"], n["trigger_id"]) for n in pending] == [("Messages", T0, "t1")]
    [item] = w.take({})["items"]
    assert (item["kind"], item["text"]) == ("trigger", "Jamie: did the rug thing get sorted?")
    assert w.take({})["items"] == [] and w.state()["pending_notifications"] == []


def test_due_is_judged_on_the_time_not_its_text():
    clock = _frozen_clock()
    w = _notifications(clock)
    w.notify("Messages", "on the second")                    # 15:00:00Z sorts after 15:00:00.500000Z as text
    w.schedule("half a second on", at="PT0.5S")
    items = w.take({"until": "PT0.75S"})["items"]
    assert [i.get("text") or i.get("note") for i in items] == ["on the second", "half a second on"]


def test_turning_checks_off_cancels_the_ones_still_to_come():
    clock = _frozen_clock()
    w = _notifications(clock)
    w.schedule("sweep", every="P1D")
    w.configure({"scheduled_checks": False})
    clock.set_time("2027-02-05T15:00:00Z", 0)
    assert w.state()["next_due_at"] is None and w.take({})["items"] == []
    assert [c["status"] for c in w.state()["checks"]] == ["cancelled"]


def test_history_is_bounded_and_delivered_notifications_are_dropped():
    from agent_env.env.gateway import notifications as notifications_mod
    w = _notifications()
    for n in range(notifications_mod._HISTORY + 10):
        w.cancel(w.schedule(f"check {n}", at="PT1H")["check_id"])
    checks = w.state()["checks"]
    assert len(checks) == notifications_mod._HISTORY and checks[-1]["note"] == f"check {notifications_mod._HISTORY + 9}"
    w.notify("Messages", "hi")
    w.take({})
    assert w._queue == []


def test_clear_turns_checks_off_and_forgets_everything():
    w = _notifications()
    w.schedule("x", at="PT1H")
    w.notify("Gmail", "hello")
    w.clear()
    st = w.state()
    assert (st["scheduled_checks"], st["checks"], st["pending_notifications"]) == (False, [], [])


@pytest.mark.parametrize("body, error", [
    ({"scheduled_checks": "yes"}, "boolean"),
    ({"max_active": 0}, "positive integer"),
    ({"min_interval": "1h"}, "min_interval"),
])
def test_bad_config_is_refused(body, error):
    with pytest.raises(NotificationError, match=error):
        Notifications(lambda: _frozen_clock()).configure(body)


# ---- the gateway ----

@pytest.fixture
def gateway(tmp_path, monkeypatch) -> Gateway:
    monkeypatch.setattr(gw_mod, "GATEWAY_TRAJECTORY_FILE", str(tmp_path / "trajectory.jsonl"))
    gw = Gateway(host="127.0.0.1", port=0, server_name="AgentEnvGateway", internal_mcp_servers=[])
    gw._clock = _frozen_clock()
    gw._server_tools["mcp-svc"] = [MCPTool(name="svc_thing", description="", inputSchema={"type": "object"})]
    gw._tools_discovered = True
    yield gw
    gw._close_trajectory_file()


def _request(gw: Gateway, method: str, path: str, body: dict | None = None) -> httpx.Response:
    async def _do() -> httpx.Response:
        transport = httpx.ASGITransport(app=gw._mcp.streamable_http_app())
        async with httpx.AsyncClient(transport=transport, base_url="http://gateway") as client:
            return await client.request(method, path, json=body)

    return asyncio.run(_do())


def _gateway_tool_names(gw: Gateway) -> set[str]:
    return {t.name for t in gw._server_tools.get("gateway", [])}


def test_the_card_names_the_extension_notifications_and_not_a2a_push_notifications():
    [ext] = [e for e in GATEWAY_EXTENSIONS if e["uri"] == "urn:agentenv:notifications/v1"]
    assert "Unrelated to A2A push notifications" in ext["description"]
    assert {name: m["endpoint"] for name, m in ext["params"]["methods"].items()} == {
        "config": "/notifications/config", "state": "/notifications/state", "take": "/notifications/take",
        "clear": "/notifications/clear"}


def test_the_tools_appear_only_once_a_task_turns_them_on_and_go_on_clear(gateway):
    tools = gateway._mcp._tool_manager._tools
    assert not set(CHECK_TOOL_NAMES) & set(tools)
    r = _request(gateway, "POST", "/notifications/config", {"max_active": 5})
    assert r.status_code == 200 and r.json()["max_active"] == 5
    assert set(CHECK_TOOL_NAMES) <= set(tools) and set(CHECK_TOOL_NAMES) <= _gateway_tool_names(gateway)
    assert _request(gateway, "POST", "/notifications/clear", {}).status_code == 200
    assert not set(CHECK_TOOL_NAMES) & set(tools) and not set(CHECK_TOOL_NAMES) & _gateway_tool_names(gateway)


def test_turning_checks_off_removes_the_tools(gateway):
    _request(gateway, "POST", "/notifications/config", {})
    _request(gateway, "POST", "/notifications/config", {"scheduled_checks": False})
    assert not set(CHECK_TOOL_NAMES) & set(gateway._mcp._tool_manager._tools)


def test_config_is_refused_over_a_backing_server_tool_of_the_same_name(gateway):
    gateway._tool_server_urls["schedule_check"] = "http://svc:1/mcp"
    r = _request(gateway, "POST", "/notifications/config", {})
    assert r.status_code == 409 and "schedule_check" in r.json()["error"]


def test_config_waits_for_tool_discovery_and_sees_what_it_found(gateway, monkeypatch):
    gateway._tools_discovered = False

    async def discover():
        gateway._tool_server_urls["schedule_check"] = "http://svc:1/mcp"
        gateway._tools_discovered = True

    monkeypatch.setattr(gateway, "_ensure_tools_discovered", discover)
    r = _request(gateway, "POST", "/notifications/config", {})
    assert r.status_code == 409 and "schedule_check" in r.json()["error"]


def test_config_is_a_503_while_tool_discovery_fails(gateway, monkeypatch):
    async def discover():
        raise RuntimeError("backing server down")

    monkeypatch.setattr(gateway, "_ensure_tools_discovered", discover)
    r = _request(gateway, "POST", "/notifications/config", {})
    assert r.status_code == 503 and "retry" in r.json()["error"]
    assert not set(CHECK_TOOL_NAMES) & set(gateway._mcp._tool_manager._tools)


def test_a_bad_config_is_a_400(gateway):
    assert _request(gateway, "POST", "/notifications/config", {"max_active": -1}).status_code == 400


def test_the_agent_schedules_through_the_tool_and_the_harness_takes_it(gateway):
    _request(gateway, "POST", "/notifications/config", {})
    tool = gateway._mcp._tool_manager._tools["schedule_check"]
    made = asyncio.run(tool.fn(note="check rugs", at="PT2H"))
    assert made["next"] == "2027-02-01T17:00:00Z"
    refused = asyncio.run(tool.fn(note="check", every="PT5M"))
    assert isinstance(refused, CallToolResult) and refused.isError
    state = _request(gateway, "GET", "/notifications/state").json()
    assert state["next_due_at"] == "2027-02-01T17:00:00Z"
    gateway._clock.set_time("2027-02-01T17:00:00Z", 0)
    items = _request(gateway, "POST", "/notifications/take", {}).json()["items"]
    assert [i["note"] for i in items] == ["check rugs"]


def test_list_is_read_only_so_it_does_not_re_check_state_triggers(gateway):
    _request(gateway, "POST", "/notifications/config", {})
    annotations = {t.name: t.annotations for t in gateway._server_tools["gateway"]}
    assert annotations["list_scheduled_checks"].readOnlyHint is True
    assert annotations["schedule_check"].readOnlyHint is False


# ---- notify ----

class _FakeGateway:
    def __init__(self) -> None:
        self.events: list[dict] = []
        self._role_rules_lock = asyncio.Lock()
        self._server_tools: dict = {}
        self._tool_server_urls: dict = {}
        self._clock = _frozen_clock()
        self._notifications = Notifications(lambda: self._clock)
        self.TOOL_CALL_TIMEOUT_S = 30

    def _query_changelog_id(self):
        return 1

    async def _log_event(self, event):
        self.events.append(dict(event))
        return f"event_{len(self.events)}"

    async def _ensure_tools_discovered(self):
        pass


def _ok(text: str = "{}") -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=text)], isError=False)


def _notify_trigger(**action) -> dict:
    return {"id": "jamie-texts", "when": {"type": "action", "tool": "messaging_send_message"},
            "actions": [{"type": "notify", "channel": "Messages", "text": "Jamie: ok", **action}]}


@pytest.mark.asyncio
async def test_notify_queues_a_templated_notification():
    gw = _FakeGateway()
    engine = TriggerEngine(gw)
    engine.register({"watch_roles": ["default"],
                     "triggers": [_notify_trigger(text="${args.to}: got your text about ${args.subject}")]})
    engine.on_tool_call("default", "messaging_send_message", {"to": "Jamie", "subject": "the rug"}, _ok())
    await asyncio.sleep(0.2)
    [note] = gw._notifications.state()["pending_notifications"]
    assert (note["channel"], note["text"], note["at"]) == ("Messages", "Jamie: got your text about the rug", T0)
    ok = [e for e in engine.state()["events"] if e["kind"] == "action_ok"]
    assert ok and ok[0]["detail"]["notify"] == "Messages" and "got your text" not in json.dumps(ok[0])


@pytest.mark.parametrize("action, error", [
    ({"channel": ""}, "channel"),
    ({"text": " "}, "text"),
    ({"verify": {"tool": "x", "predicate": {"exists": True}}}, "nothing to verify"),
    ({"text": "${result.id}", "channel": "Gmail"}, None),
])
def test_notify_is_validated_at_registration(action, error):
    engine = TriggerEngine(_FakeGateway())
    body = {"watch_roles": ["default"], "triggers": [_notify_trigger(**action)]}
    if error is None:
        engine.register(body)
    else:
        with pytest.raises(TriggerError, match=error):
            engine.register(body)
