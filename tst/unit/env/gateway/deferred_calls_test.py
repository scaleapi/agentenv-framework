"""A tool result's `_meta` "agentenv.io/deferredCalls" asks the gateway to make a call on the same server later, at a time
on its clock, as the role that made the declaring call, and to report it to the trigger engine as the call it stands
for (a chat server's scheduled message posted at its time, as if its sender had posted it then). The declaration is
taken off the result before any caller sees it, each key is scheduled once, and a pending call ends with the
gateway."""
from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import httpx
import pytest
from mcp.types import CallToolResult, TextContent, Tool as MCPTool

from agent_env.env.gateway import AGENT_ENV_ROLE_HEADER, AGENT_ENV_ROLE_META_KEY, InternalMCPServer
from agent_env.env.gateway import gateway as gw_mod
from agent_env.env.gateway.caller import Caller
from agent_env.env.gateway.clock import _iso
from agent_env.env.gateway.gateway import Gateway
from agent_env.task_step.task_steps.prompt_agent import PromptAgentTaskStep

DEFERRED = "agentenv.io/deferredCalls"
URL = "http://chat/mcp"
SEND, SCHEDULE, DELIVER, NOTE = "chat_send", "chat_schedule", "chat_deliver", "chat_note"
T0 = "2026-07-06T19:00:00Z"
AT = "2026-07-06T19:03:00Z"


def _at(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _declared(key: str = "m1", at: str = AT, channel: str = "C1", text: str = "hi") -> dict:
    return {"key": key, "at": at, "call": {"tool": DELIVER, "arguments": {"id": key}},
            "report_as": {"tool": SEND, "arguments": {"channel": channel, "message": text}}}


def _json(data, *, is_error: bool = False, meta: dict | None = None) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=json.dumps(data))], isError=is_error, _meta=meta)


class _Chat:
    """A chat server: chat_schedule declares what `declare` holds, chat_deliver answers as `deliveries` say (an exception
    to raise or a result to return, else a posted message), and every call is recorded with the gateway clock's time."""

    def __init__(self):
        self.gw: Gateway | None = None
        self.calls: list[dict] = []
        self.declare: list = []
        self.schedule_error = False
        self.deliveries: list = []

    async def list_tools(self):
        schema = {"type": "object", "properties": {}}
        return SimpleNamespace(tools=[MCPTool(name=n, description="", inputSchema=schema)
                                      for n in (SEND, SCHEDULE, DELIVER, NOTE)])

    async def call_tool(self, name, arguments, *args, meta=None, **kwargs):
        self.calls.append({"tool": name, "args": dict(arguments or {}), "meta": meta,
                           "virtual": self.gw._clock.now(), "wall": datetime.now(timezone.utc)})
        if name == SCHEDULE:
            return _json({"ok": True, "scheduled": True}, is_error=self.schedule_error,
                         meta={DEFERRED: list(self.declare), "other": 1})
        if name == DELIVER:
            outcome = self.deliveries.pop(0) if self.deliveries else None
            if isinstance(outcome, BaseException):
                raise outcome
            if outcome is not None:
                return outcome
            return _json({"ok": True, "channel": "C1", "ts": "1783364580.000100", "message": {"text": "hi"}})
        return _json({"ok": True})

    def made(self, tool: str) -> list[dict]:
        return [c for c in self.calls if c["tool"] == tool]


@asynccontextmanager
async def _chat(tmp_path, monkeypatch):
    """A gateway in front of the chat server, inside its process lifespan, whose shutdown unwinds it at the end."""
    monkeypatch.setattr(gw_mod, "GATEWAY_TRAJECTORY_FILE", str(tmp_path / "trajectory.jsonl"))
    chat = _Chat()
    gw = Gateway(host="127.0.0.1", port=0, server_name="unit", internal_mcp_servers=[InternalMCPServer("chat", URL)])
    chat.gw = gw

    async def open_session(server, stack, report):
        return chat

    @asynccontextmanager
    async def no_mcp_sessions(app):
        yield

    gw._sessions.open_session = open_session  # type: ignore[method-assign]
    try:
        async with gw._process_lifespan(no_mcp_sessions)(None):
            yield gw, chat
    finally:
        gw._close_trajectory_file()


async def _request(gw: Gateway, method: str, path: str, **kw) -> httpx.Response:
    transport = httpx.ASGITransport(app=gw._mcp.streamable_http_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://gateway") as client:
        response = await client.request(method, path, **kw)
    assert response.status_code == 200, response.text
    return response


async def _mcp_call(gw: Gateway, role: str, tool: str, arguments: dict | None = None):
    token = Caller(role=role).bind()
    try:
        return await gw._mcp._tool_manager.call_tool(tool, arguments or {})
    finally:
        Caller.reset(token)


async def _schedule(gw: Gateway, chat: _Chat, *declared: dict, role: str = "alice"):
    chat.declare = list(declared)
    return await _mcp_call(gw, role, SCHEDULE)


async def _arm(gw: Gateway, virtual_time: str = T0, rate: float = 0) -> None:
    await _request(gw, "PUT", "/clock/set-time", json={"virtual_time": virtual_time,
                                                       "virtual_seconds_per_real_second": rate})


async def _deferred(gw: Gateway) -> list[dict]:
    return (await _request(gw, "GET", "/triggers/state")).json()["deferred_calls"]


async def _until(predicate, seconds: float = 5.0) -> None:
    deadline = time.monotonic() + seconds
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached"
        await asyncio.sleep(0.01)


async def _status(gw: Gateway, key: str = "m1") -> str:
    return next(c["status"] for c in await _deferred(gw) if c["key"] == key)


def _events(tmp_path, kind: str) -> list[dict]:
    lines = (tmp_path / "trajectory.jsonl").read_text().splitlines()
    return [e for e in map(json.loads, lines) if e.get("event_type") == kind]


def _reply_trigger(**when) -> dict:
    """An action trigger on the reported tool, its tool action echoing the call's arguments and its result."""
    return {"watch_roles": ["alice"], "triggers": [
        {"id": "reply", "when": {"type": "action", "tool": SEND, "repeat": True, **when},
         "actions": [{"type": "tool", "tool": NOTE, "as": "bob",
                      "args": {"text": "${args.message}", "channel": "${args.channel}", "ts": "${result.ts}"}}]}]}


async def _fired(gw: Gateway) -> int:
    state = (await _request(gw, "GET", "/triggers/state")).json()
    return next(t["fire_count"] for t in state["triggers"] if t["id"] == "reply")


# ---- the declaration on the result ----------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_the_declaration_is_taken_off_the_result_on_the_mcp_and_step_paths_and_out_of_the_trajectory(
        tmp_path, monkeypatch):
    async with _chat(tmp_path, monkeypatch) as (gw, chat):
        await _arm(gw)
        result = await _schedule(gw, chat, _declared("m1"))
        assert result.meta == {"other": 1}

        chat.declare = [_declared("m2")]
        step = await _request(gw, "POST", "/step", headers={AGENT_ENV_ROLE_HEADER: "alice"},
                              json={"action": "call_tool", "tool_name": SCHEDULE, "arguments": {}})
        assert step.json()["meta"] == {"other": 1}
        assert DEFERRED not in step.text

        assert [c["key"] for c in await _deferred(gw)] == ["m1", "m2"]
        assert DEFERRED not in (tmp_path / "trajectory.jsonl").read_text()


@pytest.mark.asyncio
async def test_a_result_without_the_declaration_is_answered_as_before(tmp_path, monkeypatch):
    async with _chat(tmp_path, monkeypatch) as (gw, chat):
        result = await _mcp_call(gw, "alice", SEND, {"channel": "C1"})
        assert result == _json({"ok": True})
        step = await _request(gw, "POST", "/step", json={"action": "call_tool", "tool_name": SEND, "arguments": {}})
        assert "meta" not in step.json()
        assert await _deferred(gw) == []


@pytest.mark.asyncio
async def test_an_error_result_has_the_declaration_taken_off_but_nothing_scheduled(tmp_path, monkeypatch):
    async with _chat(tmp_path, monkeypatch) as (gw, chat):
        chat.schedule_error = True
        result = await _schedule(gw, chat, _declared())
        assert result.isError and result.meta == {"other": 1}
        assert await _deferred(gw) == []


@pytest.mark.asyncio
async def test_each_key_is_scheduled_once_while_pending_and_once_done(tmp_path, monkeypatch):
    async with _chat(tmp_path, monkeypatch) as (gw, chat):
        await _arm(gw)
        await _schedule(gw, chat, _declared("m1"), _declared("m1", at="2026-07-06T19:01:00Z"))
        await _schedule(gw, chat, _declared("m1"))
        assert [(c["key"], c["at"]) for c in await _deferred(gw)] == [("m1", AT)]

        await _request(gw, "PUT", "/clock/advance", json={"virtual_time": AT})
        await _until(lambda: chat.made(DELIVER))
        await _schedule(gw, chat, _declared("m1", at=T0))
        await asyncio.sleep(0.1)
        assert len(chat.made(DELIVER)) == 1
        assert [(c["key"], c["status"]) for c in await _deferred(gw)] == [("m1", "done")]


@pytest.mark.asyncio
async def test_an_entry_that_breaks_the_contract_is_skipped_and_its_neighbours_scheduled(tmp_path, monkeypatch):
    async with _chat(tmp_path, monkeypatch) as (gw, chat):
        await _arm(gw)
        no_offset = {**_declared("bad-at"), "at": "2026-07-06T19:03:00"}
        no_call = {k: v for k, v in _declared("no-call").items() if k != "call"}
        await _schedule(gw, chat, no_offset, no_call, {"key": ""}, "junk", _declared("ok"))
        assert [c["key"] for c in await _deferred(gw)] == ["ok"]


# ---- when it is made ------------------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_it_is_made_once_the_clock_is_advanced_to_its_time_as_the_role_that_declared_it(tmp_path, monkeypatch):
    async with _chat(tmp_path, monkeypatch) as (gw, chat):
        await _arm(gw)
        await _schedule(gw, chat, _declared(), role="alice")
        await _request(gw, "PUT", "/clock/advance", json={"virtual_time": "2026-07-06T19:02:59Z"})
        await asyncio.sleep(0.2)
        assert chat.made(DELIVER) == [] and await _status(gw) == "pending"

        await _request(gw, "PUT", "/clock/advance", json={"virtual_time": AT})
        await _until(lambda: chat.made(DELIVER))
        [delivery] = chat.made(DELIVER)
        assert delivery["args"] == {"id": "m1"} and delivery["virtual"] >= _at(AT)
        assert delivery["meta"] == {AGENT_ENV_ROLE_META_KEY: "alice"}
        await _until(lambda: _events(tmp_path, "deferred_call"))
        assert [(c["status"], c["attempts"], c["role"], c["tool"]) for c in await _deferred(gw)] == [
            ("done", 1, "alice", DELIVER)]


@pytest.mark.asyncio
async def test_a_rearm_past_its_time_makes_it_and_one_before_its_time_holds_it(tmp_path, monkeypatch):
    async with _chat(tmp_path, monkeypatch) as (gw, chat):
        await _arm(gw)
        await _schedule(gw, chat, _declared("soon"), _declared("later", at="2026-07-08T09:00:00Z"))
        await _arm(gw, "2026-07-07T00:00:00Z")
        await _until(lambda: len(chat.made(DELIVER)) == 1)
        assert chat.made(DELIVER)[0]["args"] == {"id": "soon"}

        await _arm(gw, "2026-07-01T00:00:00Z")
        await asyncio.sleep(0.2)
        assert len(chat.made(DELIVER)) == 1 and await _status(gw, "later") == "pending"
        await _arm(gw, "2026-07-08T09:00:00Z")
        await _until(lambda: len(chat.made(DELIVER)) == 2)


@pytest.mark.asyncio
async def test_a_running_clock_makes_it_at_its_time_holds_it_at_rate_0_and_resumes_with_the_rate(
        tmp_path, monkeypatch):
    async with _chat(tmp_path, monkeypatch) as (gw, chat):
        await _arm(gw, rate=3600)  # a virtual hour per real second
        await _request(gw, "PUT", "/clock/set-rate", json={"virtual_seconds_per_real_second": 0})
        await _schedule(gw, chat, _declared(at=_iso(gw._clock.now() + timedelta(minutes=30))))
        await asyncio.sleep(0.8)
        assert chat.made(DELIVER) == [] and await _status(gw) == "pending"

        await _request(gw, "PUT", "/clock/set-rate", json={"virtual_seconds_per_real_second": 3600})
        await _until(lambda: chat.made(DELIVER))
        at = _at((await _deferred(gw))[0]["at"])
        assert at <= chat.made(DELIVER)[0]["virtual"] < at + timedelta(minutes=10)


@pytest.mark.asyncio
async def test_clearing_the_clock_moves_a_waiting_call_onto_wall_time(tmp_path, monkeypatch):
    async with _chat(tmp_path, monkeypatch) as (gw, chat):
        await _arm(gw)
        await _schedule(gw, chat, _declared())  # due in three virtual minutes, months before the wall clock
        await asyncio.sleep(0.2)
        assert chat.made(DELIVER) == []
        await _request(gw, "POST", "/clock/clear")
        await _until(lambda: chat.made(DELIVER))
        assert chat.made(DELIVER)[0]["virtual"] is None


@pytest.mark.asyncio
async def test_an_unarmed_clock_makes_it_at_its_wall_time(tmp_path, monkeypatch):
    async with _chat(tmp_path, monkeypatch) as (gw, chat):
        soon = datetime.now(timezone.utc) + timedelta(seconds=0.3)
        await _schedule(gw, chat, _declared("soon", at=_iso(soon)),
                        _declared("next-hour", at=_iso(soon + timedelta(hours=1))))
        await _until(lambda: chat.made(DELIVER))
        await asyncio.sleep(0.1)
        [delivery] = chat.made(DELIVER)
        assert delivery["args"] == {"id": "soon"} and delivery["wall"] >= soon and delivery["virtual"] is None
        assert await _status(gw, "next-hour") == "pending"


# ---- what the trigger engine sees ------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_it_is_reported_to_the_trigger_engine_as_the_declared_call_with_the_delivery_result(
        tmp_path, monkeypatch):
    async with _chat(tmp_path, monkeypatch) as (gw, chat):
        await _arm(gw)
        await _request(gw, "POST", "/triggers/register", json=_reply_trigger(where={"args.channel": {"equals": "C1"}}))
        await _schedule(gw, chat, _declared("m1", channel="C1", text="see you at 3"),
                        _declared("m2", channel="C2", at="2026-07-06T19:04:00Z"))
        assert chat.made(NOTE) == []  # declaring is not posting

        await _request(gw, "PUT", "/clock/advance", json={"virtual_time": "2026-07-06T19:05:00Z"})
        await _until(lambda: len(chat.made(DELIVER)) == 2 and chat.made(NOTE))
        await asyncio.sleep(0.1)
        [note] = chat.made(NOTE)
        assert note["args"] == {"text": "see you at 3", "channel": "C1", "ts": "1783364580.000100"}
        assert note["meta"] == {AGENT_ENV_ROLE_META_KEY: "bob"}
        assert await _fired(gw) == 1
        assert chat.made(SEND) == []  # reported as a send, never made as one


@pytest.mark.asyncio
async def test_it_is_reported_as_the_declaring_role_which_a_trigger_s_roles_select(tmp_path, monkeypatch):
    async with _chat(tmp_path, monkeypatch) as (gw, chat):
        await _arm(gw)
        await _request(gw, "POST", "/triggers/register", json=_reply_trigger(roles=["carol"]))
        await _schedule(gw, chat, _declared("by-alice"), role="alice")
        await _schedule(gw, chat, _declared("by-carol"), role="carol")
        await _request(gw, "PUT", "/clock/advance", json={"virtual_time": AT})
        await _until(lambda: len(chat.made(DELIVER)) == 2 and chat.made(NOTE))
        await asyncio.sleep(0.1)
        assert await _fired(gw) == 1
        assert {c["key"]: c["role"] for c in await _deferred(gw)} == {"by-alice": "alice", "by-carol": "carol"}


@pytest.mark.asyncio
async def test_a_trigger_tool_action_s_declaration_is_made_as_its_role_and_the_default_without_one(
        tmp_path, monkeypatch):
    async with _chat(tmp_path, monkeypatch) as (gw, chat):
        await _arm(gw)
        chat.declare = [_declared("as-dave")]
        await gw._trigger_engine._internal_call(SCHEDULE, {}, role="dave")
        chat.declare = [_declared("unattributed")]
        await gw._trigger_engine._internal_call(SCHEDULE, {})
        assert {c["key"]: c["role"] for c in await _deferred(gw)} == {"as-dave": "dave", "unattributed": "default"}


@pytest.mark.asyncio
async def test_an_error_delivery_is_final_logged_and_not_reported(tmp_path, monkeypatch):
    async with _chat(tmp_path, monkeypatch) as (gw, chat):
        await _arm(gw)
        await _request(gw, "POST", "/triggers/register", json=_reply_trigger())
        chat.deliveries = [_json({"ok": False, "error": "message_not_found"}, is_error=True)]
        await _schedule(gw, chat, _declared())
        await _request(gw, "PUT", "/clock/advance", json={"virtual_time": AT})
        await _until(lambda: chat.made(DELIVER))
        await asyncio.sleep(0.2)
        assert len(chat.made(DELIVER)) == 1 and chat.made(NOTE) == [] and await _fired(gw) == 0
        assert [(c["status"], c["attempts"]) for c in await _deferred(gw)] == [("failed", 1)]
        [event] = _events(tmp_path, "deferred_call")
        assert (event["key"], event["tool"], event["role"], event["ok"]) == ("m1", DELIVER, "alice", False)
        assert "message_not_found" in event["error"]

        await _schedule(gw, chat, _declared(at="2026-07-06T19:10:00Z"))  # a failed key may be declared afresh
        assert [(c["status"], c["at"]) for c in await _deferred(gw)] == [("pending", "2026-07-06T19:10:00Z")]


@pytest.mark.asyncio
async def test_a_transport_failure_is_retried_and_the_delivery_reported_once(tmp_path, monkeypatch):
    async with _chat(tmp_path, monkeypatch) as (gw, chat):
        monkeypatch.setattr(gw._deferred, "RETRY_BACKOFF_S", 0.01)
        await _arm(gw)
        await _request(gw, "POST", "/triggers/register", json=_reply_trigger())
        chat.deliveries = [ConnectionError("chat restarting"), RuntimeError("socket closed")]
        await _schedule(gw, chat, _declared())
        await _request(gw, "PUT", "/clock/advance", json={"virtual_time": AT})
        await _until(lambda: chat.made(NOTE))
        await asyncio.sleep(0.1)
        assert len(chat.made(DELIVER)) == 3 and len(chat.made(NOTE)) == 1 and await _fired(gw) == 1
        assert [(c["status"], c["attempts"]) for c in await _deferred(gw)] == [("done", 3)]


@pytest.mark.asyncio
async def test_retries_are_bounded_and_a_call_that_never_goes_through_fails(tmp_path, monkeypatch):
    async with _chat(tmp_path, monkeypatch) as (gw, chat):
        monkeypatch.setattr(gw._deferred, "RETRY_BACKOFF_S", 0.01)
        await _arm(gw)
        chat.deliveries = [ConnectionError("down")] * 50
        await _schedule(gw, chat, _declared())
        await _request(gw, "PUT", "/clock/advance", json={"virtual_time": AT})
        await _until(lambda: _events(tmp_path, "deferred_call"))
        assert len(chat.made(DELIVER)) == gw._deferred.RETRY_ATTEMPTS > 1
        assert [(c["status"], c["attempts"]) for c in await _deferred(gw)] == [("failed", gw._deferred.RETRY_ATTEMPTS)]
        assert _events(tmp_path, "deferred_call")[0]["ok"] is False


# ---- state, trajectory, shutdown ------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_state_lists_each_deferred_call_and_prompt_agent_does_not_wait_for_a_pending_one(tmp_path, monkeypatch):
    async with _chat(tmp_path, monkeypatch) as (gw, chat):
        await _arm(gw)
        await _schedule(gw, chat, _declared())
        state = (await _request(gw, "GET", "/triggers/state")).json()
        assert state["deferred_calls"] == [
            {"key": "m1", "at": AT, "tool": DELIVER, "role": "alice", "status": "pending", "attempts": 0}]
        assert state["triggers"] == [] and state["next_event_at"] is None
        assert PromptAgentTaskStep._is_settling(state) is False


@pytest.mark.asyncio
async def test_the_trajectory_records_each_deferred_call_on_both_clocks(tmp_path, monkeypatch):
    async with _chat(tmp_path, monkeypatch) as (gw, chat):
        await _arm(gw)
        await _schedule(gw, chat, _declared())
        await _request(gw, "PUT", "/clock/advance", json={"virtual_time": AT})
        await _until(lambda: _events(tmp_path, "deferred_call"))
        [event] = _events(tmp_path, "deferred_call")
        assert (event["key"], event["tool"], event["role"], event["ok"]) == ("m1", DELIVER, "alice", True)
        assert event["virtual_time"] == AT and event["timestamp_utc"]


@pytest.mark.asyncio
async def test_a_pending_call_is_cancelled_when_the_gateway_shuts_down(tmp_path, monkeypatch):
    async with _chat(tmp_path, monkeypatch) as (gw, chat):
        await _arm(gw)
        await _schedule(gw, chat, _declared())
        tasks = set(gw._deferred._tasks)
        assert len(tasks) == 1
    assert all(task.cancelled() for task in tasks) and not gw._deferred._tasks
    gw._clock.advance("2026-07-07T00:00:00Z")
    gw._trigger_engine.clock_changed()
    await asyncio.sleep(0.1)
    assert chat.made(DELIVER) == []
