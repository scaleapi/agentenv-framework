"""@tool(defers=Defers(...)): a tool's successful reply declares, under DEFERRED_CALLS_META_KEY, the deferred calls its
``using`` method builds, on both FastMCPs. The reply is otherwise unchanged, the method called in process returns its
plain value, a declaration that cannot be made leaves the reply as it is, and what a Defers names is checked at mount."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import anyio
import httpx
import pytest

pytest.importorskip("mcp.server.fastmcp")
fastmcp = pytest.importorskip("fastmcp")

from mcp.shared.memory import create_connected_server_and_client_session  # noqa: E402
from mcp.types import CallToolResult, TextContent  # noqa: E402

from agentenv_protocol import (  # noqa: E402
    DEFERRED_CALLS_META_KEY,
    AgentEnvEnvironment,
    DeferredCall,
    Defers,
    EnvironmentTool,
    environment_card,
    tool,
)

AT = "2026-07-06T19:03:00Z"
_MOMENT = datetime(2026, 7, 6, 19, 3, tzinfo=timezone.utc)
_REMINDER = Defers(call="deliver_reminder", report_as="send_notification", using="_delivery")


def _reminder(arguments: dict, result) -> DeferredCall | None:
    if not result.get("ok"):
        return None
    return DeferredCall(key=result["reminder_id"], at=result["remind_at"],
                        arguments={"reminder_id": result["reminder_id"]},
                        report_arguments={"event_id": arguments["event_id"]})


@environment_card(name="reminders")
class _Reminders(AgentEnvEnvironment):
    def __init__(self, delivery=_reminder) -> None:
        self.delivery = delivery
        self.seen: list = []

    @tool(hidden_by_default=True)
    def deliver_reminder(self, reminder_id: str) -> str:
        return json.dumps({"ok": True, "reminder_id": reminder_id})

    @tool()
    def send_notification(self, event_id: str) -> str:
        return json.dumps({"ok": True})

    @tool(defers=_REMINDER)
    def schedule_reminder(self, event_id: str, remind_at: str) -> str:
        return json.dumps({"ok": event_id != "none", "reminder_id": f"r-{event_id}", "remind_at": remind_at})

    @tool(defers=_REMINDER)
    async def schedule_result(self, event_id: str) -> CallToolResult:
        return CallToolResult(content=[TextContent(type="text", text="scheduled")],
                              structuredContent={"ok": True, "reminder_id": f"r-{event_id}", "remind_at": AT},
                              _meta={"agentenv.io/group": "reminders"})

    def _delivery(self, arguments: dict, result):
        self.seen.append((arguments, result))
        return self.delivery(arguments, result)


class _Undeclared(_Reminders):
    @tool()
    def schedule_reminder(self, event_id: str, remind_at: str) -> str:
        return _Reminders.schedule_reminder(self, event_id, remind_at)

    @tool()
    async def schedule_result(self, event_id: str) -> CallToolResult:
        return await _Reminders.schedule_result(self, event_id)


async def _called(env: AgentEnvEnvironment, backend: str, name: str, arguments: dict) -> tuple[list, dict | None, dict]:
    """The reply's text blocks, structured content and ``_meta``, through mcp's FastMCP or fastmcp."""
    with anyio.fail_after(30):
        if backend == "mcp":
            async with create_connected_server_and_client_session(env.create_app()) as session:
                result = await session.call_tool(name, arguments)
            return [c.text for c in result.content], result.structuredContent, dict(result.meta or {})
        app = fastmcp.FastMCP("reminders")
        env.mount(app)
        async with fastmcp.Client(app) as client:
            result = await client.call_tool(name, arguments)
        return [c.text for c in result.content], result.structured_content, dict(result.meta or {})


_BACKENDS = pytest.mark.parametrize("backend", ["mcp", "fastmcp"])


@_BACKENDS
@pytest.mark.asyncio
async def test_a_successful_reply_declares_its_deferred_call_and_is_otherwise_unchanged(backend):
    env = _Reminders()
    arguments = {"event_id": "e1", "remind_at": "2026-07-06T14:03:00-05:00"}
    text, structured, meta = await _called(env, backend, "schedule_reminder", arguments)
    plain = await _called(_Undeclared(), backend, "schedule_reminder", arguments)

    assert meta.pop(DEFERRED_CALLS_META_KEY) == [{
        "key": "r-e1", "at": AT,
        "call": {"tool": "deliver_reminder", "arguments": {"reminder_id": "r-e1"}},
        "report_as": {"tool": "send_notification", "arguments": {"event_id": "e1"}}}]
    assert (text, structured, meta) == plain
    assert env.seen == [(arguments, {"ok": True, "reminder_id": "r-e1", "remind_at": arguments["remind_at"]})]


@_BACKENDS
@pytest.mark.asyncio
async def test_none_declares_nothing(backend):
    text, _structured, meta = await _called(_Reminders(), backend, "schedule_reminder", {"event_id": "none", "remind_at": AT})
    assert DEFERRED_CALLS_META_KEY not in meta and json.loads(text[0])["ok"] is False


@_BACKENDS
@pytest.mark.asyncio
async def test_a_list_declares_each_of_its_calls(backend):
    def twice(arguments: dict, result) -> list[DeferredCall]:
        return [DeferredCall(key=f"{result['reminder_id']}-{n}", at=AT, arguments={"reminder_id": result["reminder_id"]})
                for n in (1, 2)]

    _text, _structured, meta = await _called(_Reminders(twice), backend, "schedule_reminder", {"event_id": "e1", "remind_at": AT})
    assert meta[DEFERRED_CALLS_META_KEY] == [
        {"key": f"r-e1-{n}", "at": AT, "call": {"tool": "deliver_reminder", "arguments": {"reminder_id": "r-e1"}},
         "report_as": {"tool": "send_notification", "arguments": {}}} for n in (1, 2)]


@_BACKENDS
@pytest.mark.asyncio
async def test_a_call_tool_result_keeps_its_meta_and_gains_the_declaration(backend):
    env = _Reminders()
    text, structured, meta = await _called(env, backend, "schedule_result", {"event_id": "e2"})
    plain_text, plain_structured, plain_meta = await _called(_Undeclared(), backend, "schedule_result", {"event_id": "e2"})

    assert [entry["key"] for entry in meta.pop(DEFERRED_CALLS_META_KEY)] == ["r-e2"]
    assert meta == plain_meta and meta["agentenv.io/group"] == "reminders"
    assert (text, structured) == (plain_text, plain_structured) == (["scheduled"], {"ok": True, "reminder_id": "r-e2", "remind_at": AT})
    assert env.seen == [({"event_id": "e2"}, structured)]


class _SelfDeclared(_Reminders):
    @tool(defers=_REMINDER)
    async def schedule_result(self, event_id: str) -> CallToolResult:
        own = {"key": "own-1", "at": AT, "call": {"tool": "deliver_reminder", "arguments": {"reminder_id": "own-1"}},
               "report_as": {"tool": "send_notification", "arguments": {"event_id": event_id}}}
        return CallToolResult(content=[TextContent(type="text", text="scheduled")],
                              structuredContent={"ok": True, "reminder_id": f"r-{event_id}", "remind_at": AT},
                              _meta={DEFERRED_CALLS_META_KEY: [own], "agentenv.io/group": "reminders"})


@_BACKENDS
@pytest.mark.asyncio
async def test_calls_a_call_tool_result_declares_itself_are_kept_before_the_decorators(backend):
    _, _, meta = await _called(_SelfDeclared(), backend, "schedule_result", {"event_id": "e3"})
    assert [entry["key"] for entry in meta[DEFERRED_CALLS_META_KEY]] == ["own-1", "r-e3"]
    assert meta["agentenv.io/group"] == "reminders"


@pytest.mark.parametrize("at, expected", [
    (int(_MOMENT.timestamp()), AT),
    (_MOMENT.timestamp() + 0.5, "2026-07-06T19:03:00.500000Z"),
    (_MOMENT.astimezone(timezone(timedelta(hours=9))), AT),
    ("2026-07-06T14:03:00-05:00", AT),
    ("2026-07-06 19:03:00.25z", "2026-07-06T19:03:00.250000Z"),
])
@pytest.mark.asyncio
async def test_at_is_declared_as_a_utc_instant(at, expected):
    env = _Reminders(lambda arguments, result: DeferredCall(key="k", at=at))
    _text, _structured, meta = await _called(env, "mcp", "schedule_reminder", {"event_id": "e1", "remind_at": AT})
    assert meta[DEFERRED_CALLS_META_KEY][0]["at"] == expected


def _raises(arguments: dict, result) -> DeferredCall:
    raise RuntimeError("the reminder store is down")


@_BACKENDS
@pytest.mark.parametrize("delivery, reason", [
    (lambda arguments, result: DeferredCall(key="k", at=datetime(2026, 7, 6, 19, 3)), "has no UTC offset"),
    (lambda arguments, result: DeferredCall(key="k", at="2026-07-06T19:03:00"), "not an RFC 3339 instant with an offset"),
    (lambda arguments, result: DeferredCall(key="k", at="tomorrow"), "not an RFC 3339 instant with an offset"),
    (lambda arguments, result: DeferredCall(key="k", at=True), "not a time"),
    (lambda arguments, result: DeferredCall(key="", at=AT), "a non-empty string key"),
    (lambda arguments, result: {"key": "k", "at": AT}, "dict, not a DeferredCall"),
    (lambda arguments, result: [DeferredCall(key="k", at=AT), "k2"], "str, not a DeferredCall"),
    (lambda arguments, result: DeferredCall(key="k", at=AT, arguments={"when": _MOMENT}), "JSON"),
    (_raises, "the reminder store is down"),
])
@pytest.mark.asyncio
async def test_a_declaration_that_cannot_be_made_is_logged_and_the_reply_kept(backend, delivery, reason, caplog):
    arguments = {"event_id": "e1", "remind_at": AT}
    text, structured, meta = await _called(_Reminders(delivery), backend, "schedule_reminder", arguments)

    assert (text, structured, meta) == await _called(_Undeclared(), backend, "schedule_reminder", arguments)
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING" and "schedule_reminder" in r.getMessage()]
    assert len(warnings) == 1 and reason in warnings[0], caplog.text


@environment_card(name="replies")
class _Replies(_Reminders):
    @tool(defers=_REMINDER)
    def as_dict(self, event_id: str) -> dict:
        return {"ok": True, "reminder_id": event_id, "remind_at": AT}

    @tool(defers=_REMINDER)
    def as_text(self, event_id: str) -> str:
        return f"scheduled {event_id}"

    @tool(defers=_REMINDER)
    def as_json_list(self, event_id: str) -> str:
        return json.dumps([event_id])

    @tool(defers=_REMINDER)
    def as_text_result(self, event_id: str) -> CallToolResult:
        return CallToolResult(content=[TextContent(type="text", text=json.dumps({"ok": False}))])

    @tool(defers=_REMINDER)
    def as_error(self, event_id: str) -> CallToolResult:
        return CallToolResult(isError=True, content=[TextContent(type="text", text="no such event")])


@pytest.mark.asyncio
async def test_using_gets_the_reply_parsed_when_it_is_json_and_as_it_is_otherwise_and_no_call_for_an_error():
    seen: list = []
    for name in ("as_dict", "as_text", "as_json_list", "as_text_result", "as_error"):
        await _called(_Replies(lambda arguments, result: seen.append(result)), "mcp", name, {"event_id": "e3"})
    assert seen == [{"ok": True, "reminder_id": "e3", "remind_at": AT}, "scheduled e3", ["e3"], {"ok": False}]


@pytest.mark.asyncio
async def test_the_card_advertises_what_a_tool_defers():
    env = _Reminders()
    app = env.create_app()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app.streamable_http_app()), base_url="http://t") as http:
        card = {t["name"]: t for t in (await http.get("/.well-known/agent-env.json")).json()["capabilities"]["tools"]}
    assert card["schedule_reminder"]["defers"] == card["schedule_result"]["defers"] == {
        "call": "deliver_reminder", "report_as": "send_notification"}
    assert card["send_notification"].get("defers") is None and card["deliver_reminder"].get("defers") is None
    assert {t.name: t.defers for t in env.mounted_card().capabilities.tools}["schedule_reminder"] == {
        "call": "deliver_reminder", "report_as": "send_notification"}


def test_the_method_called_in_process_returns_its_plain_value():
    env = _Reminders()
    env.create_app()
    assert env.schedule_reminder("e1", AT) == json.dumps({"ok": True, "reminder_id": "r-e1", "remind_at": AT})
    assert env.seen == []


@pytest.mark.asyncio
async def test_defers_names_resolve_environment_name_as_tool_names_do():
    @environment_card(name="cal")
    class _Prefixed(AgentEnvEnvironment):
        @tool(name="{environment_name}_deliver", hidden_by_default=True)
        def deliver(self, reminder_id: str) -> str:
            return "delivered"

        @tool(name="{environment_name}_schedule",
              defers=Defers(call="{environment_name}_deliver", report_as="{environment_name}_schedule", using="_at"))
        def schedule(self, event_id: str) -> str:
            return "scheduled"

        def _at(self, arguments: dict, result) -> DeferredCall:
            return DeferredCall(key=arguments["event_id"], at=AT, arguments={"reminder_id": arguments["event_id"]})

    _text, _structured, meta = await _called(_Prefixed(), "mcp", "cal_schedule", {"event_id": "e4"})
    assert meta[DEFERRED_CALLS_META_KEY] == [{"key": "e4", "at": AT,
                                              "call": {"tool": "cal_deliver", "arguments": {"reminder_id": "e4"}},
                                              "report_as": {"tool": "cal_schedule", "arguments": {}}}]


@pytest.mark.parametrize("defers, error", [
    (Defers(call="deliver_later", report_as="send_notification", using="_delivery"),
     "'schedule_reminder' defers 'deliver_later', which is not a tool reminders serves"),
    (Defers(call="send_notification", report_as="send_notification", using="_delivery"),
     "'schedule_reminder' defers 'send_notification', which is not hidden_by_default"),
    (Defers(call="deliver_reminder", report_as="notify", using="_delivery"),
     "'schedule_reminder' reports its deferred calls as 'notify', which is not a tool reminders serves"),
    (Defers(call="deliver_reminder", report_as="send_notification", using="_missing"),
     "'schedule_reminder' builds its deferred calls with '_missing', which is not a method of _Bad"),
    (Defers(call="deliver_reminder", report_as="send_notification", using="label"),
     "'schedule_reminder' builds its deferred calls with 'label', which is not a method of _Bad"),
    (Defers(call="deliver_reminder", report_as="send_notification", using="_later"),
     "'schedule_reminder' builds its deferred calls with '_later', a coroutine; it must be a plain method"),
])
def test_a_defers_naming_what_the_environment_does_not_serve_is_refused_at_mount(defers, error):
    class _Bad(_Reminders):
        label = "reminders"

        @tool(defers=defers)
        def schedule_reminder(self, event_id: str, remind_at: str) -> str:
            return "scheduled"

        async def _later(self, arguments: dict, result) -> None:
            return None

    with pytest.raises(ValueError, match=error):
        _Bad().create_app()
    with pytest.raises(ValueError, match=error):
        _Bad().mount(fastmcp.FastMCP("reminders"))


def test_a_deferred_call_the_served_tools_leave_out_is_refused_at_mount():
    class _Unserved(_Reminders):
        def served_tools(self, tools: list[EnvironmentTool]) -> list[EnvironmentTool]:
            return [t for t in tools if t.name != "deliver_reminder"]

    with pytest.raises(ValueError, match="defers 'deliver_reminder', which is not a tool reminders serves"):
        _Unserved().create_app()


def test_defers_must_be_a_defers():
    with pytest.raises(TypeError, match="defers takes a Defers"):
        tool(defers={"call": "deliver_reminder"})
