"""Who changed what (urn:agentenv:changes/v1): GET /changes joins changelog rows onto the gateway's write windows."""
from __future__ import annotations

import asyncio
import contextvars
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from urllib.parse import unquote, urlsplit

import httpx
import pytest
from mcp.types import CallToolResult, TextContent, Tool as MCPTool

from agent_env.env.gateway import InternalMCPServer
from agent_env.env.gateway import rest_proxy as rest_proxy_module
from agent_env.env.gateway.caller import Caller
from agent_env.env.gateway.changes import ChangeAttribution
from agent_env.env.gateway.gateway import GATEWAY_TRAJECTORY_FILE, Gateway

URL = "http://backing/mcp"
TOOL = MCPTool(name="mail_send_message", description="", inputSchema={"type": "object", "properties": {}})
_RealAsyncClient = httpx.AsyncClient


def _call(event_id: str, tool: str, role: str, **extra) -> dict:
    return {"event_type": "tool_call", "event_id": event_id, "tool_call": {"function_name": tool, "arguments": {}},
            "role": role, **extra}


def _result(call_id: str, before, after) -> dict:
    return {"event_type": "tool_call_result", "event_id": f"{call_id}-r", "tool_call_event_id": call_id,
            "changelog_id_before": before, "changelog_id": after}


# ---------------------------------------------------------------- the join

def test_a_row_is_attributed_to_the_call_whose_window_holds_it():
    changes = ChangeAttribution()
    fire = {"trigger_id": "reply", "action_index": 0, "fire": 1}
    for event in (
        _call("event_1", "mail_send_message", "default"), _result("event_1", 0, 2),
        _call("event_3", "mail_send_message", "dana@example.com", caused_by=[fire]), _result("event_3", 2, 3),
        {"event_type": "internal_tool_call", "event_id": "event_5", "trigger_id": "mirror", "action_index": 1,
         "tool": "drive_touch_file", "as": None, "changelog_id_before": 3, "changelog_id_after": 4},
        {"event_type": "http_request", "event_id": "event_6", "route": "website", "service": "mail", "method": "POST",
         "path": "/api/drafts", "status": 201, "role": "default", "changelog_id_before": 4, "changelog_id_after": 5},
    ):
        changes.observe(event)

    assert changes.actors([1, 2, 3, 4, 5, 6]) == [
        [{"kind": "tool_call", "event_id": "event_1", "tool": "mail_send_message", "role": "default"}],
        [{"kind": "tool_call", "event_id": "event_1", "tool": "mail_send_message", "role": "default"}],
        [{"kind": "tool_call", "event_id": "event_3", "tool": "mail_send_message", "role": "dana@example.com",
          "caused_by": [fire]}],
        [{"kind": "trigger_action", "event_id": "event_5", "trigger_id": "mirror", "action_index": 1,
          "tool": "drive_touch_file", "role": None}],
        [{"kind": "request", "event_id": "event_6", "role": "default", "request": {
            "route": "website", "service": "mail", "method": "POST", "path": "/api/drafts", "status": 201}}],
        [],
    ]


def test_overlapping_calls_both_name_a_row_written_between_them():
    changes = ChangeAttribution()
    changes.observe(_call("event_1", "a", "default"))
    changes.observe(_call("event_2", "b", "carol@example.com"))
    changes.observe(_result("event_2", 0, 2))
    changes.observe(_result("event_1", 0, 3))
    assert [[a["event_id"] for a in who] for who in changes.actors([1, 2, 3])] == [
        ["event_2", "event_1"], ["event_2", "event_1"], ["event_1"]]


def test_calls_without_a_write_window_attribute_nothing():
    changes = ChangeAttribution()
    for event in (
        _call("event_1", "read_only", "default"), _result("event_1", 4, 4),
        _call("event_3", "no_changelog", "default"), _result("event_3", None, None),
        _call("event_5", "list_website_urls", "default"),
        {"event_type": "tool_call_result", "event_id": "event_6", "tool_call_event_id": "event_5", "tool_call_result": {}},
        _result("event_unknown", 0, 9),
        _call("event_7", "raised", "default"),
    ):
        changes.observe(event)
    assert changes.actors([1, 4, 5, 9]) == [[], [], [], []]


# ---------------------------------------------------------------- the gateway records the windows

class _Changelog:
    """The env's changelog as the gateway reads it: MAX(id) through `_query_changelog_id`, rows through a cursor."""

    def __init__(self):
        self.rows: list[dict] = []
        self.queries: list[tuple[str, tuple]] = []

    def write(self, table: str, row_id: str) -> None:
        self.rows.append({"id": len(self.rows) + 1, "timestamp": datetime(2026, 10, 14, 8, len(self.rows), tzinfo=timezone.utc),
                          "schema_name": "mail", "table_name": table, "operation": "INSERT", "row_id": row_id,
                          "summary": f"Inserted: {row_id}", "changed_fields": {"id": row_id}})

    def max_id(self) -> int | None:
        return len(self.rows) or None

    def cursor(self):
        log = self

        class _Cursor:
            description = [(name,) for name in ("id", "timestamp", "schema_name", "table_name", "operation",
                                                "row_id", "summary", "changed_fields")]

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def execute(self, sql, params):
                log.queries.append((sql, params))
                after_id, count = params
                self._rows = [r for r in log.rows if r["id"] > after_id][:count]

            def fetchall(self):
                return [tuple(r[name] for (name,) in self.description) for r in self._rows]

        return _Cursor()


class _WritingSession:
    """The child server: each call writes one changelog row, then raises `failure` when one is set."""

    def __init__(self, changelog: _Changelog, failure: Exception | None = None):
        self.changelog = changelog
        self.failure = failure

    async def list_tools(self):
        return SimpleNamespace(tools=[TOOL])

    async def call_tool(self, name, arguments, *args, meta=None, **kwargs):
        self.changelog.write("messages", f"m{len(self.changelog.rows) + 1}")
        if self.failure is not None:
            raise self.failure
        return CallToolResult(content=[TextContent(type="text", text="ok")], isError=False)


def _gateway(changelog: _Changelog, failure: Exception | None = None, **kw) -> Gateway:
    gw = Gateway(host="127.0.0.1", port=0, server_name="t", internal_mcp_servers=[InternalMCPServer("mail", URL)], **kw)
    session = _WritingSession(changelog, failure)

    async def _open(stack):
        return {URL: session}

    gw._open_persistent_sessions = _open  # type: ignore[method-assign]
    gw._query_changelog_id = changelog.max_id  # type: ignore[method-assign]
    gw._db_conn = changelog
    gw._create_and_register_proxy_tool(URL, TOOL)
    gw._tool_server_urls[TOOL.name] = URL
    return gw


def _trajectory() -> list[dict]:
    with open(GATEWAY_TRAJECTORY_FILE) as f:
        return [json.loads(line) for line in f]


async def _http(gw: Gateway, method: str, target: str, *, headers: dict | None = None, body: bytes = b"") -> tuple[int, bytes]:
    """One request through the gateway's ASGI app, its scope as uvicorn builds it."""
    url = urlsplit(target)
    scope = {
        "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"}, "http_version": "1.1",
        "method": method, "scheme": "http", "path": unquote(url.path), "raw_path": url.path.encode(),
        "query_string": url.query.encode(), "root_path": "", "server": ("gateway", 80), "client": ("c", 1),
        "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
    }
    done = asyncio.Event()
    received = False
    sent: list[dict] = []

    async def receive():
        nonlocal received
        if not received:
            received = True
            return {"type": "http.request", "body": body, "more_body": False}
        await done.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    try:
        await gw._mcp.streamable_http_app()(scope, receive, send)
    finally:
        done.set()
    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    return status, b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")


@pytest.mark.asyncio
async def test_tool_calls_record_the_changelog_ids_they_spanned_and_changes_names_their_role():
    changelog = _Changelog()
    gw = _gateway(changelog)
    before = len(_trajectory())
    try:
        resp = await gw._step_call_tool({"tool_name": TOOL.name, "arguments": {"to": "dana@example.com"}}, "default")
        assert resp.status_code == 200
        ctx = contextvars.copy_context()
        ctx.run(Caller(role="carol@example.com").bind)
        await ctx.run(asyncio.create_task, gw._mcp._tool_manager.call_tool(TOOL.name, {}))
    finally:
        await gw._close_child_sessions()

    results = [e for e in _trajectory()[before:] if e["event_type"] == "tool_call_result"]
    assert [(e["changelog_id_before"], e["changelog_id"]) for e in results] == [(None, 1), (1, 2)]
    status, body = await _http(gw, "GET", "/changes")
    changes = json.loads(body)
    assert status == 200 and changes["has_more"] is False
    assert [(c["id"], c["row_id"], [(a["kind"], a["tool"], a["role"]) for a in c["actors"]]) for c in changes["changes"]] == [
        (1, "m1", [("tool_call", TOOL.name, "default")]),
        (2, "m2", [("tool_call", TOOL.name, "carol@example.com")]),
    ]
    assert changes["changes"][0]["timestamp"] == "2026-10-14T08:00:00+00:00"
    assert changes["changes"][0]["changed_fields"] == {"id": "m1"}


def test_a_result_that_measured_no_window_claims_no_row():
    changes = ChangeAttribution()
    changes.observe(_call("event_1", "list_website_urls", "default"))
    changes.observe({"event_type": "tool_call_result", "event_id": "event_2", "tool_call_event_id": "event_1",
                     "changelog_id": 5})
    assert changes.actors([1, 5]) == [[], []]


@pytest.mark.asyncio
async def test_a_gateway_tool_called_through_step_after_a_write_claims_none_of_it():
    changelog = _Changelog()
    gw = _gateway(changelog, website_urls={"mail": "http://mail-frontend"})
    try:
        assert (await gw._step_call_tool({"tool_name": TOOL.name, "arguments": {}}, "default")).status_code == 200
        assert (await gw._step_call_tool({"tool_name": "list_website_urls", "arguments": {}},
                                         "carol@example.com")).status_code == 200
    finally:
        await gw._close_child_sessions()

    _, body = await _http(gw, "GET", "/changes")
    assert [[(a["tool"], a["role"]) for a in c["actors"]] for c in json.loads(body)["changes"]] == [
        [(TOOL.name, "default")]]


@pytest.mark.asyncio
async def test_a_call_that_writes_then_fails_still_names_its_write_and_keeps_its_error():
    changelog = _Changelog()
    gw = _gateway(changelog, failure=RuntimeError("lost the connection after committing"))
    try:
        resp = await gw._step_call_tool({"tool_name": TOOL.name, "arguments": {}}, "default")
        ctx = contextvars.copy_context()
        ctx.run(Caller(role="carol@example.com").bind)
        with pytest.raises(Exception, match="lost the connection after committing"):
            await ctx.run(asyncio.create_task, gw._mcp._tool_manager.call_tool(TOOL.name, {}))
    finally:
        await gw._close_child_sessions()

    assert resp.status_code == 500
    assert json.loads(resp.body)["error"] == "Tool execution failed (RuntimeError): lost the connection after committing"
    _, body = await _http(gw, "GET", "/changes")
    assert [[(a["role"], a.get("failed")) for a in c["actors"]] for c in json.loads(body)["changes"]] == [
        [("default", True)], [("carol@example.com", True)]]


@pytest.mark.asyncio
async def test_a_request_keeps_the_trigger_fire_it_started_under(monkeypatch):
    changelog = _Changelog()
    cause = {"trigger_id": "reply", "action_index": 0, "fire": 1}
    gw = _gateway(changelog, rest_proxy_urls={"mail": "http://mail-backend"})
    gw._trigger_engine._in_flight["bob@example.com"] = [cause]

    async def body():
        yield b"a"
        gw._trigger_engine._in_flight.pop("bob@example.com")
        changelog.write("drafts", "d1")
        yield b"b"

    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=body()))
    monkeypatch.setattr(rest_proxy_module.httpx, "AsyncClient", lambda **kw: _RealAsyncClient(transport=transport, **kw))
    before = len(_trajectory())

    await _http(gw, "POST", "/svc/mail/api/drafts", headers={"AgentEnv-Role": "bob@example.com"})

    [logged] = [e for e in _trajectory()[before:] if e["event_type"] == "http_request"]
    assert logged["caused_by"] == [cause]


@pytest.mark.asyncio
async def test_a_write_through_the_rest_proxy_is_a_logged_window_and_a_read_is_not(monkeypatch):
    changelog = _Changelog()

    async def body():
        yield b'{"ok": true}'

    def website(request: httpx.Request) -> httpx.Response:
        if request.method != "GET":
            changelog.write("drafts", "d1")
        return httpx.Response(201 if request.method == "POST" else 200, content=body())

    transport = httpx.MockTransport(website)
    monkeypatch.setattr(rest_proxy_module.httpx, "AsyncClient", lambda **kw: _RealAsyncClient(transport=transport, **kw))
    gw = _gateway(changelog, website_urls={"mail": "http://mail-frontend"}, rest_proxy_urls={"mail": "http://mail-backend"})
    gw._trigger_engine._in_flight["bob@example.com"] = [{"trigger_id": "reply", "action_index": 0, "fire": 2}]
    before = len(_trajectory())

    assert (await _http(gw, "GET", "/website/mail/inbox"))[0] == 200
    assert (await _http(gw, "POST", "/website/mail/api/drafts", headers={"AgentEnv-Role": "bob@example.com"}))[0] == 201
    assert (await _http(gw, "PATCH", "/svc/mail/api/drafts/d1"))[0] == 200

    requests = [e for e in _trajectory()[before:] if e["event_type"] == "http_request"]
    assert [{k: e[k] for k in ("route", "service", "method", "path", "status", "role", "changelog_id_before",
                               "changelog_id_after")} for e in requests] == [
        {"route": "website", "service": "mail", "method": "POST", "path": "/api/drafts", "status": 201,
         "role": "bob@example.com", "changelog_id_before": None, "changelog_id_after": 1},
        {"route": "svc", "service": "mail", "method": "PATCH", "path": "/api/drafts/d1", "status": 200,
         "role": "default", "changelog_id_before": 1, "changelog_id_after": 2},
    ]
    assert requests[0]["caused_by"] == [{"trigger_id": "reply", "action_index": 0, "fire": 2}]
    _, body_bytes = await _http(gw, "GET", "/changes")
    [first, second] = json.loads(body_bytes)["changes"]
    assert first["actors"] == [{"kind": "request", "event_id": requests[0]["event_id"], "role": "bob@example.com",
                                "caused_by": [{"trigger_id": "reply", "action_index": 0, "fire": 2}],
                                "request": {"route": "website", "service": "mail", "method": "POST",
                                            "path": "/api/drafts", "status": 201}}]
    assert second["actors"][0]["request"]["route"] == "svc"


@pytest.mark.asyncio
async def test_a_write_made_while_the_response_streams_is_inside_the_window(monkeypatch):
    """Headers first, then a write while the body streams (an MCP SSE answer): the window closes at the body's end."""
    changelog = _Changelog()

    async def body():
        yield b"event: message\n"
        changelog.write("messages", "m1")
        yield b'data: {"result": {}}\n\n'

    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=body()))
    monkeypatch.setattr(rest_proxy_module.httpx, "AsyncClient", lambda **kw: _RealAsyncClient(transport=transport, **kw))
    gw = _gateway(changelog, rest_proxy_urls={"mcp-mail": "http://mail:8000"})
    before = len(_trajectory())

    status, body_bytes = await _http(gw, "POST", "/svc/mcp-mail/mcp", headers={"AgentEnv-Role": "carol@example.com"},
                                     body=b'{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {}}')

    assert status == 200 and body_bytes.endswith(b'{"result": {}}\n\n')
    [logged] = [e for e in _trajectory()[before:] if e["event_type"] == "http_request"]
    assert (logged["path"], logged["changelog_id_before"], logged["changelog_id_after"]) == ("/mcp", None, 1)
    _, changes = await _http(gw, "GET", "/changes")
    assert [a["role"] for a in json.loads(changes)["changes"][0]["actors"]] == ["carol@example.com"]


@pytest.mark.asyncio
async def test_a_write_window_does_not_hold_the_stream_back(monkeypatch):
    changelog = _Changelog()
    gate = asyncio.Event()

    async def body():
        yield b"a" * 1024
        await gate.wait()
        changelog.write("messages", "m1")
        yield b"b" * 1024

    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=body()))
    monkeypatch.setattr(rest_proxy_module.httpx, "AsyncClient", lambda **kw: _RealAsyncClient(transport=transport, **kw))
    gw = _gateway(changelog, rest_proxy_urls={"mcp-mail": "http://mail:8000"})
    before = len(_trajectory())
    first_chunk = asyncio.Event()
    received = False

    async def receive():
        nonlocal received
        if not received:
            received = True
            return {"type": "http.request", "body": b"{}", "more_body": False}
        await asyncio.sleep(3600)

    async def send(message):
        if message["type"] == "http.response.body" and message.get("body"):
            first_chunk.set()

    scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"}, "http_version": "1.1", "method": "POST",
             "scheme": "http", "path": "/svc/mcp-mail/mcp", "raw_path": b"/svc/mcp-mail/mcp", "query_string": b"",
             "root_path": "", "server": ("g", 80), "client": ("c", 1), "headers": []}
    task = asyncio.create_task(gw._mcp.streamable_http_app()(scope, receive, send))
    await asyncio.wait_for(first_chunk.wait(), 5)
    assert not [e for e in _trajectory()[before:] if e["event_type"] == "http_request"]
    gate.set()
    await asyncio.wait_for(task, 5)
    [logged] = [e for e in _trajectory()[before:] if e["event_type"] == "http_request"]
    assert (logged["changelog_id_before"], logged["changelog_id_after"]) == (None, 1)


@pytest.mark.asyncio
async def test_a_mutating_request_that_wrote_nothing_is_not_logged(monkeypatch):
    changelog = _Changelog()
    changelog.write("messages", "m1")

    async def body():
        yield b'{"jsonrpc": "2.0", "id": 1, "result": {"parts": []}}'

    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=body()))
    monkeypatch.setattr(rest_proxy_module.httpx, "AsyncClient", lambda **kw: _RealAsyncClient(transport=transport, **kw))
    gw = _gateway(changelog, rest_proxy_urls={"mail": "http://mail-backend"})
    before = len(_trajectory())

    for method, path in (("POST", "/svc/mail/agentenv"), ("PUT", "/svc/mail/api/drafts/d1"), ("DELETE", "/svc/mail/x")):
        assert (await _http(gw, method, path, body=b'{"method": "data/get"}'))[0] == 200

    assert [e for e in _trajectory()[before:] if e["event_type"] == "http_request"] == []


@pytest.mark.asyncio
async def test_changes_pages_by_after_id_and_limit():
    changelog = _Changelog()
    for i in range(5):
        changelog.write("messages", f"m{i + 1}")
    gw = _gateway(changelog)

    status, body = await _http(gw, "GET", "/changes?after_id=1&limit=2")
    page = json.loads(body)
    assert status == 200 and [c["id"] for c in page["changes"]] == [2, 3] and page["has_more"] is True
    assert all(c["actors"] == [] for c in page["changes"])
    _, body = await _http(gw, "GET", "/changes?after_id=3&limit=2")
    assert [c["id"] for c in json.loads(body)["changes"]] == [4, 5] and json.loads(body)["has_more"] is False
    assert changelog.queries[-1][1] == (3, 3)


@pytest.mark.asyncio
@pytest.mark.parametrize("query,status,fragment", [
    ("?after_id=x", 400, "must be integers"),
    ("?after_id=-1", 400, "after_id must be >= 0"),
    ("?limit=0", 400, "limit between 1 and 10000"),
    ("?limit=10001", 400, "limit between 1 and 10000"),
])
async def test_changes_refuses_a_bad_page(query, status, fragment):
    got, body = await _http(_gateway(_Changelog()), "GET", f"/changes{query}")
    assert got == status and fragment in json.loads(body)["error"]


@pytest.mark.asyncio
async def test_an_env_without_a_changelog_says_so():
    gw = Gateway(host="127.0.0.1", port=0, server_name="t", internal_mcp_servers=[])
    status, body = await _http(gw, "GET", "/changes")
    assert status == 404 and json.loads(body)["error"] == "this env keeps no changelog"


def test_an_empty_changelog_still_reads_null():
    queries = []

    class _Conn:
        def cursor(self):
            class _Cursor:
                def __enter__(self):
                    return self

                def __exit__(self, *exc):
                    return False

                def execute(self, sql):
                    queries.append(sql)

                def fetchone(self):
                    return (None,)
            return _Cursor()

    gw = Gateway(host="127.0.0.1", port=0, server_name="t", internal_mcp_servers=[])
    gw._db_conn = _Conn()
    assert gw._query_changelog_id() is None
    assert queries == ["SELECT MAX(id) FROM public._changelog"]


def test_a_first_write_on_an_empty_changelog_still_has_a_window():
    changes = ChangeAttribution()
    changes.observe(_call("event_1", "mail_send_message", "default"))
    changes.observe(_result("event_1", None, 2))
    assert [[a["event_id"] for a in who] for who in changes.actors([1, 2, 3])] == [["event_1"], ["event_1"], []]
