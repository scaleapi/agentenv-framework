"""ToolSession against a real AgentEnv FastMCP server in each streamable-HTTP mode, and against a scripted server for
the wire cases a real one rarely produces: paged lists, busy event streams, lost sessions and broken replies."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Annotated

import httpx
import pytest
from pydantic import Field

from agentenv_protocol import (
    ROLE_HEADER,
    EnvironmentCard,
    EnvironmentTool,
    ToolResult,
    ToolSession,
    ToolSessionError,
    create_fastmcp_app,
    tool,
    tool_definitions,
)

pytest.importorskip("mcp.server.fastmcp")

_URL = "http://env/mcp"


class _Items:
    def __init__(self) -> None:
        self.items: list[str] = []

    @tool(name="{environment_name}_add_item")
    async def add_item(self, item: Annotated[str, Field(description="The item to add.")], times: int = 1) -> str:
        """Add an item to the store."""
        self.items.extend([item] * times)
        return json.dumps({"count": len(self.items)})

    @tool()
    async def list_items(self) -> str:
        """List the stored items."""
        return json.dumps({"items": self.items})

    @tool()
    async def echo_after(self, value: str, seconds: float) -> str:
        """Answer with value after a pause."""
        await asyncio.sleep(seconds)
        return value

    @tool()
    def explode(self) -> str:
        """Always fails."""
        raise RuntimeError("the tool blew up")


def _route_clients_to(monkeypatch, transport: httpx.AsyncBaseTransport) -> None:
    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: real(transport=transport, **kwargs))


@pytest.fixture
def serve_items(monkeypatch):
    """Serve an _Items env with create_fastmcp_app in the given mode, in process; yields the handler."""
    async def serve(**fastmcp_kwargs):
        handler = _Items()
        app = create_fastmcp_app(handler, card=EnvironmentCard(name="items"), **fastmcp_kwargs).streamable_http_app()
        _route_clients_to(monkeypatch, httpx.ASGITransport(app=app))
        return handler, app.router.lifespan_context(app)
    return serve


class _Scripted:
    """A streamable-HTTP MCP server whose answer to each method a test sets, recording every request it receives."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.sessions = {"s-1"}
        self.answers: dict = {}

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        session = request.headers.get("mcp-session-id")
        if request.method == "DELETE":
            self.sessions.discard(session)
            return httpx.Response(200)
        message = json.loads(request.content)
        if session is not None and session not in self.sessions:
            return httpx.Response(404, json={"jsonrpc": "2.0", "id": "server-error",
                                             "error": {"code": -32600, "message": "Session not found"}})
        if "id" not in message:
            return httpx.Response(202)
        answer = self.answers.get(message["method"])
        if callable(answer):
            return answer(message)
        if message["method"] == "initialize":
            return httpx.Response(200, headers={"mcp-session-id": "s-1"}, json=_reply(message, {
                "protocolVersion": "2025-06-18", "capabilities": {"tools": {}}, "serverInfo": {"name": "s", "version": "1"}}))
        return httpx.Response(200, json=_reply(message, answer if answer is not None else {}))

    def bodies(self, method: str) -> list[dict]:
        return [json.loads(r.content) for r in self.requests if r.content and json.loads(r.content)["method"] == method]


def _reply(message: dict, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": message["id"], "result": result}


def _events(*blocks: str) -> httpx.Response:
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content="".join(blocks).encode())


@pytest.fixture
def scripted(monkeypatch) -> _Scripted:
    server = _Scripted()
    _route_clients_to(monkeypatch, httpx.MockTransport(server))
    return server


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [{}, {"json_response": True}, {"stateless_http": True}],
                         ids=["event-stream", "json", "stateless"])
async def test_an_agentenv_servers_tools_are_listed_and_called_in_each_streamable_http_mode(serve_items, mode):
    handler, lifespan = await serve_items(**mode)
    async with lifespan, ToolSession(_URL) as session:
        tools = {t.name: t for t in await session.list_tools()}
        added = await session.call_tool("items_add_item", {"item": "crate", "times": 2})
        listed = await session.call_tool("list_items")

    assert {"items_add_item", "list_items", "echo_after", "explode"} <= set(tools)
    assert tools["items_add_item"].description == "Add an item to the store."
    assert tools["items_add_item"].inputSchema["properties"]["item"]["description"] == "The item to add."
    assert (added.isError, json.loads(added.text)) == (False, {"count": 2})
    assert json.loads(listed.text) == {"items": ["crate", "crate"]} and handler.items == ["crate", "crate"]
    assert (session.session_id is None) == ("stateless_http" in mode)


@pytest.mark.asyncio
async def test_a_failing_or_unknown_tool_is_an_error_result_not_an_exception(serve_items):
    _, lifespan = await serve_items()
    async with lifespan, ToolSession(_URL) as session:
        failed = await session.call_tool("explode")
        unknown = await session.call_tool("no_such_tool")
        after = await session.call_tool("list_items")

    assert failed.isError and "the tool blew up" in failed.text
    assert unknown.isError and "no_such_tool" in unknown.text
    assert not after.isError


@pytest.mark.asyncio
async def test_concurrent_calls_on_one_session_each_get_their_own_reply(serve_items):
    _, lifespan = await serve_items()
    async with lifespan, ToolSession(_URL) as session:
        results = await asyncio.gather(*(session.call_tool("echo_after", {"value": f"v{i}", "seconds": (i % 5) / 50})
                                         for i in range(25)))

    assert [r.text for r in results] == [f"v{i}" for i in range(25)]


@pytest.mark.asyncio
async def test_a_request_that_outlives_the_timeout_raises_timeout_error_and_the_session_carries_on(serve_items):
    _, lifespan = await serve_items()
    async with lifespan, ToolSession(_URL, timeout=0.3) as session:
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="tools/call did not finish within 0.3s"):
            await session.call_tool("echo_after", {"value": "late", "seconds": 5})
        waited = time.monotonic() - started
        assert (await session.call_tool("echo_after", {"value": "on time", "seconds": 0})).text == "on time"

    assert waited < 2


def test_text_joins_the_text_blocks_and_skips_the_rest():
    result = ToolResult(content=[{"type": "text", "text": "first"}, {"type": "image", "data": "...", "mimeType": "image/png"},
                                 {"type": "text", "text": "second"}])
    assert result.text == "first\nsecond" and ToolResult().text == ""


def test_tool_definitions_declare_each_tool_for_each_model_api():
    schema = {"type": "object", "properties": {"item": {"type": "string"}}, "required": ["item"]}
    tools = [EnvironmentTool(name="add", description="Add one.", inputSchema=schema), EnvironmentTool(name="bare")]

    chat, responses, anthropic = (tool_definitions(tools, api) for api in ("openai_chat", "openai_responses", "anthropic"))

    assert chat[0] == {"type": "function", "function": {"name": "add", "description": "Add one.", "parameters": schema}}
    assert responses[0] == {"type": "function", "name": "add", "description": "Add one.", "parameters": schema}
    assert anthropic[0] == {"name": "add", "description": "Add one.", "input_schema": schema}
    assert anthropic[1] == {"name": "bare", "description": "", "input_schema": {"type": "object", "properties": {}}}
    chat[0]["function"]["parameters"]["additionalProperties"] = False
    assert "additionalProperties" not in tools[0].inputSchema
    with pytest.raises(ValueError, match="api must be one of anthropic, openai_chat, openai_responses"):
        tool_definitions(tools, "gemini")


@pytest.mark.asyncio
async def test_the_session_sends_its_role_and_headers_echoes_the_session_and_ends_it_on_close(scripted):
    async with ToolSession(_URL, role="cli", headers={"X-Trace": "t-1"}) as session:
        await session.list_tools()

    initialize, initialized, listing, delete = scripted.requests
    assert [r.method for r in scripted.requests] == ["POST", "POST", "POST", "DELETE"]
    assert all(r.headers[ROLE_HEADER] == "cli" and r.headers["x-trace"] == "t-1" for r in scripted.requests)
    assert initialize.headers["accept"] == "application/json, text/event-stream"
    assert "mcp-session-id" not in initialize.headers and "mcp-protocol-version" not in initialize.headers
    assert json.loads(initialize.content)["params"]["protocolVersion"] == "2025-11-25"
    assert json.loads(initialized.content) == {"jsonrpc": "2.0", "method": "notifications/initialized"}
    for later in (initialized, listing, delete):
        assert (later.headers["mcp-session-id"], later.headers["mcp-protocol-version"]) == ("s-1", "2025-06-18")
    assert scripted.sessions == set()


@pytest.mark.asyncio
async def test_a_closed_session_opens_again_as_a_new_session(scripted):
    session = ToolSession(_URL)
    async with session:
        pass
    scripted.sessions.add("s-1")
    async with session:
        await session.list_tools()

    second_initialize = [r for r in scripted.requests if r.content and json.loads(r.content)["method"] == "initialize"][1]
    assert "mcp-session-id" not in second_initialize.headers and session.session_id == "s-1"


@pytest.mark.asyncio
async def test_tools_list_follows_cursors_until_the_last_page(scripted):
    pages = {None: (["a", "b"], "p2"), "p2": (["c"], "p3"), "p3": (["d"], None)}

    def page(message):
        names, cursor = pages[message["params"].get("cursor")]
        return httpx.Response(200, json=_reply(message, {"tools": [{"name": n} for n in names],
                                                         **({"nextCursor": cursor} if cursor else {})}))
    scripted.answers["tools/list"] = page

    async with ToolSession(_URL) as session:
        assert [t.name for t in await session.list_tools()] == ["a", "b", "c", "d"]
    assert [b["params"] for b in scripted.bodies("tools/list")] == [{}, {"cursor": "p2"}, {"cursor": "p3"}]


@pytest.mark.asyncio
async def test_an_event_stream_reply_is_found_among_comments_priming_events_notifications_and_other_ids(scripted):
    def stream(message):
        reply = json.dumps(_reply(message, {"content": [{"type": "text", "text": "found"}]}), indent=1)
        return _events(
            ": keep-alive\n\n",
            "id: prime-1\ndata:\n\n",
            'event: message\ndata: {"jsonrpc": "2.0", "method": "notifications/progress", "params": {}}\n\n',
            f'data: {json.dumps(_reply({"id": 999}, {"content": []}))}\n\n',
            "event: ping\ndata: {}\n\n",
            "".join(f"data: {line}\n" for line in reply.splitlines()) + "\n",
        )
    scripted.answers["tools/call"] = stream

    async with ToolSession(_URL) as session:
        assert (await session.call_tool("t")).text == "found"


@pytest.mark.asyncio
async def test_a_session_the_server_dropped_raises_and_says_the_state_may_be_gone(scripted):
    async with ToolSession(_URL) as session:
        scripted.sessions.clear()
        with pytest.raises(ToolSessionError, match="no longer knows session s-1.*state may be gone"):
            await session.call_tool("t")


@pytest.mark.asyncio
@pytest.mark.parametrize("answer, match", [
    (lambda m: httpx.Response(200, json={"jsonrpc": "2.0", "id": m["id"], "error": {"code": -32602, "message": "bad args"}}),
     r"tools/call failed: bad args \(code -32602\)"),
    (lambda m: httpx.Response(500, text="upstream exploded"), "tools/call: HTTP 500 from http://env/mcp: upstream exploded"),
    (lambda m: httpx.Response(200, headers={"content-type": "application/json"}, content=b"{not json"),
     "tools/call: unreadable reply"),
    (lambda m: _events("data: {broken\n\n"), "tools/call: unreadable reply"),
    (lambda m: _events('data: {"jsonrpc": "2.0", "method": "notifications/message"}\n\n'),
     "tools/call: the event stream ended before the reply"),
    (lambda m: _events(f"data: {json.dumps(_reply(m, {}))}"), "tools/call: the event stream ended before the reply"),
    (lambda m: httpx.Response(200, headers={"content-type": "text/plain"}, content=b"hi"),
     "tools/call: unexpected content type 'text/plain'"),
    (lambda m: httpx.Response(200, json=[_reply(m, {})]), "tools/call: the reply is not a JSON-RPC object"),
    (lambda m: httpx.Response(200, json={"jsonrpc": "2.0", "id": m["id"], "result": "ok"}),
     "tools/call: the reply has no result object"),
    (lambda m: httpx.Response(200, json=_reply(m, {"content": "not a list"})), "tools/call: unreadable result"),
], ids=["jsonrpc-error", "http-500", "bad-json", "bad-event-json", "no-reply-in-stream", "unterminated-event",
        "wrong-content-type", "batch-reply", "result-not-object", "result-wrong-shape"])
async def test_a_failure_at_the_mcp_level_raises_a_tool_session_error(scripted, answer, match):
    scripted.answers["tools/call"] = answer
    async with ToolSession(_URL) as session:
        with pytest.raises(ToolSessionError, match=match):
            await session.call_tool("t")


@pytest.mark.asyncio
async def test_a_listed_tool_without_a_name_raises_a_tool_session_error(scripted):
    scripted.answers["tools/list"] = {"tools": [{"description": "nameless"}]}
    async with ToolSession(_URL) as session:
        with pytest.raises(ToolSessionError, match="tools/list: unreadable result"):
            await session.list_tools()


@pytest.mark.asyncio
async def test_a_failed_initialize_raises_and_leaves_the_session_closed(scripted):
    scripted.answers["initialize"] = lambda m: httpx.Response(503, text="starting up")
    session = ToolSession(_URL)
    with pytest.raises(ToolSessionError, match="initialize: HTTP 503"):
        await session.open()
    with pytest.raises(RuntimeError, match="the ToolSession is not open"):
        await session.list_tools()


@pytest.mark.asyncio
async def test_closing_after_the_server_is_gone_does_not_raise(monkeypatch):
    async def gone_on_delete(request: httpx.Request) -> httpx.Response:
        if request.method == "DELETE":
            raise httpx.ConnectError("connection refused", request=request)
        message = json.loads(request.content)
        if "id" not in message:
            return httpx.Response(202)
        return httpx.Response(200, headers={"mcp-session-id": "s-9"}, json=_reply(message, {"protocolVersion": "2025-11-25"}))
    _route_clients_to(monkeypatch, httpx.MockTransport(gone_on_delete))

    async with ToolSession(_URL) as session:
        assert session.session_id == "s-9"
    await session.close()


@pytest.mark.asyncio
async def test_closing_waits_for_the_server_no_longer_than_the_sessions_timeout(monkeypatch):
    async def stalls_on_delete(request: httpx.Request) -> httpx.Response:
        if request.method == "DELETE":
            await asyncio.sleep(30)
        message = json.loads(request.content or b"{}")
        if "id" not in message:
            return httpx.Response(202)
        return httpx.Response(200, headers={"mcp-session-id": "s-2"}, json=_reply(message, {"protocolVersion": "2025-11-25"}))
    _route_clients_to(monkeypatch, httpx.MockTransport(stalls_on_delete))

    session = await ToolSession(_URL, timeout=0.3).open()
    started = time.monotonic()
    await session.close()
    assert time.monotonic() - started < 1
