"""One process-wide generation of child MCP sessions: opened on first need by one owner task (the client contexts are
anyio task groups, exited only by the task that entered them), shared by the MCP proxy, /step and the trigger engine,
reopened once when it turns out dead (the call retried only if it never reached the child) and never re-discovered;
and the process lifespan that brackets it."""
from __future__ import annotations

import asyncio
import contextlib
from types import SimpleNamespace

import anyio
import pytest
from mcp.shared.exceptions import McpError
from mcp.types import CONNECTION_CLOSED, CallToolResult, ErrorData, TextContent, Tool as MCPTool, ToolAnnotations

from agent_env.env.gateway import InternalMCPServer
from agent_env.env.gateway import gateway as gateway_module
from agent_env.env.gateway.gateway import Gateway

URL = "http://backing/mcp"
TOOLS = [MCPTool(name="svc_thing", description="", inputSchema={"type": "object", "properties": {}},
                 annotations=ToolAnnotations(readOnlyHint=True))]


def _result() -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text="ok")], isError=False)


class _FakeSession:
    def __init__(self):
        self.calls: list[str] = []
        self.failures: list[BaseException] = []  # raised by call_tool, in order
        self.list_failures: list[BaseException] = []  # raised by list_tools, in order

    async def list_tools(self):
        if self.list_failures:
            raise self.list_failures.pop(0)
        return SimpleNamespace(tools=TOOLS)

    async def call_tool(self, name, arguments, *args, meta=None, **kwargs):
        self.calls.append(name)
        if self.failures:
            raise self.failures.pop(0)
        return _result()


class _Opener:
    """Stands in for _open_persistent_sessions: counts attempts and generations, records the entering/exiting task."""

    def __init__(self, session: _FakeSession, *, delay: float = 0.0, failures=(), gate: asyncio.Event | None = None):
        self.session, self.delay, self.failures, self.gate = session, delay, list(failures), gate
        self.attempts = 0
        self.opened = 0
        self.entered: list[asyncio.Task] = []
        self.exited: list[asyncio.Task] = []

    async def __call__(self, stack):
        self.attempts += 1
        await asyncio.sleep(self.delay)
        if self.gate is not None:
            await self.gate.wait()
        if self.failures:
            raise self.failures.pop(0)
        self.opened += 1
        self.entered.append(asyncio.current_task())

        async def exited():
            self.exited.append(asyncio.current_task())

        stack.push_async_callback(exited)
        return {URL: self.session}


def _gateway(opener: _Opener, **kw) -> Gateway:
    gw = Gateway(host="127.0.0.1", port=0, server_name="t", internal_mcp_servers=[InternalMCPServer("svc", URL)], **kw)
    gw._open_persistent_sessions = opener  # type: ignore[method-assign]
    return gw


def _explode(*a, **kw):
    raise AssertionError("a completed pass must neither reopen nor rediscover")


async def _run_lifespan(app, release: asyncio.Event, sent: list[dict]) -> None:
    """Drive one ASGI lifespan: startup now, shutdown once `release` is set; what the app sends lands in `sent`."""
    scope = {"type": "lifespan", "asgi": {"version": "3.0", "spec_version": "2.0"}, "state": {}}
    pending = [{"type": "lifespan.startup"}]

    async def receive():
        if pending:
            return pending.pop()
        await release.wait()
        return {"type": "lifespan.shutdown"}

    async def send(message):
        sent.append(message)

    await app(scope, receive, send)


async def _startup_answered(sent: list[dict]) -> None:
    for _ in range(500):
        if sent:
            return
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_concurrent_first_use_opens_one_generation():
    opener = _Opener(_FakeSession(), delay=0.02)
    gw = _gateway(opener)
    try:
        results = await asyncio.gather(*(gw._ensure_child_sessions() for _ in range(5)))
        assert all(r is results[0] for r in results) and opener.opened == 1
        assert gw._child_sessions is results[0] and not gw._child_owner.done()
    finally:
        await gw._close_child_sessions()


@pytest.mark.asyncio
async def test_the_generation_is_exited_by_the_task_that_entered_it():
    opener = _Opener(_FakeSession())
    gw = _gateway(opener)
    await gw._ensure_child_sessions()
    owner = gw._child_owner
    await gw._close_child_sessions()
    assert opener.entered == [owner] and opener.exited == [owner] and owner is not asyncio.current_task()
    assert owner.done() and gw._child_sessions == {} and gw._child_owner is None and gw._child_close is None


@pytest.mark.parametrize("death", [
    anyio.ClosedResourceError(),
    anyio.BrokenResourceError(),
    McpError(ErrorData(code=32600, message="Session terminated")),
], ids=["closed", "broken", "session_terminated"])
@pytest.mark.asyncio
async def test_a_dead_session_is_reopened_once_and_the_call_retried_without_rediscovery(death):
    session = _FakeSession()
    session.failures = [death]
    opener = _Opener(session)
    gw = _gateway(opener)
    try:
        await gw._ensure_tools_discovered()
        first_owner, proxy = gw._child_owner, gw._mcp._tool_manager._tools["svc_thing"]
        result = await gw._call_child_tool(URL, "svc_thing", {})
        assert not result.isError and session.calls == ["svc_thing", "svc_thing"]
        assert opener.opened == 2 and first_owner.done()
        assert gw._child_owner is not first_owner and not gw._child_owner.done()
        assert gw._tools_discovered and gw._mcp._tool_manager._tools["svc_thing"] is proxy
    finally:
        await gw._close_child_sessions()


@pytest.mark.asyncio
async def test_a_call_lost_in_flight_retires_the_generation_but_is_not_retried():
    session = _FakeSession()
    session.failures = [McpError(ErrorData(code=CONNECTION_CLOSED, message="Connection closed"))]
    opener = _Opener(session)
    gw = _gateway(opener)
    try:
        await gw._ensure_child_sessions()
        first_owner = gw._child_owner
        with pytest.raises(McpError, match="Connection closed"):
            await gw._call_child_tool(URL, "svc_thing", {})
        assert session.calls == ["svc_thing"]  # the child may already have written; a retry could write twice
        assert first_owner.done() and gw._child_owner is None and opener.opened == 1
        assert not (await gw._call_child_tool(URL, "svc_thing", {})).isError  # the next call reopens
        assert opener.opened == 2 and session.calls == ["svc_thing", "svc_thing"]
    finally:
        await gw._close_child_sessions()


@pytest.mark.asyncio
async def test_any_other_error_is_the_caller_s_and_keeps_the_generation():
    session = _FakeSession()
    session.failures = [McpError(ErrorData(code=-32602, message="Invalid params"))]
    opener = _Opener(session)
    gw = _gateway(opener)
    try:
        with pytest.raises(McpError, match="Invalid params"):
            await gw._call_child_tool(URL, "svc_thing", {})
        assert session.calls == ["svc_thing"] and opener.opened == 1 and not gw._child_owner.done()
    finally:
        await gw._close_child_sessions()


@pytest.mark.asyncio
async def test_concurrent_dead_session_errors_reset_the_generation_once():
    session = _FakeSession()
    opener = _Opener(session)
    gw = _gateway(opener)

    async def call_tool(name, arguments, *args, meta=None, **kwargs):  # every call on generation 1 dies
        session.calls.append(name)
        if opener.opened == 1:
            raise anyio.ClosedResourceError()
        return _result()

    session.call_tool = call_tool
    try:
        results = await asyncio.gather(gw._call_child_tool(URL, "a", {}), gw._call_child_tool(URL, "b", {}))
        assert all(not r.isError for r in results) and opener.opened == 2
    finally:
        await gw._close_child_sessions()


@pytest.mark.asyncio
async def test_a_failed_open_leaves_no_generation_and_is_retried_on_the_next_use():
    opener = _Opener(_FakeSession(), failures=[RuntimeError("refused")])
    gw = _gateway(opener)
    with pytest.raises(RuntimeError, match="refused"):
        await gw._ensure_child_sessions()
    assert gw._child_owner is None and gw._child_sessions == {}
    try:
        assert (await gw._ensure_child_sessions())[URL] is opener.session
        assert opener.attempts == 2 and opener.opened == 1
    finally:
        await gw._close_child_sessions()


@pytest.mark.asyncio
async def test_a_failed_discovery_keeps_the_gateway_s_own_entry_and_heals_on_the_next_attempt():
    session = _FakeSession()
    session.list_failures = [RuntimeError("boom")]
    opener = _Opener(session)
    gw = _gateway(opener, website_urls={"web": "http://web"})
    engine = gw._trigger_engine
    # registered at construction, not per MCP session
    assert [t.name for t in gw._server_tools["gateway"]] == ["list_website_urls"]
    assert "list_website_urls" in gw._mcp._tool_manager._tools
    assert engine._is_readonly("svc_thing") is False  # a memo built before the backing tools are known
    try:
        with pytest.raises(RuntimeError, match="boom"):
            await gw._ensure_tools_discovered()
        assert gw._tools_discovered is False and "svc" not in gw._server_tools
        assert [t.name for t in gw._server_tools["gateway"]] == ["list_website_urls"]
        assert not gw._child_owner.done()  # a failed tools/list is not a dead transport

        await gw._ensure_tools_discovered()
        assert gw._tools_discovered and [t.name for t in gw._server_tools["svc"]] == ["svc_thing"]
        assert engine._is_readonly("svc_thing") is True  # discovery dropped the early memo
        assert opener.opened == 1
        gw._open_persistent_sessions = _explode  # type: ignore[method-assign]
        await gw._ensure_tools_discovered()
    finally:
        await gw._close_child_sessions()


@pytest.mark.asyncio
async def test_discovery_on_a_dead_session_retires_the_generation():
    session = _FakeSession()
    session.list_failures = [McpError(ErrorData(code=32600, message="Session terminated"))]
    opener = _Opener(session)
    gw = _gateway(opener)
    try:
        with pytest.raises(McpError):
            await gw._ensure_tools_discovered()
        assert gw._child_owner is None
        await gw._ensure_tools_discovered()
        assert gw._tools_discovered and opener.opened == 2
    finally:
        await gw._close_child_sessions()


@pytest.mark.asyncio
async def test_a_child_that_never_answers_initialize_is_a_failed_connect_attempt(monkeypatch):
    class _HungSession:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

        async def initialize(self):
            await asyncio.Event().wait()

    @contextlib.asynccontextmanager
    async def streams(url):
        yield None, None, None

    monkeypatch.setattr(gateway_module, "streamable_http_client", streams)
    monkeypatch.setattr(gateway_module, "ClientSession", _HungSession)
    monkeypatch.setattr(gateway_module, "CARD_FETCH_TIMEOUT_S", 0.02)
    gw = Gateway(host="127.0.0.1", port=0, server_name="t", internal_mcp_servers=[InternalMCPServer("svc", URL)])
    async with contextlib.AsyncExitStack() as stack:
        with pytest.raises(TimeoutError):
            await gw._open_persistent_sessions(stack, max_retries=2, retry_delay=0)


@pytest.mark.asyncio
async def test_the_process_lifespan_starts_the_driver_tolerates_a_failed_startup_discovery_and_unwinds():
    opener = _Opener(_FakeSession(), failures=[RuntimeError("refused")])
    gw = _gateway(opener)
    app = gw._asgi_app()
    release, sent = asyncio.Event(), []
    task = asyncio.create_task(_run_lifespan(app, release, sent))
    await _startup_answered(sent)
    assert sent == [{"type": "lifespan.startup.complete"}]  # fail-open: a raised startup would exit uvicorn
    driver = gw._trigger_engine._driver_task
    assert driver is not None and not driver.done()
    assert gw._child_owner is None and opener.attempts == 1  # the failed open left nothing behind
    await gw._ensure_tools_discovered()  # the next use heals it
    assert gw._tools_discovered and not gw._child_owner.done()
    release.set()
    await asyncio.wait_for(task, 5)
    assert sent[-1] == {"type": "lifespan.shutdown.complete"}
    assert gw._trigger_engine._driver_task is None and gw._child_owner is None and gw._child_sessions == {}

    # the MCP session manager runs once per app, so a second lifespan fails its startup; the process state still unwinds
    sent2: list[dict] = []
    with pytest.raises(RuntimeError, match="only be called once"):
        await _run_lifespan(app, asyncio.Event(), sent2)
    assert sent2[0]["type"] == "lifespan.startup.failed"
    assert gw._trigger_engine._driver_task is None and gw._child_owner is None


@pytest.mark.asyncio
async def test_startup_discovery_is_bounded_and_a_hung_open_leaves_no_generation(monkeypatch):
    monkeypatch.setattr(Gateway, "STARTUP_DISCOVERY_TIMEOUT_S", 0.05)
    gate = asyncio.Event()
    opener = _Opener(_FakeSession(), gate=gate)  # the open hangs until released
    gw = _gateway(opener)
    app = gw._asgi_app()
    release, sent = asyncio.Event(), []
    task = asyncio.create_task(_run_lifespan(app, release, sent))
    await _startup_answered(sent)
    assert sent == [{"type": "lifespan.startup.complete"}] and not gw._trigger_engine._driver_task.done()
    assert gw._child_owner is None and gw._child_sessions == {} and not gw._tools_discovered
    assert not gw._child_lock.locked() and not gw._discover_lock.locked()
    gate.set()  # the hung open now returns to an owner nobody waits on; it exits what it opened
    try:
        assert (await gw._ensure_child_sessions())[URL] is opener.session
        await asyncio.sleep(0.01)
        assert opener.attempts == 2 and opener.opened == 2 and opener.exited == [opener.entered[0]]
        assert opener.entered[1] is gw._child_owner and not gw._child_owner.done()
        await gw._ensure_tools_discovered()
        assert gw._tools_discovered
    finally:
        release.set()
        await asyncio.wait_for(task, 5)
    assert sent[-1] == {"type": "lifespan.shutdown.complete"} and gw._child_owner is None


@pytest.mark.asyncio
async def test_shutdown_cancels_scheduled_actions_and_nothing_reopens_the_child_sessions():
    session = _FakeSession()
    opener = _Opener(session)
    gw = _gateway(opener)
    app = gw._asgi_app()
    release, sent = asyncio.Event(), []
    task = asyncio.create_task(_run_lifespan(app, release, sent))
    await _startup_answered(sent)
    assert sent == [{"type": "lifespan.startup.complete"}] and not gw._child_owner.done()
    outcome: list[str] = []

    async def late_action():  # a trigger action still waiting when the process shuts down
        try:
            await asyncio.sleep(60)
            await gw._call_child_tool(URL, "svc_thing", {})
        except asyncio.CancelledError:
            outcome.append("cancelled")
            raise

    scheduled = gw._trigger_engine._schedule(late_action())
    await asyncio.sleep(0)
    release.set()
    await asyncio.wait_for(task, 5)
    assert sent[-1] == {"type": "lifespan.shutdown.complete"}
    assert scheduled.cancelled() and outcome == ["cancelled"] and gw._trigger_engine._tasks == set()
    assert session.calls == [] and opener.opened == 1 and opener.exited == opener.entered
    assert gw._child_owner is None and gw._child_sessions == {}
    with pytest.raises(RuntimeError, match="shutting down"):
        await gw._ensure_child_sessions()
    assert gw._child_owner is None and opener.attempts == 1
