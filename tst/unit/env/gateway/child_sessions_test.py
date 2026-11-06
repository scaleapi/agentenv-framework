"""Each child's MCP session: opened on first need by an owner task of its own (the client contexts are anyio task
groups, exited only by the task that entered them), shared by the MCP proxy, /step and the trigger engine, replaced
when the child rejects it (each call it never reached retried once on a fresh session) or retired when a call dies in
flight, never touching another child's, and never re-discovered; and the process lifespan that brackets them."""
from __future__ import annotations

import asyncio
import contextlib
from types import SimpleNamespace

import anyio
import httpx
import pytest
from mcp.shared.exceptions import McpError
from mcp.types import CONNECTION_CLOSED, CallToolResult, ErrorData, TextContent, Tool as MCPTool, ToolAnnotations

from agent_env.env.gateway import InternalMCPServer
from agent_env.env.gateway import child_sessions as child_sessions_module
from agent_env.env.gateway import gateway as gateway_module
from agent_env.env.gateway.gateway import Gateway

URL = "http://backing/mcp"
OTHER = "http://other/mcp"
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
    """Stands in for ChildSessions.open_session: counts attempts and sessions opened, records the entering/exiting task."""

    def __init__(self, session: _FakeSession, *, delay: float = 0.0, failures=(), gate: asyncio.Event | None = None):
        self.session, self.delay, self.failures, self.gate = session, delay, list(failures), gate
        self.attempts = 0
        self.opened = 0
        self.entered: list[asyncio.Task] = []
        self.exited: list[asyncio.Task] = []

    async def __call__(self, server, stack, report):
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
        return self.session


def _gateway(opener: _Opener, **kw) -> Gateway:
    gw = Gateway(host="127.0.0.1", port=0, server_name="t", internal_mcp_servers=[InternalMCPServer("svc", URL)], **kw)
    gw._sessions.open_session = opener  # type: ignore[method-assign]
    return gw


def _owner(gw: Gateway, url: str = URL) -> asyncio.Task | None:
    child = gw._sessions.current.get(url)
    return child.owner if child is not None else None


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
async def test_concurrent_first_use_opens_one_session():
    opener = _Opener(_FakeSession(), delay=0.02)
    gw = _gateway(opener)
    try:
        results = await asyncio.gather(*(gw._ensure_child_sessions() for _ in range(5)))
        assert all(r[URL] is opener.session for r in results) and opener.opened == 1
        assert gw._sessions.current[URL].session is opener.session and not _owner(gw).done()
    finally:
        await gw._close_child_sessions()


@pytest.mark.asyncio
async def test_a_session_is_exited_by_the_task_that_entered_it():
    opener = _Opener(_FakeSession())
    gw = _gateway(opener)
    await gw._ensure_child_sessions()
    owner = _owner(gw)
    await gw._close_child_sessions()
    assert opener.entered == [owner] and opener.exited == [owner] and owner is not asyncio.current_task()
    assert owner.done() and gw._sessions.current == {} and gw._sessions.draining == set()


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
        first_owner, proxy = _owner(gw), gw._mcp._tool_manager._tools["svc_thing"]
        result = await gw._call_child_tool(URL, "svc_thing", {})
        assert not result.isError and session.calls == ["svc_thing", "svc_thing"]
        assert opener.opened == 2 and first_owner.done()
        assert _owner(gw) is not first_owner and not _owner(gw).done()
        assert gw._tools_discovered and gw._mcp._tool_manager._tools["svc_thing"] is proxy
    finally:
        await gw._close_child_sessions()


@pytest.mark.asyncio
async def test_a_call_lost_in_flight_retires_the_session_but_is_not_retried():
    session = _FakeSession()
    session.failures = [McpError(ErrorData(code=CONNECTION_CLOSED, message="Connection closed"))]
    opener = _Opener(session)
    gw = _gateway(opener)
    try:
        await gw._ensure_child_sessions()
        first_owner = _owner(gw)
        with pytest.raises(McpError, match="Connection closed"):
            await gw._call_child_tool(URL, "svc_thing", {})
        assert session.calls == ["svc_thing"]  # the child may already have written; a retry could write twice
        await asyncio.gather(*gw._sessions.retirements)
        assert first_owner.done() and URL not in gw._sessions.current and opener.opened == 1
        assert not (await gw._call_child_tool(URL, "svc_thing", {})).isError  # the next call reopens
        assert opener.opened == 2 and session.calls == ["svc_thing", "svc_thing"]
    finally:
        await gw._close_child_sessions()


@pytest.mark.asyncio
async def test_any_other_error_is_the_caller_s_and_keeps_the_session():
    session = _FakeSession()
    session.failures = [McpError(ErrorData(code=-32602, message="Invalid params"))]
    opener = _Opener(session)
    gw = _gateway(opener)
    try:
        with pytest.raises(McpError, match="Invalid params"):
            await gw._call_child_tool(URL, "svc_thing", {})
        assert session.calls == ["svc_thing"] and opener.opened == 1 and not _owner(gw).done()
    finally:
        await gw._close_child_sessions()


@pytest.mark.asyncio
async def test_concurrent_dead_session_errors_reopen_the_session_once():
    session = _FakeSession()
    opener = _Opener(session)
    gw = _gateway(opener)

    async def call_tool(name, arguments, *args, meta=None, **kwargs):  # every call on the first session dies
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
async def test_a_failed_open_leaves_no_session_and_is_retried_on_the_next_use():
    opener = _Opener(_FakeSession(), failures=[RuntimeError("refused")])
    gw = _gateway(opener)
    with pytest.raises(RuntimeError, match="refused"):
        await gw._ensure_child_sessions()
    assert gw._sessions.current == {}
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
        assert not _owner(gw).done()  # a failed tools/list is not a dead transport

        await gw._ensure_tools_discovered()
        assert gw._tools_discovered and [t.name for t in gw._server_tools["svc"]] == ["svc_thing"]
        assert engine._is_readonly("svc_thing") is True  # discovery dropped the early memo
        assert opener.opened == 1
        gw._sessions.open_session = _explode  # type: ignore[method-assign]
        await gw._ensure_tools_discovered()
    finally:
        await gw._close_child_sessions()


@pytest.mark.asyncio
async def test_a_child_that_hangs_on_tools_list_fails_discovery_within_its_bound_and_keeps_the_session(monkeypatch):
    class _HangsOnList(_FakeSession):
        hang = True

        async def list_tools(self):
            if self.hang:
                await asyncio.Event().wait()
            return await super().list_tools()

    session = _HangsOnList()
    opener = _Opener(session)
    gw = _gateway(opener)
    monkeypatch.setattr(gateway_module, "CHILD_DISCOVERY_TIMEOUT_S", 0.02)
    try:
        with pytest.raises(TimeoutError):
            await gw._ensure_tools_discovered()
        assert gw._tools_discovered is False
        session.hang = False
        await gw._ensure_tools_discovered()
        assert gw._tools_discovered is True
        assert opener.opened == 1  # a slow child is not a dead one: the session stays
    finally:
        await gw._close_child_sessions()


@pytest.mark.asyncio
async def test_discovery_on_a_dead_session_replaces_it():
    session = _FakeSession()
    session.list_failures = [McpError(ErrorData(code=32600, message="Session terminated"))]
    opener = _Opener(session)
    gw = _gateway(opener)
    try:
        with pytest.raises(McpError):
            await gw._ensure_tools_discovered()
        assert URL not in gw._sessions.current
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
    async def streams(url, **kwargs):
        yield None, None, None

    monkeypatch.setattr(child_sessions_module, "streamable_http_client", streams)
    monkeypatch.setattr(child_sessions_module, "ClientSession", _HungSession)
    monkeypatch.setattr(child_sessions_module, "CARD_FETCH_TIMEOUT_S", 0.02)
    gw = Gateway(host="127.0.0.1", port=0, server_name="t", internal_mcp_servers=[InternalMCPServer("svc", URL)])
    async with contextlib.AsyncExitStack() as stack:
        with pytest.raises(TimeoutError):
            await gw._sessions.open_session(gw.internal_mcp_servers[0], stack, lambda exc: None, max_retries=2,
                                         retry_delay=0)


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
    assert gw._sessions.current == {} and opener.attempts == 1  # the failed open left nothing behind
    await gw._ensure_tools_discovered()  # the next use heals it
    assert gw._tools_discovered and not _owner(gw).done()
    release.set()
    await asyncio.wait_for(task, 5)
    assert sent[-1] == {"type": "lifespan.shutdown.complete"}
    assert gw._trigger_engine._driver_task is None and gw._sessions.current == {}

    # the MCP session manager runs once per app, so a second lifespan fails its startup; the process state still unwinds
    sent2: list[dict] = []
    with pytest.raises(RuntimeError, match="only be called once"):
        await _run_lifespan(app, asyncio.Event(), sent2)
    assert sent2[0]["type"] == "lifespan.startup.failed"
    assert gw._trigger_engine._driver_task is None and gw._sessions.current == {}


@pytest.mark.asyncio
async def test_startup_discovery_is_bounded_and_a_hung_open_leaves_no_session(monkeypatch):
    monkeypatch.setattr(Gateway, "STARTUP_DISCOVERY_TIMEOUT_S", 0.05)
    gate = asyncio.Event()
    opener = _Opener(_FakeSession(), gate=gate)  # the open hangs until released
    gw = _gateway(opener)
    app = gw._asgi_app()
    release, sent = asyncio.Event(), []
    task = asyncio.create_task(_run_lifespan(app, release, sent))
    await _startup_answered(sent)
    assert sent == [{"type": "lifespan.startup.complete"}] and not gw._trigger_engine._driver_task.done()
    assert gw._sessions.current == {} and not gw._tools_discovered
    assert not gw._sessions.lock(URL).locked() and not gw._discover_lock.locked()
    await asyncio.sleep(0)
    # the abandoned open was cancelled, not left running to finish for nobody
    assert not [t for t in asyncio.all_tasks() if t.get_name() == "gateway-child-session" and not t.done()]
    assert opener.attempts == 1 and opener.opened == 0
    gate.set()
    try:
        assert (await gw._ensure_child_sessions())[URL] is opener.session
        await asyncio.sleep(0.01)
        assert opener.attempts == 2 and opener.opened == 1 and opener.exited == []
        assert opener.entered[0] is _owner(gw) and not _owner(gw).done()
        await gw._ensure_tools_discovered()
        assert gw._tools_discovered
    finally:
        release.set()
        await asyncio.wait_for(task, 5)
    assert sent[-1] == {"type": "lifespan.shutdown.complete"} and gw._sessions.current == {}


@pytest.mark.asyncio
async def test_shutdown_cancels_scheduled_actions_and_nothing_reopens_the_child_sessions():
    session = _FakeSession()
    opener = _Opener(session)
    gw = _gateway(opener)
    app = gw._asgi_app()
    release, sent = asyncio.Event(), []
    task = asyncio.create_task(_run_lifespan(app, release, sent))
    await _startup_answered(sent)
    assert sent == [{"type": "lifespan.startup.complete"}] and not _owner(gw).done()
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
    assert gw._sessions.current == {}
    with pytest.raises(RuntimeError, match="shutting down"):
        await gw._ensure_child_sessions()
    assert gw._sessions.current == {} and opener.attempts == 1


@pytest.mark.asyncio
async def test_a_reported_transport_failure_retires_the_session_once_and_the_next_call_reopens():
    """The child transport reports a failed POST; the session is retired at once (so its pending calls fail with
    CONNECTION_CLOSED) and a second report for the same session, or one for a session already gone, is nothing."""
    session = _FakeSession()
    opener = _Opener(session)
    gw = _gateway(opener)
    first = await gw._sessions.ensure(URL)
    gw._sessions.transport_failed(first, httpx.ReadError("peer went away"))
    gw._sessions.transport_failed(first, httpx.ReadError("again"))  # every request in flight reports; one retirement
    assert len(gw._sessions.retirements) == 1
    await asyncio.gather(*gw._sessions.retirements)
    assert gw._sessions.retirements == set() and gw._sessions.current == {}
    assert opener.exited == opener.entered and len(opener.exited) == 1, "retired exactly once, by the task that opened it"

    second = await gw._sessions.ensure(URL)
    assert second is not first and opener.opened == 2
    gw._sessions.transport_failed(first, httpx.ReadError("stale"))  # a report about the retired session
    assert gw._sessions.retirements == set() and gw._sessions.current[URL] is second

    gw._sessions.closing = True
    gw._sessions.transport_failed(second, httpx.ReadError("at shutdown"))
    assert gw._sessions.retirements == set(), "nothing is retired on the way down; the process lifespan closes it"


@pytest.mark.asyncio
async def test_a_session_the_child_rejects_is_replaced_and_each_call_already_sent_on_it_is_retried_once():
    """A restarted child answers 404 for the old session id, to each call sent on it: the first such answer replaces
    the session, and every other call sent on it still gets its own answer, so each is retried once on the fresh one
    rather than failed; the old session closes once the last of them ends."""
    first, second = _FakeSession(), _FakeSession()
    rejected = asyncio.Event()

    async def rejects(name, arguments, *args, meta=None, **kwargs):
        first.calls.append(name)
        await rejected.wait()
        raise McpError(ErrorData(code=32600, message="Session terminated"))

    first.call_tool = rejects
    opener = _Opener(first)
    gw = _gateway(opener)
    try:
        old = await gw._sessions.ensure(URL)
        opener.session = second
        calls = [asyncio.create_task(gw._call_child_tool(URL, f"t{i}", {})) for i in range(5)]
        while len(first.calls) < 5:
            await asyncio.sleep(0)
        rejected.set()
        results = await asyncio.gather(*calls)
        assert all(not r.isError for r in results)
        assert sorted(second.calls) == [f"t{i}" for i in range(5)] and opener.opened == 2
        await asyncio.wait_for(old.owner, 1)
        assert old.close.is_set() and old not in gw._sessions.draining and gw._sessions.current[URL].session is second
    finally:
        await gw._close_child_sessions()


@pytest.mark.asyncio
async def test_one_child_s_dead_session_leaves_another_child_s_calls_alone():
    """A call that dies in flight on one child retires that child's session alone: a call in flight on another child
    finishes, and its session stays open."""
    dying, slow = _FakeSession(), _FakeSession()
    dying.failures = [McpError(ErrorData(code=CONNECTION_CLOSED, message="Connection closed"))]
    release = asyncio.Event()

    async def held(name, arguments, *args, meta=None, **kwargs):
        slow.calls.append(name)
        await release.wait()
        return _result()

    slow.call_tool = held
    sessions = {URL: dying, OTHER: slow}

    async def opener(server, stack, report):
        return sessions[server.mcp_url]

    gw = Gateway(host="127.0.0.1", port=0, server_name="t",
                 internal_mcp_servers=[InternalMCPServer("svc", URL), InternalMCPServer("other", OTHER)])
    gw._sessions.open_session = opener  # type: ignore[method-assign]
    try:
        in_flight = asyncio.create_task(gw._call_child_tool(OTHER, "other_thing", {}))
        while not slow.calls:
            await asyncio.sleep(0)
        other_owner = _owner(gw, OTHER)
        with pytest.raises(McpError, match="Connection closed"):
            await gw._call_child_tool(URL, "svc_thing", {})
        release.set()
        assert not (await in_flight).isError
        await asyncio.gather(*gw._sessions.retirements)
        assert URL not in gw._sessions.current and _owner(gw, OTHER) is other_owner and not other_owner.done()
    finally:
        await gw._close_child_sessions()


@pytest.mark.asyncio
async def test_a_replaced_session_whose_transport_then_fails_fails_the_calls_still_on_it():
    """A replaced session waits for the calls already sent on it, unless its transport fails: then they may have
    reached the child, so they fail with CONNECTION_CLOSED, and are not retried."""
    session = _FakeSession()
    release = asyncio.Event()

    async def held(name, arguments, *args, meta=None, **kwargs):
        session.calls.append(name)
        await release.wait()
        return _result()

    session.call_tool = held
    opener = _Opener(session)
    gw = _gateway(opener)
    try:
        old = await gw._sessions.ensure(URL)
        pending = asyncio.create_task(gw._call_child_tool(URL, "svc_thing", {}))
        while not session.calls:
            await asyncio.sleep(0)
        await gw._sessions.replace(old)  # as a call the child answered 404 does
        assert old in gw._sessions.draining and not old.close.is_set()
        gw._sessions.transport_failed(old, httpx.ReadError("peer went away"))
        await asyncio.gather(*gw._sessions.retirements)
        with pytest.raises(McpError, match="Connection closed"):
            await pending
        assert old.owner.done() and old not in gw._sessions.draining and session.calls == ["svc_thing"]
    finally:
        await gw._close_child_sessions()


@pytest.mark.asyncio
async def test_a_replaced_session_whose_transport_fails_while_the_next_one_opens_fails_its_calls_at_once():
    """The open of the next session holds the child's lock through its connection attempts; retiring the old one
    doesn't wait for it, so the calls still on the old one fail now, not once the open is done."""
    session = _FakeSession()
    release = asyncio.Event()

    async def held(name, arguments, *args, meta=None, **kwargs):
        session.calls.append(name)
        await release.wait()
        return _result()

    session.call_tool = held
    opener = _Opener(session)
    gw = _gateway(opener)
    opening = None
    try:
        old = await gw._sessions.ensure(URL)
        pending = asyncio.create_task(gw._call_child_tool(URL, "svc_thing", {}))
        while not session.calls:
            await asyncio.sleep(0)
        await gw._sessions.replace(old)
        opener.gate = asyncio.Event()  # the next open waits, holding the child's lock
        opening = asyncio.create_task(gw._sessions.ensure(URL))
        while opener.attempts < 2:
            await asyncio.sleep(0)
        gw._sessions.transport_failed(old, httpx.ReadError("peer went away"))
        with pytest.raises(McpError, match="Connection closed"):
            await asyncio.wait_for(pending, 1)
        assert not opening.done()
        opener.gate.set()
        assert (await opening).session is session and old.owner.done() and old not in gw._sessions.draining
    finally:
        if opening is not None and not opening.done():
            opener.gate.set()
            await opening
        await gw._close_child_sessions()
