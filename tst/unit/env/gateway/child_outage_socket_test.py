"""A child that dies with a call in flight, over real loopback sockets: the call fails within seconds instead of
waiting out the tool-call timeout, and the next call reopens the session once the child is back."""
from __future__ import annotations

import asyncio
import functools
import socket
import time

import pytest
import uvicorn
from mcp.server.fastmcp import FastMCP
from mcp.shared.exceptions import McpError
from mcp.types import CONNECTION_CLOSED
from pytest_socket import enable_socket

from agent_env.env.gateway import InternalMCPServer
from agent_env.env.gateway.child_sessions import ChildSessions
from agent_env.env.gateway.gateway import Gateway


@pytest.fixture(autouse=True)
def _loopback():
    """Override the suite-wide socket block (tst/unit/conftest.py): loopback is the point here."""
    enable_socket()
    yield


def _child(slow_started: asyncio.Event) -> FastMCP:
    app = FastMCP("child")

    @app.tool()
    async def fast() -> str:
        return "ok"

    @app.tool()
    async def slow() -> str:
        slow_started.set()
        await asyncio.Event().wait()  # never set: the call is in flight until the child dies
        return "never"

    return app


def _bind(port: int = 0) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", port))
    return sock


async def _serve(app: FastMCP, sock: socket.socket) -> uvicorn.Server:
    config = uvicorn.Config(app.streamable_http_app(), log_level="critical", access_log=False, timeout_graceful_shutdown=1)
    config.load()
    server = uvicorn.Server(config)
    server.lifespan = config.lifespan_class(config)
    await server.startup(sockets=[sock])
    return server


def _text(result) -> str:
    return result.content[0].text


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_a_child_dying_mid_call_fails_the_call_fast_and_the_next_call_reopens():
    sock = _bind()
    port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}/mcp"
    slow_started = asyncio.Event()
    server = await _serve(_child(slow_started), sock)
    gw = Gateway(host="127.0.0.1", port=0, server_name="t", internal_mcp_servers=[InternalMCPServer("svc", url)])
    try:
        await gw._ensure_tools_discovered()
        assert _text(await gw._call_child_tool(url, "fast", {})) == "ok"
        first_session = gw._sessions.current[url]

        in_flight = asyncio.create_task(gw._call_child_tool(url, "slow", {}))
        await asyncio.wait_for(slow_started.wait(), 10)  # the child holds the call
        assert not in_flight.done()

        died = time.monotonic()
        await server.shutdown()  # the child goes away with the call in flight
        with pytest.raises(McpError) as info:
            await asyncio.wait_for(in_flight, timeout=30)
        assert info.value.error.code == CONNECTION_CLOSED
        assert time.monotonic() - died < 10, "the in-flight call must fail promptly, not after the tool-call timeout"
        await asyncio.gather(*gw._sessions.retirements)
        assert url not in gw._sessions.current, "the session was retired"

        server = await _serve(_child(asyncio.Event()), _bind(port))  # the child is back on the same address
        assert _text(await gw._call_child_tool(url, "fast", {})) == "ok", "the next call reopens"
        assert gw._sessions.current[url] is not first_session
    finally:
        await gw._close_child_sessions()
        await server.shutdown()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_a_call_arriving_while_the_child_is_still_down_waits_for_the_reconnect_not_for_its_own_timeout():
    """mcp's client runs its requests in a task group whose cancel scope belongs to the task that opened the
    session: a reconnect attempt the child refuses cancels the session's owner rather than raising to it. The
    attempt must read as a failure to retry, so a call that arrives during the outage completes as soon as the child
    is back, and fails within the reconnect schedule if it never comes back; before, that call waited out its own
    timeout with every other call queued behind the reconnect lock."""
    sock = _bind()
    port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}/mcp"
    server = await _serve(_child(asyncio.Event()), sock)
    gw = Gateway(host="127.0.0.1", port=0, server_name="t", internal_mcp_servers=[InternalMCPServer("svc", url)])
    gw._sessions.open_session = functools.partial(ChildSessions.open_session, gw._sessions, max_retries=4,
                                                 retry_delay=0.5)
    try:
        await gw._ensure_tools_discovered()
        await server.shutdown()
        with pytest.raises(McpError) as info:  # the call that finds the child gone retires its session
            await asyncio.wait_for(gw._call_child_tool(url, "fast", {}), timeout=10)
        assert info.value.error.code == CONNECTION_CLOSED

        started = time.monotonic()
        during = asyncio.create_task(gw._call_child_tool(url, "fast", {}))  # reconnects; refused until the child is back
        await asyncio.sleep(1.2)
        assert not during.done()
        server = await _serve(_child(asyncio.Event()), _bind(port))
        assert _text(await asyncio.wait_for(during, timeout=10)) == "ok"
        assert time.monotonic() - started < 10

        await server.shutdown()
        with pytest.raises(McpError):
            await asyncio.wait_for(gw._call_child_tool(url, "fast", {}), timeout=10)
        started = time.monotonic()
        with pytest.raises(RuntimeError, match="could not open a session"):  # the child stays down: 4 refused attempts
            await asyncio.wait_for(gw._call_child_tool(url, "fast", {}), timeout=20)
        assert time.monotonic() - started < 10
        server = await _serve(_child(asyncio.Event()), _bind(port))
        assert _text(await gw._call_child_tool(url, "fast", {})) == "ok"
    finally:
        await gw._close_child_sessions()
        await server.shutdown()


def _counting_child(name: str, writes: list[str]) -> FastMCP:
    app = FastMCP(name)

    @app.tool(name=f"{name}_place")
    async def place(item: str) -> str:
        writes.append(item)
        return f"placed {item}"

    @app.tool(name=f"{name}_slow_place")
    async def slow_place(item: str) -> str:
        await asyncio.sleep(0.5)
        writes.append(item)
        return f"placed {item}"

    return app


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_after_a_child_restarts_a_burst_of_calls_to_it_succeeds_and_another_child_s_calls_are_untouched():
    """The restarted child answers 404 for the old session to each call sent on it, so each is retried once on a fresh
    session: none fails, and each writes once. Calls to the other child, one in flight across the burst included,
    succeed on the session it already had."""
    socks = {name: _bind() for name in ("a", "b")}
    ports = {name: sock.getsockname()[1] for name, sock in socks.items()}
    urls = {name: f"http://127.0.0.1:{port}/mcp" for name, port in ports.items()}
    writes: dict[str, list[str]] = {"a": [], "b": []}
    servers = {name: await _serve(_counting_child(name, writes[name]), sock) for name, sock in socks.items()}
    gw = Gateway(host="127.0.0.1", port=0, server_name="t",
                 internal_mcp_servers=[InternalMCPServer(name, url) for name, url in urls.items()])
    try:
        await gw._ensure_tools_discovered()
        b_session = gw._sessions.current[urls["b"]]
        await servers["a"].shutdown()
        servers["a"] = await _serve(_counting_child("a", writes["a"]), _bind(ports["a"]))  # back, its sessions gone
        slow = asyncio.create_task(gw._call_child_tool(urls["b"], "b_slow_place", {"item": "slow"}))
        results = await asyncio.gather(
            *(gw._call_child_tool(urls["a"], "a_place", {"item": f"a{i}"}) for i in range(20)),
            *(gw._call_child_tool(urls["b"], "b_place", {"item": f"b{i}"}) for i in range(10)))
        assert all(not result.isError for result in results)
        assert not (await slow).isError
        assert sorted(writes["a"]) == sorted(f"a{i}" for i in range(20))
        assert sorted(writes["b"]) == sorted(["slow", *(f"b{i}" for i in range(10))])
        assert gw._sessions.current[urls["b"]] is b_session
    finally:
        await gw._close_child_sessions()
        for server in servers.values():
            await server.shutdown()
