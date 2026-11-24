"""The gateway's server keeps an idle connection open for IDLE_CONNECTION_TIMEOUT_S, over a real loopback socket.

A client that pools connections (Claude Code's fetch does) sends a request on one the server is closing as its idle
timer fires; the request is reset unread. uvicorn's default timer, 5 s, sat inside an agent's ordinary pause between
tool calls."""
from __future__ import annotations

import asyncio
import socket
import time

import pytest
import uvicorn
from pytest_socket import enable_socket

from agent_env.env.gateway import gateway as gateway_module
from agent_env.env.gateway.constants import IDLE_CONNECTION_TIMEOUT_S
from agent_env.env.gateway.gateway import Gateway


@pytest.fixture(autouse=True)
def _disable_network():
    """Override the suite-wide socket block (tst/unit/conftest.py) for this module: loopback is the point here."""
    enable_socket()
    yield


def test_an_idle_connection_outlasts_any_pause_between_tool_calls():
    assert IDLE_CONNECTION_TIMEOUT_S >= MIN_IDLE_TIMEOUT_S
    gw = Gateway(host="127.0.0.1", port=0, server_name="t", internal_mcp_servers=[])
    assert gw._server_config().timeout_keep_alive == IDLE_CONNECTION_TIMEOUT_S


@pytest.mark.asyncio
async def test_the_server_closes_an_idle_connection_only_when_the_timeout_runs_out(monkeypatch):
    monkeypatch.setattr(gateway_module, "IDLE_CONNECTION_TIMEOUT_S", TEST_IDLE_TIMEOUT_S)
    gw = Gateway(host="127.0.0.1", port=0, server_name="t", internal_mcp_servers=[])
    config = gw._server_config()
    config.log_level = "critical"
    config.access_log = False
    config.load()
    server = uvicorn.Server(config)
    server.lifespan = config.lifespan_class(config)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    await server.startup(sockets=[sock])
    reader, writer = await asyncio.open_connection("127.0.0.1", sock.getsockname()[1])
    try:
        for idle_s in REUSE_AFTER_S:  # reused within the timeout: still served
            await asyncio.sleep(idle_s)
            writer.write(REQUEST)
            await writer.drain()
            head = await reader.readuntil(b"\r\n\r\n")
            assert head.startswith(b"HTTP/1.1 404"), head
            length = next(int(line.split(b":")[1]) for line in head.split(b"\r\n")
                          if line.lower().startswith(b"content-length"))
            await reader.readexactly(length)
        idle_from = time.monotonic()
        closed = await asyncio.wait_for(reader.read(1), timeout=TEST_IDLE_TIMEOUT_S + 2 * SCHEDULING_SLACK_S)
        assert closed == b""  # then closed by the server
        assert abs(time.monotonic() - idle_from - TEST_IDLE_TIMEOUT_S) < SCHEDULING_SLACK_S
    finally:
        writer.close()
        await server.shutdown()


# An agent pauses seconds between tool calls, minutes at most; the real timeout must sit well past that.
MIN_IDLE_TIMEOUT_S = 600
# The timeout patched in for the socket test: short enough for a unit test, well under uvicorn's default 5 s.
TEST_IDLE_TIMEOUT_S = 2
# Idle times the connection is reused after, both inside TEST_IDLE_TIMEOUT_S.
REUSE_AFTER_S = (0, 1)
# How far the measured close may land from TEST_IDLE_TIMEOUT_S under event-loop scheduling.
SCHEDULING_SLACK_S = 1.0
REQUEST = b"GET /nothing-here HTTP/1.1\r\nHost: gateway\r\n\r\n"
