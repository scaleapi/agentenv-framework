"""Drive the protocol clients through HTTPX, injecting failures at the TCP boundary."""

import asyncio
import json

import httpcore
import httpx
import pytest
from anyio.lowlevel import checkpoint
from httpcore._backends.auto import AutoBackend

from agentenv_protocol import ExtensionCallError, ToolSession, ToolSessionError, client


_URL = "http://env.test"
_CARD = {"capabilities": {"extensions": [{"uri": "urn:test", "params": {"methods": {
    "set": {"method": "POST", "endpoint": "/set"}, "get": {"method": "GET", "endpoint": "/get"},
}}}]}}


async def call(route, timeout=5):
    if route == "rpc":
        return await client.add_data(_URL, [], timeout=timeout)
    if route == "card":
        return await client.get_card(_URL, timeout=timeout)
    if route == "manifest":
        return await client.get_interface_manifest(_URL, "items", timeout=timeout)
    if route == "tool":
        async with ToolSession(mcp_url=_URL + "/mcp", timeout=timeout) as session:
            return await session.call_tool("sample")
    return await client.invoke_extension(_URL, _CARD, "urn:test", {"value": 1}, method=route, timeout=timeout)


class Wire:
    def __init__(self):
        self.attempts = 0
        self.requests = []
        self.closed = 0
        self.on_connect = None
        self.read_error = self.write_error = None
        self.status = 200
        self.redirect = False

    async def connect(self, host):
        self.attempts += 1
        if self.on_connect:
            await self.on_connect(self.attempts)
        if host == "redirect.test":
            raise httpcore.ConnectError("redirect target unavailable")
        return Stream(self)


class Stream(httpcore.AsyncMockStream):
    def __init__(self, wire):
        super().__init__([])
        self.wire = wire
        self.data = bytearray()

    async def write(self, buffer, timeout=None):
        self.data.extend(buffer)
        if self.wire.write_error:
            raise self.wire.write_error("interrupted write")

    async def read(self, max_bytes, timeout=None):
        headers, body = self.data.split(b"\r\n\r\n", 1)
        message = json.loads(body or b"{}")
        path = headers.split(b" ", 2)[1].decode()
        method = message.get("method", "")
        self.wire.requests.append((path, method))
        if self.wire.read_error:
            raise self.wire.read_error("interrupted read")
        if self.wire.redirect:
            return b"HTTP/1.1 307 Temporary Redirect\r\nLocation: http://redirect.test/mcp\r\nContent-Length: 0\r\n\r\n"
        data = json.dumps({"jsonrpc": "2.0", "id": message.get("id"), "result": {}}).encode()
        status = self.wire.status
        if method == "notifications/initialized" and status == 200:
            status, data = 202, b""
        return (f"HTTP/1.1 {status} Test\r\nContent-Type: application/json\r\nContent-Length: {len(data)}\r\n"
                "Connection: close\r\n\r\n").encode() + data

    async def aclose(self):
        self.wire.closed += 1
        await super().aclose()


@pytest.fixture
def wire(monkeypatch):
    wire = Wire()
    monkeypatch.setenv("no_proxy", "*")

    async def connect_tcp(self, host, **kwargs):
        return await wire.connect(host)

    monkeypatch.setattr(AutoBackend, "connect_tcp", connect_tcp)
    return wire


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["rpc", "card", "manifest", "set", "get", "tool"])
async def test_connection_timeout_recovers_before_the_request_is_sent(wire, route):
    failed_attempt = 3 if route == "tool" else 1

    async def fail_once(attempt):
        if attempt == failed_attempt:
            raise httpcore.ConnectTimeout("before sending")

    wire.on_connect = fail_once
    await call(route)
    assert wire.attempts == failed_attempt + 1
    assert len(wire.requests) == failed_attempt
    assert wire.closed == len(wire.requests)
    if route == "tool":
        assert [method for _, method in wire.requests] == ["initialize", "notifications/initialized", "tools/call"]


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [httpcore.ConnectError, httpcore.ConnectTimeout])
async def test_failed_connections_stop_after_three_attempts(wire, error):
    started = []

    async def fail(attempt):
        started.append(asyncio.get_running_loop().time())
        raise error(f"attempt {attempt}")

    wire.on_connect = fail
    with pytest.raises(getattr(httpx, error.__name__), match="attempt 3") as caught:
        await call("rpc")
    assert caught.value.request.url.path == "/agentenv"
    assert wire.attempts == 3 and wire.requests == []
    assert started[1] - started[0] >= 0.49
    assert started[2] - started[1] >= 0.99


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["rpc", "card", "manifest", "set", "get", "tool"])
async def test_retried_call_uses_the_original_deadline(wire, route):
    async def connect(attempt):
        if attempt == 1:
            await asyncio.sleep(0.45)
            raise httpcore.ConnectError("first attempt")
        await asyncio.sleep(0.65)

    wire.on_connect = connect
    with pytest.raises(TimeoutError if route == "tool" else httpx.TimeoutException):
        await call(route, timeout=1.4)
    assert wire.attempts == 2 and wire.requests == []


@pytest.mark.asyncio
async def test_first_attempt_can_use_up_the_backoff_budget(wire):
    async def slow_failure(attempt):
        await asyncio.sleep(0.2)
        raise httpcore.ConnectError("before sending")

    wire.on_connect = slow_failure
    with pytest.raises(httpx.ConnectError, match="before sending"):
        await call("rpc", timeout=0.6)
    assert wire.attempts == 1 and wire.requests == []


@pytest.mark.asyncio
async def test_cancellation_during_backoff_propagates(wire):
    failed = asyncio.Event()

    async def connect(attempt):
        failed.set()
        raise httpcore.ConnectError("before sending")

    wire.on_connect = connect
    task = asyncio.create_task(call("rpc"))
    await failed.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert wire.attempts == 1 and wire.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["rpc", "set", "tool"])
@pytest.mark.parametrize("failure", ["read", "write", "http"])
async def test_ambiguous_failures_and_http_responses_are_not_retried(wire, route, failure):
    if failure == "read":
        wire.read_error, expected = httpcore.ReadTimeout, httpx.ReadTimeout
    elif failure == "write":
        wire.write_error, expected = httpcore.WriteTimeout, httpx.WriteTimeout
    else:
        wire.status = 503
        expected = {"rpc": httpx.HTTPStatusError, "set": ExtensionCallError, "tool": ToolSessionError}[route]
    with pytest.raises(expected):
        await call(route)
    assert wire.attempts == 1
    assert wire.closed == 1


@pytest.mark.asyncio
async def test_failure_connecting_to_a_redirect_never_resends_the_original_post(wire):
    wire.redirect = True
    with pytest.raises(httpx.ConnectError) as caught:
        await call("tool")
    assert caught.value.request.url.host == "redirect.test"
    assert wire.requests == [("/mcp", "initialize")]
    assert wire.attempts == 2 and wire.closed == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["rpc", "tool"])
async def test_deadline_closes_an_incomplete_response_after_retry(monkeypatch, route):
    attempts = 0
    closed = []

    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"data: "
            await asyncio.sleep(5)

        async def aclose(self):
            await checkpoint()
            closed.append(True)

    def handle(request):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ConnectTimeout("first attempt", request=request)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Body())

    transport = httpx.MockTransport(handle)

    async def send(self, request):
        return await transport.handle_async_request(request)

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", send)
    with pytest.raises(TimeoutError if route == "tool" else httpx.TimeoutException):
        await call(route, timeout=0.7)
    assert attempts == 2 and closed == [True]
