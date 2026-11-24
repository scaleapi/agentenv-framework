"""Opted-in extension calls, through HTTPX's real request and redirect handling."""

import asyncio
from contextlib import nullcontext

import httpcore
import httpx
import pytest
import trio
from httpcore._backends.auto import AutoBackend
from trio.testing import MockClock

from agentenv_protocol import ExtensionCallError, ToolSession, client


_URL = "http://env.test"
_URI = "urn:example:setup/v1"
_CARD = {"capabilities": {"extensions": [{"uri": _URI, "params": {"methods": {
    "set": {"method": "POST", "endpoint": "/set"}, "get": {"method": "GET", "endpoint": "/get"},
}}}]}}


async def extension(method="set", timeout=5):
    return await client.invoke_extension(_URL, _CARD, _URI, {"value": 1}, timeout=timeout, method=method)


def mock_http(monkeypatch, handle, *, redirects=False):
    real = httpx.AsyncClient

    def make_client(**kwargs):
        kwargs.setdefault("follow_redirects", redirects)
        return real(transport=httpx.MockTransport(handle), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", make_client)


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["get", "set"])
async def test_only_opted_in_extensions_retry(monkeypatch, method):
    attempts, received = [], []

    def handle(request):
        attempts.append(request)
        if len(attempts) == 1:
            raise httpx.ConnectTimeout("before sending")
        received.append(request)
        return httpx.Response(200, json={"ok": True})

    mock_http(monkeypatch, handle)
    with pytest.raises(httpx.ConnectTimeout):
        await extension(method)
    assert len(attempts) == 1 and not received
    attempts.clear()
    with client.retry_extension_connections():
        assert await extension(method) == {"ok": True}
    assert len(attempts) == 2
    [request] = received
    if method == "get":
        assert request.method == "GET" and request.url.params["value"] == "1"
    else:
        assert request.method == "POST" and request.content
    attempts.clear()
    with pytest.raises(httpx.ConnectTimeout):
        await extension(method)
    assert len(attempts) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["data", "card", "manifest", "tool"])
async def test_other_clients_still_make_one_attempt_in_the_retry_context(monkeypatch, route):
    attempts = []

    def refuse(request):
        attempts.append(request)
        raise httpx.ConnectError("connection refused")

    mock_http(monkeypatch, refuse)
    with client.retry_extension_connections(), pytest.raises(httpx.ConnectError):
        if route == "data":
            await client.add_data(_URL, [])
        elif route == "card":
            await client.get_card(_URL)
        elif route == "manifest":
            await client.get_interface_manifest(_URL, "items")
        else:
            async with ToolSession(mcp_url=_URL + "/mcp"):
                pytest.fail("MCP initialization should fail")
    assert len(attempts) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout, expected", [(5, 3), (0.25, 1)])
async def test_exhaustion_returns_the_last_connection_error(monkeypatch, timeout, expected):
    started = []

    def refuse(request):
        started.append(asyncio.get_running_loop().time())
        raise httpx.ConnectError(f"attempt {len(started)}")

    mock_http(monkeypatch, refuse)
    with client.retry_extension_connections(), pytest.raises(httpx.ConnectError, match=f"attempt {expected}") as caught:
        await extension(timeout=timeout)
    assert len(started) == expected and caught.value.request.url.path == "/set"
    if expected == 3:
        assert started[1] - started[0] >= 0.49 and started[2] - started[1] >= 0.99


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout, ExtensionCallError])
async def test_other_errors_propagate_after_one_attempt(monkeypatch, error):
    calls = []

    def handle(request):
        calls.append(request)
        if error is ExtensionCallError:
            return httpx.Response(503)
        raise error("interrupted")

    mock_http(monkeypatch, handle)
    with client.retry_extension_connections(), pytest.raises(error):
        await extension()
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_a_failed_redirect_target_does_not_replay_the_first_hop(monkeypatch):
    paths = []

    def handle(request):
        paths.append(request.url.path)
        if request.url.path == "/set":
            return httpx.Response(307, headers={"Location": "/next"})
        raise httpx.ConnectError("redirect target unavailable")

    mock_http(monkeypatch, handle, redirects=True)
    with client.retry_extension_connections(), pytest.raises(httpx.ConnectError):
        await extension()
    assert paths == ["/set", "/next"]


@pytest.mark.asyncio
async def test_cancelling_one_opted_in_task_leaves_a_sibling_on_one_attempt(monkeypatch):
    failed = asyncio.Event()
    attempts = []

    def refuse(request):
        attempts.append(request)
        failed.set()
        raise httpx.ConnectError("refused")

    mock_http(monkeypatch, refuse)
    with client.retry_extension_connections():
        task = asyncio.create_task(extension())
    await failed.wait()
    with pytest.raises(httpx.ConnectError):
        await extension()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(attempts) == 2


@pytest.mark.parametrize("opt_in", [False, True])
def test_native_trio_uses_its_own_clock_for_backoff(monkeypatch, opt_in):
    attempts = []

    async def connect(self, host, **kwargs):
        attempts.append(trio.current_time())
        if len(attempts) == 1:
            raise httpcore.ConnectError("connection refused")
        return httpcore.AsyncMockStream([b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}"])

    async def run():
        with client.retry_extension_connections() if opt_in else nullcontext():
            if opt_in:
                assert await extension() == {}
            else:
                with pytest.raises(httpx.ConnectError):
                    await extension()

    monkeypatch.setenv("no_proxy", "*")
    monkeypatch.setattr(AutoBackend, "connect_tcp", connect)
    trio.run(run, clock=MockClock(autojump_threshold=0))
    assert attempts == ([0, 0.5] if opt_in else [0])
