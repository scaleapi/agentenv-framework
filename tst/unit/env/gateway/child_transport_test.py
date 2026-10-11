"""The reporting transport under a child session: a POST that fails at send or while its body streams is reported
once and still raises; GETs are not reported; the client keeps mcp's own timeouts."""
from __future__ import annotations

import httpx
import pytest

from agent_env.env.gateway.child_transport import CHILD_TIMEOUT, ReportingTransport, child_http_client


class _FailingBody(httpx.AsyncByteStream):
    """One chunk, then the connection dies."""

    async def __aiter__(self):
        yield b"event: message\n"
        raise httpx.ReadError("peer went away")


def _client(handler, reports: list) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=ReportingTransport(httpx.MockTransport(handler), reports.append),
                             base_url="http://child")


@pytest.mark.asyncio
async def test_a_post_that_fails_to_send_is_reported_and_still_raises():
    reports: list[httpx.TransportError] = []

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    async with _client(handler, reports) as client:
        with pytest.raises(httpx.ConnectError):
            await client.post("/mcp", json={})
    assert [type(e) for e in reports] == [httpx.ConnectError]


@pytest.mark.asyncio
async def test_a_post_whose_body_dies_mid_stream_is_reported_to_the_reader():
    reports: list[httpx.TransportError] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=_FailingBody(), headers={"content-type": "text/event-stream"})

    async with _client(handler, reports) as client:
        chunks: list[bytes] = []
        with pytest.raises(httpx.ReadError):
            async with client.stream("POST", "/mcp", json={}) as response:
                async for chunk in response.aiter_raw():
                    chunks.append(chunk)
    assert chunks == [b"event: message\n"], "what arrived before the failure is delivered"
    assert [type(e) for e in reports] == [httpx.ReadError]


@pytest.mark.asyncio
async def test_a_get_is_never_reported():
    reports: list[httpx.TransportError] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/idle":
            raise httpx.ReadTimeout("idle stream", request=request)
        return httpx.Response(200, stream=_FailingBody())

    async with _client(handler, reports) as client:
        with pytest.raises(httpx.ReadTimeout):
            await client.get("/idle")
        with pytest.raises(httpx.ReadError):
            async with client.stream("GET", "/stream") as response:
                async for _ in response.aiter_raw():
                    pass
    assert reports == []


@pytest.mark.asyncio
async def test_the_child_client_keeps_mcps_defaults():
    async with child_http_client(lambda e: None) as client:
        assert client.timeout == CHILD_TIMEOUT == httpx.Timeout(30.0, read=300.0)
        assert client.follow_redirects is False
        assert isinstance(client._transport, ReportingTransport)
