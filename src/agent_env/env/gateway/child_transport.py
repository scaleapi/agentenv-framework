"""The httpx transport under the gateway's child MCP sessions: one that reports request failures.

mcp's streamable HTTP client swallows a transport error on a request it has in flight (``_handle_sse_response``
logs "SSE stream ended" and reconnects only when the stream had carried an event id), so the request never
resolves and its caller waits out the whole tool-call timeout. The gateway cannot see inside the client, but it
owns the httpx client the transport uses: this transport reports every ``httpx.TransportError`` raised while a POST
is sent or its body read, so the gateway can retire that child session generation at once. The session's receive
loop then fails every pending request with CONNECTION_CLOSED, and the next call reopens.

Only POSTs are reported. The standalone GET stream idles past its read timeout by design, and the DELETE at
session close fails harmlessly against a child that is already gone.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable

import httpx

#: mcp's own defaults for a client under a streamable HTTP session (``create_mcp_http_client``): 30 s to connect,
#: write and wait for a pool slot, 300 s to read, because a server may hold a response stream open.
CHILD_TIMEOUT = httpx.Timeout(30.0, read=300.0)

Report = Callable[[httpx.TransportError], None]


class ReportingTransport(httpx.AsyncBaseTransport):
    """``inner`` with every POST's transport failure, at send or while its body streams, handed to ``report``
    before it propagates."""

    def __init__(self, inner: httpx.AsyncBaseTransport, report: Report):
        self._inner = inner
        self._report = report

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        watched = request.method == "POST"
        try:
            response = await self._inner.handle_async_request(request)
        except httpx.TransportError as e:
            if watched:
                self._report(e)
            raise
        if watched:
            response.stream = _ReportingStream(response.stream, self._report)
        return response

    async def aclose(self) -> None:
        await self._inner.aclose()


class _ReportingStream(httpx.AsyncByteStream):
    def __init__(self, inner: httpx.AsyncByteStream, report: Report):
        self._inner = inner
        self._report = report

    async def __aiter__(self) -> AsyncIterator[bytes]:
        try:
            async for chunk in self._inner:
                yield chunk
        except httpx.TransportError as e:
            self._report(e)
            raise

    async def aclose(self) -> None:
        await self._inner.aclose()


def child_http_client(report: Report) -> httpx.AsyncClient:
    """The client a child session runs over: mcp's defaults, redirects left to the transport's own origin rule, on a
    reporting transport."""
    return httpx.AsyncClient(transport=ReportingTransport(httpx.AsyncHTTPTransport(), report), timeout=CHILD_TIMEOUT)
