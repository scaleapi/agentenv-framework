"""Retry connections before the first request reaches the environment."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import anyio
import httpx


@asynccontextmanager
async def open_response(client: httpx.AsyncClient, request: httpx.Request) -> AsyncIterator[httpx.Response]:
    for delay in (0.5, 1.0):
        try:
            response = await client.send(request, stream=True)
            break
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            # HTTPX attaches the failing hop's request. A redirect means the original was sent.
            if exc.request is not request or anyio.current_time() + delay >= anyio.current_effective_deadline():
                raise
            await anyio.sleep(delay)
    else:
        response = await client.send(request, stream=True)
    try:
        yield response
    finally:
        with anyio.CancelScope(shield=True):
            await response.aclose()


async def request(client: httpx.AsyncClient, method: str, url: str, *, timeout: float, **kwargs: Any) -> httpx.Response:
    outgoing = client.build_request(method, url, timeout=timeout, **kwargs)
    with anyio.move_on_after(timeout):
        async with open_response(client, outgoing) as response:
            await response.aread()
            return response
    raise httpx.TimeoutException(f"{method} did not finish within {timeout}s", request=outgoing)
