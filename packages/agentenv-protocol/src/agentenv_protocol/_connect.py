"""Retry connections before the first request reaches the environment."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, nullcontext
from importlib import import_module
from math import inf, isfinite
from typing import Any

import anyio
import httpx

try:
    import sniffio
except ImportError:
    sniffio = None


def _runtime():
    # Trio installs sniffio; asyncio-only HTTPX installations can omit it.
    if sniffio is not None and sniffio.current_async_library() == "trio":
        return import_module("trio")
    return anyio


async def _send_attempt(client: httpx.AsyncClient, request: httpx.Request, deadline: float, runtime) -> httpx.Response:
    extensions = request.extensions
    previous = extensions.get("trace")
    stream = None

    async def trace(event, info):
        nonlocal deadline, stream
        if event.endswith((".connect_tcp.started", ".start_tls.started")) and isfinite(deadline):
            # httpcore passes these arguments to each handshake. TCP and TLS share this attempt's budget.
            info["timeout"] = max(0, deadline - runtime.current_time())
        elif event.endswith((".connect_tcp.complete", ".start_tls.complete")):
            stream = info["return_value"]
        elif event.endswith(".start_tls.failed") and stream is not None and not isinstance(info["exception"], Exception):
            # Cancellation can interrupt TLS before the pool takes ownership of the socket.
            with runtime.CancelScope(shield=True):
                await stream.aclose()
        elif event.endswith(".send_request_headers.started") and info["request"].method != b"CONNECT":
            # A followed redirect has its own connection. Only the original hop is eligible for retry.
            deadline = inf
        if previous is not None:
            await previous(event, info)

    request.extensions = {**extensions, "trace": trace}
    try:
        return await client.send(request, stream=True)
    finally:
        request.extensions = extensions


@asynccontextmanager
async def open_response(client: httpx.AsyncClient, request: httpx.Request) -> AsyncIterator[httpx.Response]:
    runtime = _runtime()
    timeout = request.extensions["timeout"]["connect"]
    deadline = runtime.current_effective_deadline()
    if timeout is not None:
        deadline = min(deadline, runtime.current_time() + timeout)

    delays = (0.5, 1.0, None)
    for attempt, delay in enumerate(delays):
        now = runtime.current_time()
        remaining = max(0, deadline - now)
        share = remaining / (len(delays) - attempt)
        attempt_deadline = deadline
        if delay is not None and share + delay < remaining:
            attempt_deadline = now + share
        try:
            response = await _send_attempt(client, request, attempt_deadline, runtime)
            break
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            # HTTPX attaches the failing hop's request. A redirect means the original was sent.
            if exc.request is not request or delay is None or runtime.current_time() + delay >= deadline:
                raise
            await runtime.sleep(delay)
            if runtime.current_time() >= deadline:
                raise
    try:
        yield response
    finally:
        with runtime.CancelScope(shield=True):
            await response.aclose()


async def request(client: httpx.AsyncClient, method: str, url: str, *, timeout: float, total_timeout: float | None,
                  **kwargs: Any) -> httpx.Response:
    outgoing = client.build_request(method, url, timeout=timeout, **kwargs)
    with (_runtime().move_on_after(total_timeout) if total_timeout is not None else nullcontext()):
        async with open_response(client, outgoing) as response:
            await response.aread()
            return response
    raise httpx.TimeoutException(f"{method} did not finish within {total_timeout}s", request=outgoing)
