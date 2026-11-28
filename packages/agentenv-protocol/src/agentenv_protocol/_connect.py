"""Bound connection attempts for an opted-in HTTPX extension call."""

import asyncio
from importlib import import_module
from math import inf, isfinite

import httpx

try:
    import sniffio
except ImportError:
    sniffio = None


async def send(client: httpx.AsyncClient, request: httpx.Request) -> httpx.Response:
    # Match HTTPX's active backend, including a Trio guest task running on an asyncio loop.
    trio = import_module("trio") if sniffio is not None and sniffio.current_async_library() == "trio" else None
    if trio is None:
        now, sleep = asyncio.get_running_loop().time, asyncio.sleep
    else:
        now, sleep = trio.current_time, trio.sleep

    connect_timeout = request.extensions["timeout"]["connect"]
    deadline = inf if connect_timeout is None else now() + connect_timeout
    sent = False
    extensions = request.extensions
    previous = extensions.get("trace")

    async def trace(event, info):
        nonlocal sent, stream
        if event.endswith((".connect_tcp.started", ".start_tls.started")) and not sent and isfinite(attempt_deadline):
            # TCP and TLS share one attempt's budget, including a TLS handshake through a CONNECT proxy.
            info["timeout"] = max(0, attempt_deadline - now())
        elif event.endswith((".connect_tcp.complete", ".start_tls.complete")):
            stream = info["return_value"]
        elif event.endswith(".start_tls.failed") and stream is not None and not isinstance(info["exception"], Exception):
            # httpcore hasn't taken ownership of the handshake socket when the caller cancels it.
            if trio is None:
                await asyncio.shield(stream.aclose())
            else:
                with trio.CancelScope(shield=True):
                    await stream.aclose()
        elif event.endswith(".send_request_headers.started") and info["request"].method != b"CONNECT":
            sent = True
        if previous is not None:
            await previous(event, info)

    request.extensions = {**extensions, "trace": trace}
    try:
        delays = (0.5, 1.0, None)
        for attempt, delay in enumerate(delays):
            stream = None
            started = now()
            remaining = max(0, deadline - started)
            share = remaining / (len(delays) - attempt)
            attempt_deadline = started + share if delay is not None and share + delay < remaining else deadline
            try:
                return await client.send(request)
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                if sent or exc.request is not request or delay is None or now() + delay >= deadline:
                    raise
                await sleep(delay)
                if now() >= deadline:
                    raise
    finally:
        request.extensions = extensions
