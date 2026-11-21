"""A loopback relay an agent's CLI calls in place of the model endpoint, which counts every model call the CLI makes
from the call's own response: its subagents', side calls', summaries' and retries' as much as its main loop's. The
images whose ``shared_sources.toml`` takes this folder copy this file beside their bridge, which imports it."""

from __future__ import annotations

import contextlib
import json
import secrets
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from agentenv_protocol.a2a_agent import Usage
from aiohttp import ClientError, ClientSession, ClientTimeout, web

# The routes whose POSTs are model calls: Chat Completions, Responses and Anthropic's Messages.
_CALL_ROUTES = ("/chat/completions", "/responses", "/messages")
_HOP_HEADERS = {"host", "content-length", "transfer-encoding", "connection", "keep-alive", "accept-encoding"}
_ANSWER_DROP_HEADERS = {"content-length", "transfer-encoding", "connection", "content-encoding"}
# How much of an answer is kept to read its usage from: a stream's usage comes last, so more is never needed in practice.
_KEPT_ANSWER_BYTES = 64 * 1024 * 1024
_MAX_REQUEST_BYTES = 256 * 1024 * 1024
# How long the relay, once its turn is done, lets the calls it is still forwarding finish.
_SHUTDOWN_S = 5.0
_COUNTS = ("input_tokens", "output_tokens", "total_tokens", "cache_read_tokens", "cache_write_tokens")


@dataclass
class Calls:
    """The model calls a relay forwarded: each answered call's usage, read as ``Usage.from_model_response`` reads one,
    and how many were answered without usage or refused. A refused call (an error status, or none at all) counts
    nothing."""

    answered: list[Usage] = field(default_factory=list)
    unreported: int = 0
    refused: int = 0

    def add(self, status: int | None, answer: bytes) -> None:
        if status is None or status >= 400:
            self.refused += 1
            return
        usage = answer_usage(answer)
        if usage is None:
            self.unreported += 1
        else:
            self.answered.append(usage)

    def usage(self, tool_call_count: int) -> Usage:
        """The calls' usage summed, with ``input_tokens`` counting the cache, and the calls without usage counted in
        ``unreported_calls``; no counts when no call had any."""
        details: dict[str, Any] = {}
        if self.answered or self.unreported:
            details["model_calls"] = len(self.answered) + self.unreported
        if self.unreported:
            details["unreported_calls"] = self.unreported
        if not self.answered:
            return Usage(tool_call_count=tool_call_count, provider_details=details)

        def total(name: str) -> int | None:
            values = [getattr(usage, name) for usage in self.answered if getattr(usage, name) is not None]
            return sum(values) if values else None

        flags = {usage.input_includes_cache for usage in self.answered}
        return Usage(tool_call_count=tool_call_count, **{name: total(name) for name in _COUNTS},
                     input_includes_cache=flags.pop() if len(flags) == 1 else None, provider_details=details)


def answer_usage(answer: bytes) -> Usage | None:
    """The usage an answer carried: a JSON body's ``usage``, or a stream's, from its last chunk with one (Chat
    Completions), its ``response.completed`` (Responses), or its ``message_start`` and ``message_delta`` (Anthropic).
    None when it carried none, or only zeros, as a gateway sends for a call it didn't count."""
    text = answer.decode("utf-8", "replace")
    usage: dict[str, Any] | None = None
    try:
        body = json.loads(text)
    except json.JSONDecodeError:
        for line in text.splitlines():
            if not line.startswith("data:"):
                continue
            try:
                event = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            kind = event.get("type")
            if kind == "message_start":
                usage = dict((event.get("message") or {}).get("usage") or {})
            elif kind == "message_delta" and isinstance(event.get("usage"), dict):
                usage = {**(usage or {}), **event["usage"]}
            elif kind in ("response.completed", "response.incomplete") and isinstance(event.get("response"), dict):
                usage = event["response"].get("usage") or usage
            elif isinstance(event.get("usage"), dict) and event["usage"]:
                usage = event["usage"]
    else:
        usage = body.get("usage") if isinstance(body, dict) else None
    if not isinstance(usage, dict):
        return None
    read = Usage.from_model_response(usage)
    if not any(getattr(read, name) for name in _COUNTS):
        return None
    return read


@asynccontextmanager
async def model_relay(upstream: str, calls: Calls, *, headers: Mapping[str, str] | None = None,
                      models: list[str] | None = None) -> AsyncIterator[str]:
    """Serve a relay to ``upstream``, the endpoint's API root, on loopback, and yield its root, which takes the
    endpoint's paths after it (``<root>/v1/chat/completions``). Its path names a token of its own, so a process left
    from another turn reaches no other turn's relay. Each model call it forwards is added to ``calls`` once its answer
    has ended, read to its end even when the CLI hung up. ``headers`` replace the CLI's own on every request, so the
    endpoint's credentials need never reach the CLI; without them, the CLI's are forwarded. ``models`` is the list a
    ``GET .../models`` is answered with, sparing a CLI that lists them the endpoint's whole catalog."""
    token = secrets.token_hex(16)
    async with ClientSession(timeout=ClientTimeout(total=None)) as client:
        async def forward(request: web.Request) -> web.StreamResponse:
            if request.match_info["token"] != token:
                raise web.HTTPNotFound()
            path = request.match_info["path"]
            if models is not None and request.method == "GET" and path.rstrip("/").endswith("models"):
                return web.json_response({"object": "list", "data": [{"id": model, "object": "model"} for model in models]})
            outgoing = ({k: v for k, v in request.headers.items() if k.lower() not in _HOP_HEADERS} if headers is None
                        else {**dict(headers), **({"content-type": request.headers["Content-Type"]}
                                                  if "Content-Type" in request.headers else {})})
            # Answered as sent, so its usage can be read.
            outgoing["accept-encoding"] = "identity"
            call = request.method == "POST" and path.rstrip("/").endswith(_CALL_ROUTES)
            status, kept = None, bytearray()
            try:
                async with client.request(request.method, f"{upstream}/{path}", params=request.query,
                                          data=await request.read(), headers=outgoing) as answer:
                    status = answer.status
                    response = web.StreamResponse(status=answer.status, headers={
                        name: value for name, value in answer.headers.items()
                        if name.lower() not in _ANSWER_DROP_HEADERS})
                    await response.prepare(request)
                    listening = True
                    async for chunk in answer.content.iter_any():
                        if call and len(kept) < _KEPT_ANSWER_BYTES:
                            kept.extend(chunk)
                        if listening:
                            try:
                                await response.write(chunk)
                            except (ConnectionResetError, ClientError):
                                listening = False
                    if listening:
                        with contextlib.suppress(ConnectionResetError):
                            await response.write_eof()
                    return response
            except ClientError as exc:
                if status is None:
                    raise web.HTTPBadGateway(text=f"model relay: {type(exc).__name__}") from exc
                raise
            finally:
                if call:
                    calls.add(status, bytes(kept))

        app = web.Application(client_max_size=_MAX_REQUEST_BYTES)
        app.router.add_route("*", "/{token}/{path:.*}", forward)
        runner = web.AppRunner(app, shutdown_timeout=_SHUTDOWN_S)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            yield f"http://127.0.0.1:{port}/{token}"
        finally:
            await runner.cleanup()
