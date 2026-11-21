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
# How much of a JSON answer is kept to read its usage from; a stream's is read as it passes, whatever its length.
_KEPT_BODY_BYTES = 64 * 1024 * 1024
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

    def add(self, status: int | None, usage: Usage | None) -> None:
        """A call that ended with ``status`` (None when it had none), and the usage its answer carried."""
        if status is None or status >= 400:
            self.refused += 1
            return
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


class AnswerReader:
    """Reads an answer's usage as its bytes pass: a stream's events one line at a time, keeping only the usage they
    carry (its last chunk with one in Chat Completions, its ``response.completed`` in Responses, its
    ``message_start`` and ``message_delta`` in Anthropic's Messages), or a JSON body's ``usage``."""

    def __init__(self) -> None:
        self._stream: bool | None = None
        self._line = bytearray()
        self._body = bytearray()
        self._usage: dict[str, Any] | None = None

    def feed(self, chunk: bytes) -> None:
        if self._stream is None:
            if not chunk.strip():
                return
            self._stream = chunk.lstrip().startswith((b"data:", b"event:", b":"))
        if not self._stream:
            if len(self._body) < _KEPT_BODY_BYTES:
                self._body.extend(chunk)
            return
        *lines, rest = (bytes(self._line) + chunk).split(b"\n")
        for line in lines:
            self._event(line)
        self._line = bytearray(rest[:_KEPT_BODY_BYTES])

    def _event(self, line: bytes) -> None:
        if not line.startswith(b"data:"):
            return
        try:
            event = json.loads(line[5:].strip())
        except (json.JSONDecodeError, UnicodeDecodeError):
            return
        if not isinstance(event, dict):
            return
        kind = event.get("type")
        if kind == "message_start":
            self._usage = dict((event.get("message") or {}).get("usage") or {})
        elif kind == "message_delta" and isinstance(event.get("usage"), dict):
            self._usage = {**(self._usage or {}), **event["usage"]}
        elif kind in ("response.completed", "response.incomplete") and isinstance(event.get("response"), dict):
            self._usage = event["response"].get("usage") or self._usage
        elif isinstance(event.get("usage"), dict) and event["usage"]:
            self._usage = event["usage"]

    def usage(self) -> Usage | None:
        """The answer's usage, None when it carried none, or only zeros, as a gateway sends for a call it didn't
        count."""
        if self._stream:
            self._event(bytes(self._line))
            usage = self._usage
        else:
            try:
                body = json.loads(bytes(self._body))
            except (json.JSONDecodeError, UnicodeDecodeError):
                body = None
            usage = body.get("usage") if isinstance(body, dict) else None
        if not isinstance(usage, dict):
            return None
        read = Usage.from_model_response(usage)
        return read if any(getattr(read, name) for name in _COUNTS) else None


def answer_usage(answer: bytes) -> Usage | None:
    """The usage a whole answer carried, as ``AnswerReader`` reads it."""
    reader = AnswerReader()
    reader.feed(answer)
    return reader.usage()


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
            status, reader = None, AnswerReader()
            try:
                async with client.request(request.method, f"{upstream}/{path}", params=request.query,
                                          data=await request.read(), headers=outgoing) as answer:
                    status = answer.status
                    response = web.StreamResponse(status=answer.status, headers={
                        name: value for name, value in answer.headers.items()
                        if name.lower() not in _ANSWER_DROP_HEADERS})
                    listening = True
                    try:
                        await response.prepare(request)
                    except (ConnectionResetError, ClientError):
                        listening = False
                    async for chunk in answer.content.iter_any():
                        if call:
                            reader.feed(chunk)
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
                    calls.add(status, reader.usage())

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
