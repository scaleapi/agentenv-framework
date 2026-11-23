"""A loopback relay an agent's CLI calls in place of the model endpoint, which counts every model call the CLI makes
from the call's own response: its subagents', side calls', summaries' and retries' as much as its main loop's. The
images whose ``shared_sources.toml`` takes this folder copy this file beside their bridge, which imports it."""

from __future__ import annotations

import contextlib
import json
import logging
import secrets
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from agentenv_protocol.a2a_agent import ModelUsage, Usage
from aiohttp import ClientError, ClientSession, ClientTimeout, web

logger = logging.getLogger(__name__)

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


class Refusal(Exception):
    """A model call the relay answers itself, with this reason, rather than forward."""


@dataclass
class Calls:
    """The model calls a relay forwarded: each answered call's model, as its request named it, and usage, read as
    ``Usage.from_model_response`` reads one, and how many were answered without usage or refused. A refused call (an
    error status, or none at all) counts nothing."""

    answered: list[tuple[str | None, Usage]] = field(default_factory=list)
    unreported: int = 0
    refused: int = 0

    def add(self, status: int | None, usage: Usage | None, model: str | None = None) -> None:
        """A call that ended with ``status`` (None when it had none), the usage its answer carried and the model its
        request named."""
        if status is None or status >= 400:
            self.refused += 1
            return
        if usage is None:
            self.unreported += 1
        else:
            self.answered.append((model, usage))

    def usage(self, tool_call_count: int, model: str | None = None) -> Usage:
        """The calls' usage summed, with ``input_tokens`` counting the cache, and the calls without usage counted in
        ``unreported_calls``; no counts when no call had any. ``models`` holds them by the model each named, a call
        that named none counted as ``model``'s, the one the turn was configured with."""
        details: dict[str, Any] = {}
        if self.answered or self.unreported:
            details["model_calls"] = len(self.answered) + self.unreported
        if self.unreported:
            details["unreported_calls"] = self.unreported
        if not self.answered:
            return Usage(tool_call_count=tool_call_count, provider_details=details)
        usages = [usage for _, usage in self.answered]

        def total(name: str) -> int | None:
            values = [getattr(usage, name) for usage in usages if getattr(usage, name) is not None]
            return sum(values) if values else None

        flags = {usage.input_includes_cache for usage in usages}
        return Usage(tool_call_count=tool_call_count, **{name: total(name) for name in _COUNTS},
                     input_includes_cache=flags.pop() if len(flags) == 1 else None, models=self._by_model(model),
                     provider_details=details)

    def _by_model(self, configured: str | None) -> tuple[ModelUsage, ...]:
        """The answered calls' counts by model; none when a call named no model and the turn has none configured."""
        by_model: dict[str, list[Usage]] = {}
        for named, usage in self.answered:
            if (name := named or configured) is None:
                return ()
            by_model.setdefault(name, []).append(usage)
        return tuple(ModelUsage.of(name, of) for name, of in by_model.items())


class AnswerReader:
    """Reads an answer's usage as its bytes pass: a stream's events one line at a time, keeping only the usage they
    carry (its last chunk with one in Chat Completions, its ``response.completed`` in Responses, its
    ``message_start`` and ``message_delta`` in Anthropic's Messages), or a JSON body's ``usage``. ``stream`` says which
    the answer is, as its content type does; when None, the whole answer is kept and judged by how it begins."""

    def __init__(self, stream: bool | None = None) -> None:
        self._stream = stream
        self._line = bytearray()
        self._body = bytearray()
        self._usage: dict[str, Any] | None = None

    def feed(self, chunk: bytes) -> None:
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
        if self._stream is None and self._body.lstrip().startswith((b"data:", b"event:", b":")):
            self._stream, body, self._body = True, bytes(self._body), bytearray()
            self.feed(body)
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


def _request_model(body: bytes) -> str | None:
    """The model a call's request body names, as its JSON's ``model``; None for a body that names none."""
    try:
        model = json.loads(body).get("model")
    except (ValueError, AttributeError):
        return None
    return model if isinstance(model, str) and model else None


def answer_usage(answer: bytes) -> Usage | None:
    """The usage a whole answer carried, as ``AnswerReader`` reads it."""
    reader = AnswerReader()
    reader.feed(answer)
    return reader.usage()


@asynccontextmanager
async def model_relay(upstream: str, calls: Calls, *, headers: Mapping[str, str] | None = None,
                      models: list[str] | None = None,
                      rewrite: Callable[[bytes], Awaitable[bytes]] | None = None) -> AsyncIterator[str]:
    """Serve a relay to ``upstream``, the endpoint's API root, on loopback, and yield its root, which takes the
    endpoint's paths after it (``<root>/v1/chat/completions``). Its path names a token of its own, so a process left
    from another turn reaches no other turn's relay. Each model call it forwards is added to ``calls`` once its answer
    has ended, read to its end even when the CLI hung up. ``headers`` replace the CLI's own on every request, so the
    endpoint's credentials need never reach the CLI; without them, the CLI's are forwarded. ``models`` is the list a
    ``GET .../models`` is answered with, sparing a CLI that lists them the endpoint's whole catalog. ``rewrite`` takes
    each model call's request body and gives the one to forward; what it raises is the call's answer, a 400 with a
    ``Refusal``'s reason or the error's name, which no CLI retries as it would a 500, and the call goes no further."""
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
            status, reader, body = None, AnswerReader(), b""
            try:
                body = await request.read()
                if call and rewrite is not None:
                    try:
                        body = await rewrite(body)
                    except Exception as exc:
                        status = 400
                        reason = str(exc) if isinstance(exc, Refusal) else f"model relay: {type(exc).__name__}: {exc}"
                        logger.warning("model relay refused a call: %s", reason)
                        return web.json_response({"error": {"message": reason, "type": "invalid_request_error"}},
                                                 status=status)
                async with client.request(request.method, f"{upstream}/{path}", params=request.query,
                                          data=body, headers=outgoing) as answer:
                    status = answer.status
                    reader = AnswerReader(stream=answer.content_type == "text/event-stream")
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
                    calls.add(status, reader.usage(), _request_model(body))

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
