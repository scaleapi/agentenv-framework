"""A fake Anthropic Messages endpoint, so the real Claude Code CLI runs with no model key: ``POST /v1/messages``
is answered by a responder, streamed as the Messages API's server-sent events, and every request is recorded.
A fault queued with ``fail_next`` replaces the next answer: an HTTP error, a reset or truncated stream, or one that
stalls."""

from __future__ import annotations

import json
import socket
import struct
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


@dataclass(frozen=True)
class Text:
    text: str


@dataclass(frozen=True)
class ToolUse:
    name: str
    input: dict


@dataclass(frozen=True)
class Fault:
    """``status``: answer with that HTTP error; ``reset``/``truncate``: start the stream, then drop the connection
    hard or end it without its last events; ``stall``: start it, then send nothing for ``seconds``."""

    kind: str
    status: int = 500
    seconds: float = 0.0


@dataclass
class Seen:
    path: str
    headers: dict[str, str]
    body: dict


Responder = Callable[[dict], Text | ToolUse]


def answer(text: str) -> Responder:
    """Every request answered with ``text``."""
    return lambda body: Text(text)


def tool_then_answer(tool_suffix: str, arguments: dict, *, prefix: str = "") -> Responder:
    """Call the request's tool whose name ends in ``tool_suffix``, then answer with ``prefix`` and the tool result's
    text; answer ``prefix`` alone to a request that offers no such tool."""

    def respond(body: dict) -> Text | ToolUse:
        result = _tool_result_text(body)
        if result is not None:
            return Text(f"{prefix}{result}")
        tool = next((t["name"] for t in body.get("tools") or [] if t.get("name", "").endswith(tool_suffix)), None)
        return ToolUse(tool, arguments) if tool else Text(prefix or "no tool")

    return respond


def delegate_then_answer(cue: str, peer: str, prompt: str, reply: str) -> Responder:
    """For a lead agent and its peer, both calling this endpoint: a request whose messages hold ``cue``, the lead's,
    calls ``peer_send_message`` to send ``peer`` the ``prompt``, then answers with the tool result's text; every other
    request, the peer's, is answered with ``reply``."""

    def respond(body: dict) -> Text | ToolUse:
        result = _tool_result_text(body)
        if result is not None:
            return Text(result)
        if cue not in json.dumps(body.get("messages") or []):
            return Text(reply)
        tool = next(t["name"] for t in body.get("tools") or [] if t.get("name", "").endswith("__peer_send_message"))
        return ToolUse(tool, {"name": peer, "prompt": prompt})

    return respond


def _tool_result_text(body: dict) -> str | None:
    """The text of a tool result in the latest user turn, or None. The CLI may follow it with system messages."""
    user = next((m for m in reversed(body.get("messages") or []) if m.get("role") == "user"), None)
    content = user.get("content") if user else None
    for block in content if isinstance(content, list) else ():
        if isinstance(block, dict) and block.get("type") == "tool_result":
            inner = block.get("content")
            if isinstance(inner, list):
                return " ".join(part.get("text", "") for part in inner if isinstance(part, dict))
            return str(inner)
    return None


@dataclass
class FakeAnthropic:
    host: str
    respond: Responder = field(default_factory=lambda: answer("pong"))
    # The latest requests only: a CLI that loops resends its whole history each call.
    seen: deque[Seen] = field(default_factory=lambda: deque(maxlen=200))
    faults: list[Fault] = field(default_factory=list)
    _server: ThreadingHTTPServer | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _released: threading.Event = field(default_factory=threading.Event)

    def __enter__(self) -> FakeAnthropic:
        self._server = ThreadingHTTPServer((self.host, 0), _handler(self))
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc) -> None:
        self._released.set()
        self._server.shutdown()
        self._server.server_close()

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self._server.server_port}"

    def fail_next(self, *faults: Fault) -> None:
        with self._lock:
            self.faults.extend(faults)

    def messages(self) -> list[Seen]:
        """The model calls, not the token counts."""
        return [seen for seen in self.seen if seen.path == "/v1/messages"]

    def tool_results(self) -> list[str]:
        """The text of each tool result the CLI sent back."""
        return [text for seen in self.messages() if (text := _tool_result_text(seen.body)) is not None]

    def _next_fault(self) -> Fault | None:
        with self._lock:
            return self.faults.pop(0) if self.faults else None


def _handler(fake: FakeAnthropic) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args) -> None:
            pass

        def do_POST(self) -> None:
            path = self.path.split("?", 1)[0]
            body = json.loads(self.rfile.read(int(self.headers.get("content-length") or 0)) or b"{}")
            fake.seen.append(Seen(path, {k.lower(): v for k, v in self.headers.items()}, body))
            if path == "/v1/messages/count_tokens":
                return self._json(200, {"input_tokens": 10})
            if path != "/v1/messages":
                return self._json(404, {"type": "error", "error": {"type": "not_found_error", "message": path}})
            fault = fake._next_fault()
            if fault is not None and fault.kind == "status":
                return self._json(fault.status, {"type": "error", "error": {"type": _error_type(fault.status),
                                                                             "message": f"injected {fault.status}"}})
            reply = fake.respond(body)
            if not body.get("stream"):
                return self._json(200, _message(body, reply))
            self._stream(body, reply, fault)

        def _json(self, status: int, payload: dict) -> None:
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _stream(self, body: dict, reply: Text | ToolUse, fault: Fault | None) -> None:
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("cache-control", "no-cache")
            self.send_header("connection", "close")
            self.end_headers()
            events = _events(body, reply)
            if fault is not None:
                self._send(events[:2])
                if fault.kind == "reset":
                    # SO_LINGER 0 makes close send a reset rather than a clean end.
                    self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
                elif fault.kind == "stall":
                    fake._released.wait(fault.seconds)
                self.close_connection = True
                return
            self._send(events)
            self.close_connection = True

        def _send(self, events: list[tuple[str, dict]]) -> None:
            for name, data in events:
                self.wfile.write(f"event: {name}\ndata: {json.dumps(data)}\n\n".encode())
                self.wfile.flush()

    return Handler


def _error_type(status: int) -> str:
    return {400: "invalid_request_error", 401: "authentication_error", 404: "not_found_error",
            429: "rate_limit_error", 529: "overloaded_error"}.get(status, "api_error")


def _message(body: dict, reply: Text | ToolUse) -> dict:
    content = ([{"type": "text", "text": reply.text}] if isinstance(reply, Text)
               else [{"type": "tool_use", "id": f"toolu_{time.monotonic_ns()}", "name": reply.name, "input": reply.input}])
    return {"id": f"msg_{time.monotonic_ns()}", "type": "message", "role": "assistant", "model": body.get("model"),
            "content": content, "stop_reason": "end_turn" if isinstance(reply, Text) else "tool_use",
            "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 5}}


def _events(body: dict, reply: Text | ToolUse) -> list[tuple[str, dict]]:
    message = {**_message(body, reply), "content": [], "stop_reason": None,
               "usage": {"input_tokens": 10, "output_tokens": 1}}
    if isinstance(reply, Text):
        block = {"type": "text", "text": ""}
        delta = {"type": "text_delta", "text": reply.text}
        stop = "end_turn"
    else:
        block = {"type": "tool_use", "id": f"toolu_{time.monotonic_ns()}", "name": reply.name, "input": {}}
        delta = {"type": "input_json_delta", "partial_json": json.dumps(reply.input)}
        stop = "tool_use"
    return [
        ("message_start", {"type": "message_start", "message": message}),
        ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": block}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": delta}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None},
                           "usage": {"output_tokens": 5}}),
        ("message_stop", {"type": "message_stop"}),
    ]
