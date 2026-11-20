"""A fake OpenAI Responses API endpoint, so the real Codex CLI runs with no model key: ``POST <base>/responses`` is
answered by a responder, streamed as the Responses API's server-sent events, and every request is recorded. The CLI
reaches it with LITELLM_BASE_URL set to ``url``, as the Codex agent writes its provider (``<base>/openai/v1``)."""

from __future__ import annotations

import json
import threading
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tst.util.fake_anthropic import Seen, Text, answer

SHELL_TOOLS = ("exec_command", "shell_command", "shell")


@dataclass(frozen=True)
class Call:
    name: str
    arguments: dict


Responder = Callable[[dict], Text | Call]


def _tool_output(body: dict) -> str | None:
    """The output of the tool call the request's last input item answers, or None."""
    items = body.get("input") or []
    last = items[-1] if items and isinstance(items[-1], dict) else {}
    if last.get("type") not in ("function_call_output", "custom_tool_call_output"):
        return None
    output = last.get("output")
    if isinstance(output, list):
        return "".join(part.get("text", "") for part in output if isinstance(part, dict))
    return str(output)


def shell_then_answer(command: str, *, prefix: str = "") -> Responder:
    """Run ``command`` with the shell tool the request offers, then answer with ``prefix`` and its output; answer
    ``prefix`` alone to a request that offers none, as for a model whose catalog entry runs code mode."""

    def respond(body: dict) -> Text | Call:
        output = _tool_output(body)
        if output is not None:
            return Text(f"{prefix}{output}")
        names = {tool.get("name") for tool in body.get("tools") or [] if isinstance(tool, dict)}
        tool = next((name for name in SHELL_TOOLS if name in names), None)
        if tool is None:
            return Text(prefix or "no shell tool")
        arguments = ({"cmd": command} if tool == "exec_command" else {"command": command} if tool == "shell_command"
                     else {"command": ["bash", "-lc", command]})
        return Call(tool, arguments)

    return respond


@dataclass
class FakeResponses:
    host: str
    respond: Responder = field(default_factory=lambda: answer("pong"))
    seen: deque[Seen] = field(default_factory=lambda: deque(maxlen=200))
    _server: ThreadingHTTPServer | None = None

    def __enter__(self) -> FakeResponses:
        self._server = ThreadingHTTPServer((self.host, 0), _handler(self))
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc) -> None:
        self._server.shutdown()
        self._server.server_close()

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self._server.server_port}"

    def requests(self) -> list[Seen]:
        """The model calls."""
        return [seen for seen in self.seen if seen.path.endswith("/responses")]

    def tool_results(self) -> list[str]:
        """The output of each tool call the CLI sent back."""
        return [output for seen in self.requests() if (output := _tool_output(seen.body)) is not None]


def _item(reply: Text | Call) -> dict:
    if isinstance(reply, Text):
        return {"type": "message", "role": "assistant", "id": f"msg_{uuid.uuid4().hex}",
                "content": [{"type": "output_text", "text": reply.text}]}
    return {"type": "function_call", "id": f"fc_{uuid.uuid4().hex}", "call_id": f"call_{uuid.uuid4().hex}",
            "name": reply.name, "arguments": json.dumps(reply.arguments)}


def _handler(fake: FakeResponses) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args) -> None:
            pass

        def do_GET(self) -> None:
            self._json(200, {"object": "list", "data": []} if self.path.split("?", 1)[0].endswith("/models")
                       else {"error": {"message": self.path}})

        def do_POST(self) -> None:
            path = self.path.split("?", 1)[0]
            body = json.loads(self.rfile.read(int(self.headers.get("content-length") or 0)) or b"{}")
            fake.seen.append(Seen(path, {k.lower(): v for k, v in self.headers.items()}, body))
            if not path.endswith("/responses"):
                return self._json(404, {"error": {"message": path}})
            item, rid = _item(fake.respond(body)), f"resp_{uuid.uuid4().hex}"
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("cache-control", "no-cache")
            self.send_header("connection", "close")
            self.end_headers()
            for event in ({"type": "response.created", "response": {"id": rid}},
                          {"type": "response.output_item.added", "item": item},
                          {"type": "response.output_item.done", "item": item},
                          {"type": "response.completed", "response": {"id": rid, "usage": {
                              "input_tokens": 10, "input_tokens_details": {"cached_tokens": 0}, "output_tokens": 5,
                              "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 15}}}):
                self.wfile.write(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode())
                self.wfile.flush()
            self.close_connection = True

        def _json(self, status: int, payload: dict) -> None:
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler
