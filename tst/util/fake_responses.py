"""A fake OpenAI Responses API endpoint, so an agent whose CLI speaks it (Codex) runs with no model key: ``POST
.../responses`` is answered by a responder as server-sent events, a message or a function call then the response's
completion with ``USAGE``, and every request is recorded."""

import json
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Final

from tst.util.fake_openai import Seen, Text, ToolCall

Responder = Callable[[dict], Text | ToolCall]

USAGE: Final[dict] = {"input_tokens": 100, "input_tokens_details": {"cached_tokens": 0}, "output_tokens": 10,
                      "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 110}


def answer(text: str) -> Responder:
    """Every request answered with ``text``."""
    return lambda body: Text(text)


def tool_then_answer(tool: str, arguments: dict, *, prefix: str = "") -> Responder:
    """Call the request's tool named ``tool``, then answer with ``prefix`` and the call's output; answer ``prefix``
    alone to a request that offers no such tool."""

    def respond(body: dict) -> Text | ToolCall:
        result = tool_output(body)
        if result is not None:
            return Text(prefix + result)
        names = {spec.get("name") for spec in body.get("tools") or [] if isinstance(spec, dict)}
        return ToolCall(tool, arguments) if tool in names else Text(prefix)

    return respond


def tool_output(body: dict) -> str | None:
    """The output of the latest function call since the request's last user message, else None."""
    found = None
    for item in body.get("input") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type", "message") == "message" and item.get("role") == "user":
            found = None
        elif item.get("type") == "function_call_output":
            output = item.get("output")
            found = ("".join(part.get("text", "") for part in output if isinstance(part, dict))
                     if isinstance(output, list) else output if isinstance(output, str) else json.dumps(output))
    return found


@dataclass(slots=True)
class FakeResponses:
    host: str = "127.0.0.1"
    respond: Responder = field(default_factory=lambda: answer("ok"))
    seen: list[Seen] = field(default_factory=list)
    _server: ThreadingHTTPServer | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self._server.server_address[1]}"

    def calls(self) -> list[Seen]:
        return [seen for seen in self.seen if seen.path.split("?")[0].endswith("/responses")]

    def tool_results(self) -> list[str]:
        return [text for seen in self.calls() if (text := tool_output(seen.body)) is not None]

    def __enter__(self) -> "FakeResponses":
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:
                pass

            def do_GET(self) -> None:
                self._send(404, {"error": {"message": f"no route {self.path}"}})

            def do_POST(self) -> None:
                length = int(self.headers.get("content-length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                fake.seen.append(Seen(self.path, {k.lower(): v for k, v in self.headers.items()}, body))
                if not self.path.split("?")[0].endswith("/responses"):
                    self._send(404, {"error": {"message": f"no route {self.path}"}})
                    return
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.end_headers()
                try:
                    for event in _events(fake.respond(body)):
                        self.wfile.write(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode())
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def _send(self, status: int, payload: dict) -> None:
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                try:
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self._server = ThreadingHTTPServer((self.host, 0), Handler)
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc) -> None:
        self._server.shutdown()
        self._server.server_close()


def _events(reply: Text | ToolCall) -> list[dict]:
    response_id = f"resp_{uuid.uuid4().hex[:12]}"
    if isinstance(reply, Text):
        item = {"type": "message", "role": "assistant", "id": f"msg_{uuid.uuid4().hex[:12]}",
                "content": [{"type": "output_text", "text": reply.text}]}
    else:
        item = {"type": "function_call", "id": f"fc_{uuid.uuid4().hex[:12]}", "call_id": f"call_{uuid.uuid4().hex[:12]}",
                "name": reply.name, "arguments": json.dumps(reply.arguments)}
    return [{"type": "response.created", "response": {"id": response_id}},
            {"type": "response.output_item.done", "output_index": 0, "item": item},
            {"type": "response.completed", "response": {"id": response_id, "usage": USAGE}}]
