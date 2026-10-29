"""A fake OpenAI Chat Completions endpoint, so an agent that calls the model through LiteLLM runs with no model key:
``POST .../chat/completions`` is answered by a responder, as one JSON completion or, to a request that asks for a
stream, as server-sent chunks, and every request is recorded. A fault queued with ``fail_next`` replaces the next
answer: an HTTP error, or an answer that stalls."""

import json
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Final


@dataclass(frozen=True, slots=True)
class Text:
    text: str


@dataclass(frozen=True, slots=True)
class ToolCall:
    name: str
    arguments: dict


@dataclass(frozen=True, slots=True)
class Fault:
    """``status``: answer with that HTTP error; ``stall``: send nothing for ``seconds``, then the answer."""

    kind: str
    status: int = 500
    seconds: float = 0.0


@dataclass(slots=True)
class Seen:
    path: str
    headers: dict[str, str]
    body: dict


Responder = Callable[[dict], Text | ToolCall]

USAGE: Final[dict[str, int]] = {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}


def answer(text: str) -> Responder:
    """Every request answered with ``text``."""
    return lambda body: Text(text)


def tool_then_answer(tool_suffix: str, arguments: dict, *, prefix: str = "") -> Responder:
    """Call the request's tool whose name ends in ``tool_suffix``, then answer with ``prefix`` and the tool result's
    text; answer ``prefix`` alone to a request that offers no such tool."""

    def respond(body: dict) -> Text | ToolCall:
        result = tool_result_text(body)
        if result is not None:
            return Text(prefix + result)
        names = [tool["function"]["name"] for tool in body.get("tools") or []]
        name = next((name for name in names if name.endswith(tool_suffix)), None)
        return ToolCall(name, arguments) if name else Text(prefix)

    return respond


def tool_result_text(body: dict) -> str | None:
    """The text of the last message when it is a tool result, else None."""
    messages = body.get("messages") or []
    if not messages or messages[-1].get("role") != "tool":
        return None
    content = messages[-1].get("content")
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content or ""


@dataclass(slots=True)
class FakeOpenAI:
    host: str = "127.0.0.1"
    respond: Responder = field(default_factory=lambda: answer("ok"))
    seen: list[Seen] = field(default_factory=list)
    _faults: deque = field(default_factory=deque)
    _server: ThreadingHTTPServer | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self._server.server_address[1]}"

    def fail_next(self, fault: Fault, times: int = 1) -> None:
        self._faults.extend([fault] * times)

    def completions(self) -> list[Seen]:
        return [seen for seen in self.seen if seen.path.endswith("/chat/completions")]

    def tool_results(self) -> list[str]:
        return [text for seen in self.completions() if (text := tool_result_text(seen.body)) is not None]

    def __enter__(self) -> "FakeOpenAI":
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:
                pass

            def do_POST(self) -> None:
                length = int(self.headers.get("content-length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                fake.seen.append(Seen(self.path, {k.lower(): v for k, v in self.headers.items()}, body))
                if not self.path.endswith("/chat/completions"):
                    self._send(404, {"error": {"message": f"no route {self.path}"}})
                    return
                fault = fake._faults.popleft() if fake._faults else None
                if fault is not None and fault.kind == "status":
                    self._send(fault.status, {"error": {"message": f"fake {fault.status}", "type": "fake_error"}})
                    return
                if fault is not None and fault.kind == "stall":
                    time.sleep(fault.seconds)
                completion = _completion(body, fake.respond(body))
                if body.get("stream"):
                    self._stream(completion)
                else:
                    self._send(200, completion)

            def _stream(self, completion: dict) -> None:
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.end_headers()
                try:
                    for chunk in _chunks(completion):
                        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    self.wfile.write(b"data: [DONE]\n\n")
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


def _chunks(completion: dict) -> list[dict]:
    """``completion`` as the chunks a stream sends: its message whole as one delta, its finish, then its usage."""
    (choice,) = completion["choices"]
    message = choice["message"]
    delta = {"role": "assistant", "content": message["content"]}
    if "tool_calls" in message:
        delta["tool_calls"] = [{"index": index, **call} for index, call in enumerate(message["tool_calls"])]
    head = {key: completion[key] for key in ("id", "created", "model")} | {"object": "chat.completion.chunk"}
    return [
        {**head, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
        {**head, "choices": [{"index": 0, "delta": {}, "finish_reason": choice["finish_reason"]}]},
        {**head, "choices": [], "usage": completion["usage"]},
    ]


def _completion(body: dict, reply: Text | ToolCall) -> dict:
    if isinstance(reply, Text):
        message, finish = {"role": "assistant", "content": reply.text}, "stop"
    else:
        message = {"role": "assistant", "content": None, "tool_calls": [{
            "id": f"call_{uuid.uuid4().hex[:12]}", "type": "function",
            "function": {"name": reply.name, "arguments": json.dumps(reply.arguments)}}]}
        finish = "tool_calls"
    return {"id": f"chatcmpl-{uuid.uuid4().hex[:12]}", "object": "chat.completion", "created": int(time.time()),
            "model": body.get("model", ""), "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": USAGE}
