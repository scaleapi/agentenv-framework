"""A fake Gemini API, as LiteLLM's /gemini passthrough serves it, so the real Gemini CLI runs with no model key: each
``:streamGenerateContent`` (one server-sent event) and ``:generateContent`` call is answered by a responder, and every
request is recorded. The CLI's own calls that offer no tools, such as its next-speaker check, are answered with text, or
with a JSON object when they ask for one."""

from __future__ import annotations

import json
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


@dataclass(frozen=True)
class Text:
    text: str


@dataclass(frozen=True)
class Call:
    name: str
    args: dict


@dataclass
class Seen:
    path: str
    body: dict


Responder = Callable[[dict], Text | Call]
GENERATE = (":generateContent", ":streamGenerateContent")


def answer(text: str) -> Responder:
    """Every request answered with ``text``."""
    return lambda body: Text(text)


def shell_then_answer(command: str, *, prefix: str = "") -> Responder:
    """Run ``command`` with the CLI's shell tool, then answer with ``prefix`` and the tool's output."""

    def respond(body: dict) -> Text | Call:
        result = _tool_result_text(body)
        return Text(f"{prefix}{result}") if result is not None else Call("run_shell_command", {"command": command})

    return respond


def _tool_result_text(body: dict) -> str | None:
    """The output of the tool call the latest content answers, or None."""
    contents = body.get("contents") or []
    for part in contents[-1].get("parts", []) if contents else ():
        if "functionResponse" in part:
            return json.dumps(part["functionResponse"].get("response"))
    return None


@dataclass
class FakeGemini:
    host: str
    respond: Responder = field(default_factory=lambda: answer("pong"))
    # The latest requests only: each call resends the whole history.
    seen: deque[Seen] = field(default_factory=lambda: deque(maxlen=200))
    _server: ThreadingHTTPServer | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __enter__(self) -> FakeGemini:
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

    def generate_calls(self) -> list[Seen]:
        """The model calls that offer the model tools: the turn's own, not the CLI's utility calls or token counts."""
        return [seen for seen in self.seen if seen.path.endswith(GENERATE) and seen.body.get("tools")]

    def tool_results(self) -> list[str]:
        """The output of each tool call the CLI sent back."""
        return [text for seen in self.generate_calls() if (text := _tool_result_text(seen.body)) is not None]


_SCHEMA_VALUES = {"STRING": "user", "INTEGER": 1, "NUMBER": 1, "BOOLEAN": False, "ARRAY": [], "OBJECT": {}}


def _schema_answer(schema: dict) -> dict:
    """An object with each of ``schema``'s properties, of its type: the CLI's router and next-speaker check ask for one."""
    return {name: _SCHEMA_VALUES.get(str(prop.get("type")).upper(), "user")
            for name, prop in (schema.get("properties") or {}).items()}


def _reply(fake: FakeGemini, body: dict) -> dict:
    if body.get("tools"):
        reply = fake.respond(body)
    else:
        schema = (body.get("generationConfig") or {}).get("responseJsonSchema")
        reply = Text(json.dumps(_schema_answer(schema)) if isinstance(schema, dict) else "fake")
    part = {"text": reply.text} if isinstance(reply, Text) else {"functionCall": {"name": reply.name, "args": reply.args}}
    return {"candidates": [{"content": {"role": "model", "parts": [part]}, "finishReason": "STOP", "index": 0}],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15}}


def _handler(fake: FakeGemini) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args) -> None:
            pass

        def do_POST(self) -> None:
            path = self.path.split("?", 1)[0]
            body = json.loads(self.rfile.read(int(self.headers.get("content-length") or 0)) or b"{}")
            with fake._lock:
                fake.seen.append(Seen(path, body))
            if path.endswith(":countTokens"):
                return self._send(200, json.dumps({"totalTokens": 10}).encode(), "application/json")
            if not path.endswith(GENERATE):
                return self._send(404, json.dumps({"error": {"message": path}}).encode(), "application/json")
            reply = json.dumps(_reply(fake, body))
            if path.endswith(":streamGenerateContent"):
                return self._send(200, f"data: {reply}\r\n\r\n".encode(), "text/event-stream")
            self._send(200, reply.encode(), "application/json")

        def _send(self, status: int, data: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler
