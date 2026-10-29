"""Driving the agent over HTTP the way agent-env does, its gemini home, and grants against a fake object store."""

from __future__ import annotations

import email.parser
import email.policy
import json
import os
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
from starlette.testclient import TestClient

ORIGIN = "https://bucket.s3.amazonaws.com"
SESSION_ID = "12345678-aaaa-bbbb-cccc-1234567890ab"


def card(app) -> dict:
    return app.state.agentenv_a2a.card.model_dump(mode="json", exclude_none=True, by_alias=True)


def extension_methods(app, uri: str) -> dict:
    ext = next(e for e in card(app)["capabilities"]["extensions"] if e["uri"] == uri)
    return ext["params"]["methods"]


def settings(home: Path) -> dict:
    return json.loads((home / ".gemini" / "settings.json").read_text())


def chats(home: Path) -> Path:
    return home / ".gemini" / "tmp" / "workspace" / "chats"


def transcript(fmt: str, session_id: str = SESSION_ID) -> bytes:
    """A session transcript as gemini writes it: `.json` on a first turn, `.jsonl` once resumed."""
    if fmt == "json":
        return json.dumps({"sessionId": session_id, "messages": [{"type": "user", "content": "hi"}]}, indent=2).encode()
    return (
        json.dumps({"sessionId": session_id, "projectHash": "p"}) + "\n" + json.dumps({"type": "user", "content": "hi"}) + "\n"
    ).encode()


def send(client: TestClient, parts: list[dict] | str, context_id: str = "ctx-1") -> dict:
    """message/send, which blocks until the task is terminal; the task."""
    if isinstance(parts, str):
        parts = [{"kind": "text", "text": parts}]
    message = {"kind": "message", "role": "user", "messageId": uuid.uuid4().hex, "contextId": context_id, "parts": parts}
    response = client.post("/a2a", json={"jsonrpc": "2.0", "id": 1, "method": "message/send", "params": {"message": message}})
    response.raise_for_status()
    return response.json()["result"]


def terminal(task: dict) -> tuple[str, str, list[dict]]:
    """A terminal task's state, its text, and its data parts in order."""
    parts = task["status"]["message"]["parts"]
    text = next(part["text"] for part in parts if part["kind"] == "text")
    return task["status"]["state"], text, [part["data"] for part in parts if part["kind"] == "data"]


def expiry() -> str:
    return (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat().replace("+00:00", "Z")


def write_grant(key: str, media_type: str = "application/octet-stream", max_bytes: int = 1 << 20) -> dict:
    return {
        "media_type": media_type,
        "max_bytes": max_bytes,
        "write": {"kind": "http-put", "url": f"{ORIGIN}/{key}?sig=w", "expires_at": expiry(), "headers": {"Content-Type": media_type}},
    }


def read_grant(key: str, size: int) -> dict:
    return {
        "media_type": "application/octet-stream",
        "max_bytes": max(size, 1),
        "size_bytes": size,
        "read": {"kind": "http-get", "url": f"{ORIGIN}/{key}?sig=r", "expires_at": expiry()},
    }


def namespace_grant(root_path: str = "agent_changelog/run-1/solver") -> dict:
    return {
        "root_path": root_path,
        "expires_at": expiry(),
        "max_objects": 100,
        "max_object_bytes": 1 << 20,
        "max_total_bytes": 1 << 24,
        "write": {
            "kind": "http-post-policy",
            "url": f"{ORIGIN}/",
            "fields": {"key": f"{root_path}/${{filename}}", "policy": "p"},
            "path_field": "key",
            "file_field": "file",
        },
    }


class FakeStore:
    """An object store behind grants: PUT and POST store bytes, GET serves them. Keys in
    ``refuse`` answer 403 to every method."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.requests: list[httpx.Request] = []
        self.refuse: set[str] = set()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        body = request.read()
        key = request.url.path.lstrip("/")
        if key in self.refuse:
            return httpx.Response(403, stream=httpx.ByteStream(b"<Error><Code>AccessDenied</Code></Error>"))
        if request.method == "PUT":
            self.objects[key] = body
            return httpx.Response(200)
        if request.method == "POST":
            fields = _multipart(request, body)
            self.objects[fields["key"].decode()] = fields["file"]
            return httpx.Response(204)
        if key not in self.objects:
            return httpx.Response(404, stream=httpx.ByteStream(b"<Error><Code>NoSuchKey</Code></Error>"))
        return httpx.Response(200, stream=httpx.ByteStream(self.objects[key]))

    def gets(self) -> list[str]:
        return [r.url.path.lstrip("/") for r in self.requests if r.method == "GET"]


def _multipart(request: httpx.Request, body: bytes) -> dict[str, bytes]:
    boundary = request.headers["content-type"].split("boundary=")[1].encode()
    fields = {}
    for raw in body.split(b"--" + boundary)[1:-1]:
        head, _, value = raw[2:-2].partition(b"\r\n\r\n")
        fields[head.split(b'name="')[1].split(b'"')[0].decode()] = value
    return fields


class Staging:
    """A local server taking a namespace grant's multipart POSTs, as the agent's own staging does over loopback: each
    object's bytes, by the path it was posted to."""

    ROOT_PATH = "agent_changelog/run-1/solver"
    PATH = "/staging/changelog"

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        staging = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers["content-length"]))
                message = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(
                    f"content-type: {self.headers['content-type']}\r\n\r\n".encode() + body)
                fields = {part.get_param("name", header="content-disposition"): part.get_content()
                          for part in message.iter_parts()}
                staging.objects[fields["key"]] = fields["file"]
                self.send_response(204)
                self.send_header("content-length", "0")
                self.end_headers()

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def grant(self) -> dict:
        """A namespace grant on the agent's own staging, so the hook's uploads reach this server over loopback."""
        return {
            "root_path": self.ROOT_PATH,
            "expires_at": expiry(),
            "max_objects": 10,
            "max_object_bytes": 1 << 20,
            "max_total_bytes": 1 << 22,
            "write": {
                "kind": "http-post-policy",
                "url": f"https://bucket.example.com{self.PATH}",
                "fields": {"policy": "p", "signature": "s"},
                "path_field": "key",
                "file_field": "file",
                "headers": {"AgentEnv-Staging-Path": self.PATH},
            },
        }

    def capture(self, changelog_dir: Path, tool: str, session: str) -> subprocess.CompletedProcess:
        """One AfterTool firing, as the CLI starts the hook: the agent's environment, the hook's JSON on stdin."""
        env = {**os.environ, "A2A_PORT": str(self.server.server_port)}
        return subprocess.run([sys.executable, str(changelog_dir / "capture.py")],
                              input=json.dumps({"tool_name": tool, "session_id": session}),
                              capture_output=True, text=True, env=env, timeout=120, check=True)

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
