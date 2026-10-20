"""Driving the agent over HTTP the way agent-env does, grants against a fake object store, and a scripted CLI."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone

import httpx
from starlette.testclient import TestClient

ORIGIN = "https://bucket.example.com"


def card(app) -> dict:
    return app.state.agentenv_a2a.card.model_dump(mode="json", exclude_none=True, by_alias=True)


def extension_methods(app, uri: str) -> dict:
    ext = next(e for e in card(app)["capabilities"]["extensions"] if e["uri"] == uri)
    return ext["params"]["methods"]


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


class FakeStore:
    """An object store behind grants: PUT stores bytes, GET serves them."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        body = request.read()
        key = request.url.path.lstrip("/")
        if request.method == "PUT":
            self.objects[key] = body
            return httpx.Response(200)
        if key not in self.objects:
            return httpx.Response(404, stream=httpx.ByteStream(b"<Error><Code>NoSuchKey</Code></Error>"))
        return httpx.Response(200, stream=httpx.ByteStream(self.objects[key]))


class FakeStdin:
    def __init__(self) -> None:
        self.data = b""
        self.closed = False

    def write(self, data: bytes) -> None:
        self.data += data

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


def _reader(data: bytes, *, eof: bool = True) -> asyncio.StreamReader:
    reader = asyncio.StreamReader()
    reader.feed_data(data)
    if eof:
        reader.feed_eof()
    return reader


class FakeCLI:
    """A scripted `claude -p` process: fixed stdout, stderr and exit code, and a record of its stdin. ``hang``
    keeps its output open, as a CLI that never finishes."""

    def __init__(self, stdout: bytes = b"", returncode: int = 0, stderr: bytes = b"", *, hang: bool = False):
        self.stdin = FakeStdin()
        self.stdout = _reader(stdout, eof=not hang)
        self.stderr = _reader(stderr, eof=not hang)
        self.returncode = returncode
        self.killed = False

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    async def wait(self) -> int:
        return self.returncode
