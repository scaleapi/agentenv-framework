"""How upstream OSWorld's code, run unmodified in the worker, reaches a computer's guest server: upstream builds every URL
as ``http://{vm_ip}:{port}``, so a step runs it against loopback ports of its own (``Guest``), each served for the
length of one setup or check by a forwarder that sends every call on to the guest server's base URL, path prefix and
all, with the env's control headers.

The computer's Chrome DevTools port (9222) and VLC's HTTP port (8080), which upstream reaches the same way, aren't served
by the ``osworld-server`` app, so their ports get refusers: each answers 501 and records that it was called, for the
step to say what a failed setup or check needed (``Guest.needed_since``). Once its ``stop`` is set, as when the step is
cancelled, the forwarder refuses every new call, so upstream's code winds down instead of acting on the computer."""
from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import httpx

LOOPBACK = "127.0.0.1"
#: The guest's other ports upstream calls, which the osworld-server app doesn't serve, by what they are.
CHROMIUM, VLC = "chromium", "vlc"
REFUSED = {CHROMIUM: "the computer's Chrome DevTools port 9222", VLC: "VLC's HTTP port 8080"}
#: Setup downloads a task's files and a check reads large ones.
TIMEOUT_S = 1800.0
#: Not forwarded: the hop's own headers. A body is sent whole, with its length set again.
_DROPPED_REQUEST = frozenset({"host", "content-length", "connection", "keep-alive", "transfer-encoding", "te",
                              "upgrade", "proxy-authorization", "proxy-connection", "trailer"})
_DROPPED_RESPONSE = frozenset({"content-length", "connection", "keep-alive", "transfer-encoding", "content-encoding",
                               "upgrade", "trailer"})


def guest_target(base_url: str, path: str) -> str:
    """``path``, with its query, under the guest server's ``base_url``, its prefix kept."""
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def needed(names: list[str]) -> str:
    """What a setup or check that called the refused ports ``names`` needed, for its error."""
    return (f"it needs {' and '.join(REFUSED[name] for name in names)}, which the osworld-server app doesn't expose; "
            f"not supported by the osworld_* steps")


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: _Server

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def _answer(self, status: int, body: bytes, headers: dict[str, str] | None = None) -> None:
        self.send_response(status)
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _handle(self) -> None:
        length = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(length) if length else b""
        guest = self.server.guest
        if guest.stop.is_set():
            self._answer(503, b"the osworld_* step was cancelled")
            return
        if self.server.refused is not None:
            guest.mark(self.server.refused)
            self._answer(501, needed([self.server.refused]).encode())
            return
        sent = {k: v for k, v in self.headers.items() if k.lower() not in _DROPPED_REQUEST}
        sent.update(guest.headers)
        try:
            answer = guest.client.request(self.command, guest_target(guest.base_url, self.path), headers=sent,
                                          content=body, timeout=TIMEOUT_S)
        except httpx.HTTPError as e:
            self._answer(502, f"the guest server is unreachable: {type(e).__name__}".encode())
            return
        self._answer(answer.status_code, answer.content,
                     {k: v for k, v in answer.headers.items() if k.lower() not in _DROPPED_RESPONSE})

    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = do_OPTIONS = _handle


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, guest: Guest, refused: str | None):
        super().__init__((LOOPBACK, 0), _Handler)
        self.guest = guest
        self.refused = refused


class Guest:
    """The guest server at ``base_url``, reached with ``headers``, on loopback ports while it's entered: ``port``
    forwards there, until ``stop`` is set, and ``ports[CHROMIUM]`` and ``ports[VLC]`` refuse."""

    def __init__(self, base_url: str, headers: dict[str, str], stop: threading.Event | None = None):
        self.base_url = base_url
        self.headers = dict(headers)
        self.stop = stop or threading.Event()
        self.client = httpx.Client()
        self._lock = threading.Lock()
        self._refusals: list[str] = []
        self._servers = [_Server(self, None), *(_Server(self, name) for name in REFUSED)]
        self.port = self._servers[0].server_address[1]
        self.ports = {server.refused: server.server_address[1] for server in self._servers[1:]}

    def mark(self, name: str) -> None:
        with self._lock:
            self._refusals.append(name)

    def mark_count(self) -> int:
        with self._lock:
            return len(self._refusals)

    def needed_since(self, count: int) -> str | None:
        """What the refused ports called after ``mark_count`` was ``count`` needed, or None when none was."""
        with self._lock:
            names = list(dict.fromkeys(self._refusals[count:]))
        return needed(names) if names else None

    def __enter__(self) -> Guest:
        for server in self._servers:
            threading.Thread(target=server.serve_forever, name="osworld-forwarder", daemon=True).start()
        return self

    def __exit__(self, *exc: Any) -> None:
        for server in self._servers:
            server.shutdown()
            server.server_close()
        self.client.close()
