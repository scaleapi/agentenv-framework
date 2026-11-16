"""A stand-in for the shared site proxy in this base, on 127.0.0.1:3128: it forwards every request the browser sends
it, absolute-form HTTP and CONNECT tunnels alike, to the host it names, refusing nothing and never decrypting a tunnel.
The sandbox's own network policy still decides where a connection can go. Every line it logs is JSON, in the same
shapes as the shared site proxy's: ``connection`` (host, port, address, bytes each way, start, duration) when one
ends, and ``failed`` (method, host, port, ``reason`` "doesn't resolve" or "no answer", the error, and when it was
asked for) for one that couldn't open.

It also keeps the browser's network log, in the file ``AGENTENV_NETWORK_LOG`` names, in the shared site proxy's lines:
an ``open`` line when a connection opens and an ``end`` line, naming the same ``id``, when it ends or fails to open,
each numbered after the last the file holds. Seeing no request inside a tunnel, it keeps each at connection level."""
import http.server
import itertools
import json
import os
import re
import selectors
import socket
import socketserver
import threading
import time
import urllib.parse
from datetime import datetime, timezone

PORT = 3128
CONNECT_TIMEOUT_S = 30
IDLE_S = 300
HOP = {"connection", "keep-alive", "proxy-connection", "proxy-authorization"}
NETWORK_LOG_ENV, NETWORK_LOG = "AGENTENV_NETWORK_LOG", "/var/log/agentenv/network.jsonl"
NUMBERED = re.compile(rb'^\{"seq":(\d+),"at":"[^"]*","(open|end)":\{"id":"([^"]*)"')
RESTARTED = {"reason": "the proxy restarted", "error": "it was in flight when the proxy that carried it stopped"}


def log(**fields) -> None:
    print(json.dumps(fields), flush=True)


def _when(at: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(at))


def _stamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class NetworkLog:
    """The network log in ``path``, each line numbered after the last the file holds, so a proxy started again
    carries on, ending each entry the one before left open as ``RESTARTED``, and written whole by one thread at a
    time."""

    def __init__(self, path: str):
        seq, partial, left = 0, False, {}
        try:
            with open(path, "rb") as existing:
                for raw in existing:
                    partial = not raw.endswith(b"\n")
                    if not partial and (numbered := NUMBERED.match(raw)):
                        seq = max(seq, int(numbered.group(1)))
                        if numbered.group(2) == b"open":
                            left[numbered.group(3).decode()] = True
                        else:
                            left.pop(numbered.group(3).decode(), None)
        except FileNotFoundError:
            pass
        self.file = open(path, "ab")
        if partial:
            self.file.write(b"\n")
        self.seq, self.lock, self.ids = seq, threading.Lock(), itertools.count(1)
        self.run = f"{os.getpid()}-{seq}"
        for key in left:
            self.write("end", {"id": key, "ended": _stamp(), "failed": RESTARTED})

    def key(self) -> str:
        return f"{self.run}-{next(self.ids)}"

    def write(self, kind: str, record: dict) -> None:
        with self.lock:
            self.seq += 1
            line = json.dumps({"seq": self.seq, "at": _stamp(), kind: {"id": record["id"], **record}},
                              separators=(",", ":")) + "\n"
            try:
                self.file.write(line.encode())
                self.file.flush()
            except (OSError, ValueError) as error:
                log(error={"connection": "network log", "error": _error(error)})


NETWORK: NetworkLog | None = None


def _note(kind: str, record: dict) -> None:
    if NETWORK is not None:
        NETWORK.write(kind, record)


def _key() -> str:
    return NETWORK.key() if NETWORK is not None else ""


def _error(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}".removesuffix(": ")


class Proxy(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_CONNECT(self):
        host, _, port = self.path.rpartition(":")
        host = host.removeprefix("[").removesuffix("]")
        if not port.isdigit():
            self.send_error(400)
            return
        key, started, record = _key(), _stamp(), {"kind": "tunnel", "host": host, "port": int(port)}
        if (opened := self._open(host, int(port), "CONNECT", key, started, record)) is None:
            return
        self.close_connection = True
        self._relay(*opened, host, int(port), b"", key, {**record, "id": key, "started": started}, established=True)

    def forward(self):
        url = urllib.parse.urlsplit(self.path)
        if not url.hostname:
            self.send_error(400)
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self.send_error(400)
            return
        port = url.port or 80
        key, started = _key(), _stamp()
        record = {"kind": "tunnel", "host": url.hostname, "port": port, "method": self.command, "url": self.path}
        if (opened := self._open(url.hostname, port, self.command, key, started, record)) is None:
            return
        target = (url.path or "/") + (f"?{url.query}" if url.query else "")
        headers = "".join(f"{k}: {v}\r\n" for k, v in self.headers.items() if k.lower() not in HOP)
        head = f"{self.command} {target} HTTP/1.1\r\n{headers}Connection: close\r\n\r\n".encode("latin-1")
        self.close_connection = True
        self._relay(*opened, url.hostname, port, head, key, {**record, "id": key, "started": started}, length=length)

    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = do_OPTIONS = forward

    def _open(self, host: str, port: int, method: str, key: str, started: str,
              record: dict) -> tuple[socket.socket, str] | None:
        """A connection to an address ``host`` resolves to, tried in turn within one ``CONNECT_TIMEOUT_S`` for them
        all, and that address; or None once the browser has its answer, a 502 for a host that doesn't resolve or
        answer, which ends ``record`` in the network log. The connection keeps an ``IDLE_S`` timeout, so a write to a
        peer that stopped reading ends too."""
        asked, deadline = time.time(), time.monotonic() + CONNECT_TIMEOUT_S
        try:
            found = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except OSError as error:
            log(failed={"method": method, "host": host, "port": port, "reason": "doesn't resolve",
                        "error": _error(error), "started": _when(asked)})
            _note("end", {**record, "id": key, "started": started, "ended": _stamp(),
                          "failed": {"reason": "doesn't resolve", "error": _error(error)}})
            self._answer(502, f"{host} doesn't resolve\n")
            return None
        last = None
        for address in dict.fromkeys(info[4][0] for info in found):
            if (left := deadline - time.monotonic()) <= 0:
                break
            try:
                upstream = socket.create_connection((address, port), timeout=left)
            except OSError as error:
                last = error
                continue
            upstream.settimeout(IDLE_S)
            return upstream, address
        why = _error(last) if last else f"none within {CONNECT_TIMEOUT_S} s"
        log(failed={"method": method, "host": host, "port": port, "reason": "no answer", "error": why,
                    "started": _when(asked)})
        _note("end", {**record, "id": key, "started": started, "ended": _stamp(),
                      "failed": {"reason": "no answer", "error": why}})
        self._answer(502, f"{host}:{port} didn't answer\n")
        return None

    def _answer(self, status: int, body: str) -> None:
        data = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Agentenv-Site-Proxy", "failed")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)
        self.close_connection = True

    def _relay(self, upstream: socket.socket, address: str, host: str, port: int, first: bytes, key: str,
               opening: dict, *, established: bool = False, length: int = 0) -> None:
        """The connection to ``upstream`` relayed both ways, in the network log from its ``open`` line to its ``end``
        line, which is written however it ends: a tunnel's ``established`` answer, or a request's ``length`` bytes of
        body after its head ``first``, are sent within it."""
        started, began = time.time(), time.monotonic()
        sent = received = 0
        try:
            _note("open", {**opening, "address": address})
            if established:
                self.send_response(200, "Connection established")
                self.end_headers()
            if length:
                first += self.rfile.read(length)
            self.connection.settimeout(IDLE_S)
            if first:
                upstream.sendall(first)
                sent = len(first)
            more, received = _pipe(self.connection, upstream)
            sent += more
        finally:
            upstream.close()
            _note("end", {"id": key, "ended": _stamp(), "bytes_sent": sent, "bytes_received": received})
            log(connection={"host": host, "port": port, "address": address, "bytes_sent": sent,
                            "bytes_received": received,
                            "started": _when(started),
                            "duration_s": round(time.monotonic() - began, 3)})

    def log_message(self, *args):
        pass


def _pipe(client: socket.socket, upstream: socket.socket) -> tuple[int, int]:
    """Relay both ways until either side closes or both are idle ``IDLE_S``; the bytes sent upstream and received.
    It waits with a selector (epoll on Linux), not ``select()``, which refuses a socket numbered past 1023."""
    sent = received = 0
    waiting = selectors.DefaultSelector()
    try:
        waiting.register(client, selectors.EVENT_READ)
        waiting.register(upstream, selectors.EVENT_READ)
        while True:
            ready = waiting.select(IDLE_S)
            if not ready:
                break
            for key, _ in ready:
                s = key.fileobj
                data = s.recv(65536)
                if not data:
                    return sent, received
                (upstream if s is client else client).sendall(data)
                if s is client:
                    sent += len(data)
                else:
                    received += len(data)
    except OSError:
        pass
    finally:
        waiting.close()
        upstream.close()
    return sent, received


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


if __name__ == "__main__":
    try:
        NETWORK = NetworkLog(os.environ.get(NETWORK_LOG_ENV) or NETWORK_LOG)
    except OSError as error:
        log(error={"connection": "network log", "error": _error(error)})
    log(listening={"address": "127.0.0.1", "port": PORT})
    Server(("127.0.0.1", PORT), Proxy).serve_forever()
