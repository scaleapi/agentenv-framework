"""A stand-in for the shared site proxy in this base, on 127.0.0.1:3128: it forwards every request the browser sends
it, absolute-form HTTP and CONNECT tunnels alike, to the host it names, refusing nothing and never decrypting a tunnel.
The sandbox's own network policy still decides where a connection can go. Every line it logs is JSON, in the same
shapes as the shared site proxy's: ``connection`` (host, port, address, bytes each way, start, duration) when one
ends, and ``failed`` (method, host, port, ``reason`` "doesn't resolve" or "no answer", the error, and when it was
asked for) for one that couldn't open."""
import http.server
import json
import selectors
import socket
import socketserver
import time
import urllib.parse

PORT = 3128
CONNECT_TIMEOUT_S = 30
IDLE_S = 300
HOP = {"connection", "keep-alive", "proxy-connection", "proxy-authorization"}


def log(**fields) -> None:
    print(json.dumps(fields), flush=True)


def _when(at: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(at))


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
        if (opened := self._open(host, int(port), "CONNECT")) is None:
            return
        self.send_response(200, "Connection established")
        self.end_headers()
        self.close_connection = True
        self._relay(*opened, host, int(port), b"")

    def forward(self):
        url = urllib.parse.urlsplit(self.path)
        if not url.hostname:
            self.send_error(400)
            return
        port = url.port or 80
        if (opened := self._open(url.hostname, port, self.command)) is None:
            return
        target = (url.path or "/") + (f"?{url.query}" if url.query else "")
        headers = "".join(f"{k}: {v}\r\n" for k, v in self.headers.items() if k.lower() not in HOP)
        head = f"{self.command} {target} HTTP/1.1\r\n{headers}Connection: close\r\n\r\n".encode("latin-1")
        body = self.rfile.read(length) if (length := int(self.headers.get("Content-Length") or 0)) else b""
        self.close_connection = True
        self._relay(*opened, url.hostname, port, head + body)

    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = do_OPTIONS = forward

    def _open(self, host: str, port: int, method: str) -> tuple[socket.socket, str] | None:
        """A connection to an address ``host`` resolves to, tried in turn within one ``CONNECT_TIMEOUT_S`` for them
        all, and that address; or None once the browser has its answer, a 502 for a host that doesn't resolve or
        answer. The connection keeps an ``IDLE_S`` timeout, so a write to a peer that stopped reading ends too."""
        asked, deadline = time.time(), time.monotonic() + CONNECT_TIMEOUT_S
        try:
            found = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except OSError as error:
            log(failed={"method": method, "host": host, "port": port, "reason": "doesn't resolve",
                        "error": _error(error), "started": _when(asked)})
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
        log(failed={"method": method, "host": host, "port": port, "reason": "no answer",
                    "error": _error(last) if last else f"none within {CONNECT_TIMEOUT_S} s", "started": _when(asked)})
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

    def _relay(self, upstream: socket.socket, address: str, host: str, port: int, first: bytes) -> None:
        started, began = time.time(), time.monotonic()
        sent, received = len(first), 0
        self.connection.settimeout(IDLE_S)
        try:
            if first:
                upstream.sendall(first)
            more, received = _pipe(self.connection, upstream)
            sent += more
        finally:
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
    log(listening={"address": "127.0.0.1", "port": PORT})
    Server(("127.0.0.1", PORT), Proxy).serve_forever()
