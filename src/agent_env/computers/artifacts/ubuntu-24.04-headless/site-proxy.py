"""A stand-in for the shared site proxy in this base, on 127.0.0.1:3128: it forwards every request the browser sends
it, absolute-form HTTP and CONNECT tunnels alike, to the host it names, never decrypting a tunnel. It resolves each
name itself and connects only to an address it checked: one that isn't globally routable (loopback, private,
link-local, the metadata service, their IPv6 equivalents) is refused at once, so neither a page nor an agent driving
the browser reaches the computer's own network. The computer's own loopback is reached only when its deploy sets the
base's `own_loopback` setting, as a test serving its pages inside the computer does. Each connection is logged as a
JSON line when it ends, and each refusal with its reason."""
import http.server
import ipaddress
import json
import os
import selectors
import socket
import socketserver
import sys
import time
import urllib.parse

PORT = 3128
CONNECT_TIMEOUT_S = 30
IDLE_S = 300
HOP = {"connection", "keep-alive", "proxy-connection", "proxy-authorization"}
NAT64 = ipaddress.ip_network("64:ff9b::/96")
#: Whether the deploy lets the browser reach this computer's own loopback (the base's `own_loopback` setting).
OWN_LOOPBACK = os.environ.get("AGENTENV_SETTING_OWN_LOOPBACK", "").lower() in ("true", "1")


def allowed(address: str) -> bool:
    """Whether the proxy may connect to ``address``: a globally routable one, not multicast, an IPv6 address that
    carries an IPv4 one (mapped, 6to4 or NAT64) judged as that IPv4 address."""
    ip = ipaddress.ip_address(address.split("%", 1)[0])
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            ip = ip.ipv4_mapped
        elif ip.sixtofour is not None:
            ip = ip.sixtofour
        elif ip in NAT64:
            ip = ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
    return ip.is_global and not ip.is_multicast


def _loopback(address: str) -> bool:
    ip = ipaddress.ip_address(address.split("%", 1)[0])
    return ip.is_loopback or isinstance(ip, ipaddress.IPv6Address) and bool(ip.ipv4_mapped and ip.ipv4_mapped.is_loopback)


def resolve(host: str, port: int) -> tuple[list[str], list[str]]:
    """The addresses ``host`` resolves to that the proxy may connect to, in the resolver's order, and those it may
    not; raises OSError when it doesn't resolve."""
    found = list(dict.fromkeys(info[4][0] for info in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)))
    usable = [address for address in found if allowed(address) or OWN_LOOPBACK and _loopback(address)]
    return usable, [address for address in found if address not in usable]


def log(**fields) -> None:
    print(json.dumps(fields), flush=True)


class Proxy(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_CONNECT(self):
        host, _, port = self.path.rpartition(":")
        host = host.removeprefix("[").removesuffix("]")
        if not port.isdigit():
            self.send_error(400)
            return
        if (upstream := self._open(host, int(port), "CONNECT")) is None:
            return
        self.send_response(200, "Connection established")
        self.end_headers()
        self.close_connection = True
        self._relay(upstream, host, int(port), b"")

    def forward(self):
        url = urllib.parse.urlsplit(self.path)
        if not url.hostname:
            self.send_error(400)
            return
        port = url.port or 80
        if (upstream := self._open(url.hostname, port, self.command)) is None:
            return
        target = (url.path or "/") + (f"?{url.query}" if url.query else "")
        headers = "".join(f"{k}: {v}\r\n" for k, v in self.headers.items() if k.lower() not in HOP)
        head = f"{self.command} {target} HTTP/1.1\r\n{headers}Connection: close\r\n\r\n".encode("latin-1")
        body = self.rfile.read(length) if (length := int(self.headers.get("Content-Length") or 0)) else b""
        self.close_connection = True
        self._relay(upstream, url.hostname, port, head + body)

    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = do_OPTIONS = forward

    def _open(self, host: str, port: int, method: str):
        """A connection to an address ``host`` resolves to that the proxy may reach, or None once the browser has
        its answer: a 403 for a host whose every address is refused, a 502 for one that doesn't resolve or answer."""
        try:
            usable, refused = resolve(host, port)
        except OSError as error:
            log(refused={"method": method, "host": host, "port": port, "reason": f"doesn't resolve: {error}"})
            self._answer(502, f"{host} doesn't resolve\n")
            return None
        if not usable:
            log(refused={"method": method, "host": host, "port": port, "reason": "internal address",
                         "addresses": refused})
            self._answer(403, f"{host} is an internal address, which the browser never reaches\n")
            return None
        for address in usable:
            try:
                upstream = socket.create_connection((address, port), timeout=CONNECT_TIMEOUT_S)
                upstream.settimeout(None)
                self.address = address
                return upstream
            except OSError:
                continue
        log(refused={"method": method, "host": host, "port": port, "reason": "no answer", "addresses": usable})
        self._answer(502, f"{host} didn't answer\n")
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

    def _relay(self, upstream: socket.socket, host: str, port: int, first: bytes) -> None:
        started, began = time.time(), time.monotonic()
        sent, received = len(first), 0
        try:
            if first:
                upstream.sendall(first)
            more, received = _pipe(self.connection, upstream)
            sent += more
        finally:
            log(connection={"host": host, "port": port, "address": self.address, "bytes_sent": sent,
                            "bytes_received": received,
                            "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
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
    print(f"site proxy stand-in on 127.0.0.1:{PORT}", file=sys.stderr, flush=True)
    Server(("127.0.0.1", PORT), Proxy).serve_forever()
