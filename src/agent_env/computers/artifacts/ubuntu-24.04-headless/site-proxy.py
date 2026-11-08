"""A stand-in for the shared site proxy in this base: it forwards every request the browser sends it, absolute-form
HTTP and CONNECT tunnels alike, to wherever it names, refusing nothing, on 127.0.0.1:3128."""
import http.server
import select
import socket
import socketserver
import urllib.parse

PORT = 3128
IDLE_S = 60
HOP = {"connection", "keep-alive", "proxy-connection", "proxy-authorization"}


class Proxy(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_CONNECT(self):
        host, _, port = self.path.rpartition(":")
        host = host.removeprefix("[").removesuffix("]")
        try:
            upstream = socket.create_connection((host, int(port)), timeout=10)
        except (OSError, ValueError):
            self.send_error(502)
            return
        self.send_response(200, "Connection established")
        self.end_headers()
        self.close_connection = True
        _pipe(self.connection, upstream)

    def forward(self):
        url = urllib.parse.urlsplit(self.path)
        try:
            upstream = socket.create_connection((url.hostname, url.port or 80), timeout=10)
        except (OSError, TypeError):
            self.send_error(502)
            return
        target = (url.path or "/") + (f"?{url.query}" if url.query else "")
        headers = "".join(f"{k}: {v}\r\n" for k, v in self.headers.items() if k.lower() not in HOP)
        upstream.sendall(f"{self.command} {target} HTTP/1.1\r\n{headers}Connection: close\r\n\r\n".encode("latin-1"))
        if length := int(self.headers.get("Content-Length") or 0):
            upstream.sendall(self.rfile.read(length))
        self.close_connection = True
        _pipe(self.connection, upstream)

    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = do_OPTIONS = forward

    def log_message(self, *args):
        pass


def _pipe(a: socket.socket, b: socket.socket) -> None:
    try:
        while True:
            ready, _, _ = select.select([a, b], [], [], IDLE_S)
            if not ready:
                return
            for s in ready:
                data = s.recv(65536)
                if not data:
                    return
                (b if s is a else a).sendall(data)
    except OSError:
        return
    finally:
        b.close()


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


if __name__ == "__main__":
    Server(("127.0.0.1", PORT), Proxy).serve_forever()
