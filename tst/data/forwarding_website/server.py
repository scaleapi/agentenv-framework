"""A third-party app wrapped as a website env on ForwardingSite, named shop.localhost, for integration-testing a site
whose pages an app beside it serves: the app, on loopback, signs in only a request carrying X-Auto-Login, which the
site's own session request sends and the forwarder drops from every other. Like an app with a fixed base URL, it ties
its session cookie to its name (Domain), giving an anonymous visit one too, and sets it again on every answer. Each of
its /admin pages has as its title, as JSON, who the request's cookie signs in, the Host and path the app was asked and
whether it saw X-Auto-Login."""
import json
import secrets
import threading
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from agentenv_protocol.website import ForwardingSite, session_state, website

APP = ("127.0.0.1", 8080)
NAME = "shop.localhost"
AUTO_LOGIN = "X-Auto-Login"
SESSIONS: dict[str, str | None] = {}


def session_cookie(sid: str) -> str:
    return f"sid={sid}; Domain={NAME}; Path=/admin; Max-Age=3600; HttpOnly"


def first_sid(header: str) -> str | None:
    """The first sid a Cookie header sends, as PHP reads it: a browser holding two sends the longer path's first."""
    return next((value for name, _, value in (pair.strip().partition("=") for pair in header.split(";"))
                 if name == "sid"), None)


class App(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        if self.headers.get(AUTO_LOGIN):
            sid = secrets.token_hex(8)
            SESSIONS[sid] = self.headers[AUTO_LOGIN]
            self._send(302, b"", [("Location", "/admin/home"), ("Set-Cookie", session_cookie(sid))])
            return
        sent = first_sid(self.headers.get("Cookie", ""))
        sid = sent if sent in SESSIONS else secrets.token_hex(8)
        SESSIONS.setdefault(sid, None)
        seen = {"user": SESSIONS[sid], "host": self.headers.get("Host"), "path": self.path,
                "auto_login": AUTO_LOGIN.lower() in (name.lower() for name in self.headers)}
        page = f"<title>{json.dumps(seen)}</title><h1>admin</h1>".encode()
        self._send(200, page, [("Content-Type", "text/html; charset=utf-8"), ("Cache-Control", "no-store"),
                               ("Set-Cookie", session_cookie(sid))])

    def _send(self, status, body, headers):
        self.send_response(status)
        for name, value in headers:
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@website(domains=[NAME], builtin_routes=False, clock="real")
class Shop(ForwardingSite):
    upstream = f"http://{APP[0]}:{APP[1]}"
    ready_path = "/admin/home"
    sign_in_headers = (AUTO_LOGIN,)

    @session_state
    async def signed_in(self, user):
        answer = await self.ask_app("/admin/home", headers={AUTO_LOGIN: user or "admin"})
        sid = SimpleCookie(answer.headers["set-cookie"])["sid"]
        return {"cookies": [{"name": "sid", "value": sid.value, "path": "/admin", "max_age": 3600, "httpOnly": True}],
                "local_storage": []}


if __name__ == "__main__":
    server = ThreadingHTTPServer(APP, App)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    Shop().serve()
