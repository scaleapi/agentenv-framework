"""A site that wraps an app beside it: ForwardingSite sends every request agentenv's own routes don't answer to the app
as it came, its path's encoding, its query, its Host and its body, without the hop-by-hop headers or the ones the site
drops, and streams the app's answer back as the app gave it, still encoded, its redirects not followed; the card, the
data plane and website/v1 stay the site's; it is ready once the app answers its ready path; and the switches a
third-party app needs: no built-in site routes, and a real clock, which serves no clock/v1, stamps no X-Universe-Now
and has the conformance check skip its clock checks."""

from __future__ import annotations

import asyncio
import gzip
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from agentenv_protocol.client import find_extension_method
from agentenv_protocol.website import (
    CLOCK_EXTENSION_URI,
    UNIVERSE_NOW_HEADER,
    WEBSITE_EXTENSION_URI,
    WEBSITE_SIGN_IN_HEADERS_PARAM,
    ForwardingSite,
    WebsiteEnvironment,
    conformance,
    session_state,
    website,
)
from agentenv_protocol.website.forwarding import WEBSOCKET_REFUSED, Forwarder

HOST = "shop.localhost"
PAGE = b"<html><head><title>Shop</title></head><body>home</body></html>"
ZIPPED = gzip.compress(b'{"items": [1, 2, 3]}')


class App:
    """The app a site wraps, on loopback: each request it got, by method, raw path, headers and body, and its answers
    by path."""

    def __init__(self) -> None:
        self.seen: list[dict] = []
        self.status = 200
        app = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _answer(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
                    body = b""
                    while (size := int(self.rfile.readline().split(b";")[0], 16)):
                        body += self.rfile.read(size)
                        self.rfile.readline()
                    self.rfile.readline()
                else:
                    body = self.rfile.read(length) if length else b""
                app.seen.append({"method": self.command, "path": self.path,
                                 "headers": [(k.lower(), v) for k, v in self.headers.items()], "body": body})
                path = self.path.partition("?")[0]
                if path == "/redirect":
                    self._send(302, b"", [("Location", f"http://{HOST}/elsewhere")])
                elif path == "/zipped":
                    self._send(200, ZIPPED, [("Content-Type", "application/json"), ("Content-Encoding", "gzip")])
                elif path == "/cookies":
                    self._send(200, b"ok", [("Content-Type", "text/plain"), ("Set-Cookie", "a=1; Path=/"),
                                           ("Set-Cookie", "b=2; Path=/admin; HttpOnly"),
                                           ("Set-Cookie", f"c=3; Domain={HOST}; Path=/"),
                                           ("Set-Cookie", f"d=4; domain=.{HOST.upper()}"),
                                           ("Set-Cookie", "e=5; Domain=other.example; Path=/")])
                elif path == "/streamed":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain")
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    for part in (b"one,", b"two,", b"three"):
                        self.wfile.write(b"%x\r\n%s\r\n" % (len(part), part))
                    self.wfile.write(b"0\r\n\r\n")
                else:
                    self._send(app.status, PAGE, [("Content-Type", "text/html; charset=utf-8"), ("X-App", "magento")])

            def _send(self, status: int, body: bytes, headers: list[tuple[str, str]]) -> None:
                self.send_response(status)
                for name, value in headers:
                    self.send_header(name, value)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(body)

            do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = do_OPTIONS = _answer

            def log_message(self, *args) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def last(self) -> dict:
        return self.seen[-1]


@pytest.fixture
def app() -> Iterator[App]:
    served = App()
    yield served
    served.close()


def _site(upstream: str, **declared) -> type[ForwardingSite]:
    @website(domains=[HOST], builtin_routes=False, clock="real", **declared)
    class Shop(ForwardingSite):
        sign_in_headers = ("X-Auto-Login",)
        ready_path = "/admin"

        @session_state
        def signed_in(self, user):
            return {"cookies": [{"name": "admin", "value": "s1", "path": "/admin", "max_age": 3600}],
                    "local_storage": []}

    Shop.upstream = upstream
    return Shop


@pytest.fixture
def client(app: App) -> Iterator[TestClient]:
    with TestClient(_site(app.url)().create_app(), headers={"X-Forwarded-Host": HOST}) as forwarded:
        yield forwarded


def test_a_request_goes_to_the_app_as_it_came_without_hop_by_hop_or_sign_in_headers(app, client):
    answer = client.get("/admin/a%2Fb/c?x=1&y=%20z", headers={
        "X-Custom": "kept", "Cookie": "admin=s1", "Connection": "keep-alive, X-Per-Hop", "X-Per-Hop": "gone",
        "Keep-Alive": "timeout=5", "x-auto-login": "admin:admin1234"})

    seen = app.last()
    headers = dict(seen["headers"])
    assert (answer.status_code, answer.content, answer.headers["x-app"]) == (200, PAGE, "magento")
    assert (seen["method"], seen["path"]) == ("GET", "/admin/a%2Fb/c?x=1&y=%20z")
    assert (headers["host"], headers["x-custom"], headers["cookie"]) == (HOST, "kept", "admin=s1")
    assert not {"x-per-hop", "keep-alive", "x-auto-login"} & set(headers), headers


async def _forwarded(forwarder: Forwarder, path: str, headers: list[tuple[bytes, bytes]]) -> list[dict]:
    """The ASGI messages ``forwarder`` sends for a GET of ``path`` with ``headers`` alone, as a browser's request."""
    sent: list[dict] = []
    requested, answered = asyncio.Event(), asyncio.Event()

    async def receive() -> dict:
        if not requested.is_set():
            requested.set()
            return {"type": "http.request", "body": b"", "more_body": False}
        await answered.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict) -> None:
        sent.append(message)
        if message["type"] == "http.response.body" and not message.get("more_body"):
            answered.set()

    await forwarder({"type": "http", "method": "GET", "path": path, "raw_path": path.encode(), "root_path": "",
                     "query_string": b"", "headers": headers}, receive, send)
    return sent


def _each_forwarded(forwarder: Forwarder, *requests: tuple[str, list[tuple[bytes, bytes]]]) -> list[list[dict]]:
    """What ``forwarder`` sends for each of ``requests`` in turn, on one event loop, its client closed after."""
    async def forward() -> list[list[dict]]:
        try:
            return [await _forwarded(forwarder, path, headers) for path, headers in requests]
        finally:
            await forwarder.client.aclose()

    return asyncio.run(forward())


def test_the_app_gets_only_the_headers_the_request_had_and_none_of_the_forwarders_own(app):
    """A request with no Accept-Encoding, User-Agent or Accept reaches the app with none, so the app never encodes
    an answer for a client that didn't ask."""
    [sent] = _each_forwarded(Forwarder(app.url), ("/admin", [(b"host", HOST.encode())]))

    assert sent[0]["status"] == 200
    assert app.last()["headers"] == [("host", HOST)], app.last()["headers"]


def test_a_cookie_one_browser_was_answered_with_never_goes_on_with_anothers_request(app):
    """The app sets cookies on one browser's answer, host-only ones among them; another browser's request, with no
    cookie of its own, reaches the app with none, and one with its own cookie with that alone."""
    host = [(b"host", HOST.encode())]
    answered, bare, own = _each_forwarded(Forwarder(app.url, names=[HOST]), ("/cookies", host), ("/admin", host),
                                          ("/admin", [*host, (b"cookie", b"admin=other")]))
    bare_seen = app.seen[-2]["headers"]

    set_cookies = [value for name, value in answered[0]["headers"] if name.lower() == b"set-cookie"]
    assert b"a=1; Path=/" in set_cookies and (bare[0]["status"], own[0]["status"]) == (200, 200)
    assert bare_seen == [("host", HOST)], bare_seen
    assert app.last()["headers"] == [("host", HOST), ("cookie", "admin=other")], app.last()["headers"]


def test_ask_app_asks_as_the_site_under_its_name_with_a_header_only_its_own_calls_send(app):
    """The site's own call reaches the app under the site's name, with the headers it gives, a dropped one among them,
    and its method and body; a cookie the app answered one call with goes on with no later one."""
    site = _site(app.url)()

    async def ask() -> tuple:
        try:
            return (await site.ask_app("/cookies", headers={"X-Auto-Login": "admin:test"}),
                    await site.ask_app("admin/form", method="POST", content=b"user=amy"))
        finally:
            await site.fallback_app().client.aclose()

    signed, posted = asyncio.run(ask())
    first, second = (dict(seen["headers"]) for seen in app.seen[-2:])

    assert (signed.status_code, first["host"], first["x-auto-login"]) == (200, HOST, "admin:test")
    assert signed.headers.get_list("set-cookie")[0] == "a=1; Path=/"
    assert (posted.status_code, app.seen[-1]["method"], app.seen[-1]["path"], app.seen[-1]["body"]) == (
        200, "POST", "/admin/form", b"user=amy")
    assert "cookie" not in second and second["host"] == HOST, second


def test_a_body_goes_to_the_app_whole_sent_at_once_or_chunked(app, client):
    client.post("/admin/form", content=b"user=amy&next=%2F", headers={"Content-Type": "application/x-www-form-urlencoded"})
    sent = app.last()
    client.put("/admin/upload", content=iter([b"part one, ", b"part two"]))
    chunked = app.last()

    assert (sent["method"], sent["body"]) == ("POST", b"user=amy&next=%2F")
    assert dict(sent["headers"])["content-type"] == "application/x-www-form-urlencoded"
    assert (chunked["method"], chunked["body"]) == ("PUT", b"part one, part two")


def test_the_answer_comes_back_as_the_app_gave_it_but_a_cookie_tied_to_the_sites_name_is_the_hosts_alone(app, client):
    """Redirects unfollowed, bodies still encoded, chunks streamed, cookies as the app set them, but for one whose
    Domain names the site, which is the host's alone, as agentenv sets a sign-in's cookies."""
    moved = client.get("/redirect", follow_redirects=False)
    cookies = client.get("/cookies")
    with client.stream("GET", "/zipped") as zipped:
        raw = b"".join(zipped.iter_raw())
    streamed = client.get("/streamed")

    assert (moved.status_code, moved.headers["location"]) == (302, f"http://{HOST}/elsewhere")
    assert cookies.headers.get_list("set-cookie") == ["a=1; Path=/", "b=2; Path=/admin; HttpOnly", "c=3; Path=/",
                                                      "d=4", "e=5; Domain=other.example; Path=/"]
    assert (zipped.headers["content-encoding"], raw) == ("gzip", ZIPPED)
    assert streamed.content == b"one,two,three"


def test_every_method_reaches_the_app_head_and_options_included(app, client):
    head = client.head("/admin")
    options = client.options("/admin")
    deleted = client.delete("/admin/item/7")

    assert [seen["method"] for seen in app.seen] == ["HEAD", "OPTIONS", "DELETE"]
    assert (head.status_code, head.content, options.status_code, deleted.status_code) == (200, b"", 200, 200)


def test_the_card_names_the_sites_sign_in_headers_and_a_site_with_none_names_none(client):
    card = client.get("/.well-known/agent-env.json").json()
    params = next(ext for ext in card["capabilities"]["extensions"] if ext["uri"] == WEBSITE_EXTENSION_URI)["params"]

    @website(domains=[HOST], builtin_routes=False, clock="real")
    class Bare(ForwardingSite):
        pass

    bare = Bare().extension_params(WEBSITE_EXTENSION_URI)

    assert params[WEBSITE_SIGN_IN_HEADERS_PARAM] == ["X-Auto-Login"]
    assert WEBSITE_SIGN_IN_HEADERS_PARAM not in bare


def test_the_card_and_a_data_plane_of_data_get_alone_stay_the_sites_and_api_session_reaches_the_app(app, client):
    card = client.get("/.well-known/agent-env.json").json()
    exported = client.post("/agentenv", json={"jsonrpc": "2.0", "id": 1, "method": "data/get", "params": {}}).json()
    reset = client.post("/agentenv", json={"jsonrpc": "2.0", "id": 2, "method": "data/reset", "params": {}}).json()
    signed_in = client.get("/api/session")

    site = next(ext for ext in card["capabilities"]["extensions"] if ext["uri"] == WEBSITE_EXTENSION_URI)
    assert card["capabilities"]["operations"] == ["data/get"] and exported["result"]["parts"] == []
    assert reset["error"]["code"] == -32601
    assert [seen["path"] for seen in app.seen] == ["/api/session"] and signed_in.content == PAGE
    assert "session_path" not in site["params"] and "session_cookie" not in site["params"]


def test_its_session_state_signs_a_browser_in_with_the_cookies_it_chose(client):
    card = client.get("/.well-known/agent-env.json").json()
    op = find_extension_method(card, WEBSITE_EXTENSION_URI, "session")
    answer = client.request(op.get("method", "POST"), op["endpoint"], json={"user": None}).json()

    assert answer == {"cookies": [{"name": "admin", "value": "s1", "path": "/admin", "max_age": 3600}],
                      "local_storage": []}


def test_an_app_not_answering_is_a_502_and_the_site_not_ready_until_its_ready_path_answers(app):
    gone = App()
    url = gone.url
    gone.close()
    with TestClient(_site(url)().create_app(), headers={"X-Forwarded-Host": HOST}) as down:
        answer = down.get("/admin")
        ready = down.get(_ready_endpoint(down)).json()

    app.status = 503
    with TestClient(_site(app.url)().create_app()) as failing:
        failing_ready = failing.get(_ready_endpoint(failing)).json()
    app.status = 200
    with TestClient(_site(app.url)().create_app()) as up:
        up_ready = up.get(_ready_endpoint(up)).json()

    assert (answer.status_code, answer.json()["code"]) == (502, "upstream_unavailable") and url in answer.json()["error"]
    assert ready["ready"] is False and "upstream_answers" in ready["problems"][0], ready
    assert failing_ready["ready"] is False and "answered 503" in failing_ready["problems"][0], failing_ready
    assert up_ready["ready"] is True
    assert (app.last()["path"], dict(app.last()["headers"])["host"]) == ("/admin", HOST)


def _ready_endpoint(client: TestClient) -> str:
    return find_extension_method(client.get("/.well-known/agent-env.json").json(), WEBSITE_EXTENSION_URI,
                                 "ready")["endpoint"]


def test_a_forwarded_prefix_leaves_the_app_the_sites_own_path(app, client):
    client.get("/admin/x%2Fy", headers={"X-Forwarded-Prefix": "/website/shop"})

    assert app.last()["path"] == "/admin/x%2Fy"


def test_a_websocket_is_refused(client):
    with pytest.raises(WebSocketDisconnect) as refused:
        with client.websocket_connect("/socket"):
            pass

    assert refused.value.code == WEBSOCKET_REFUSED


def test_a_real_clock_serves_no_clock_v1_stamps_no_universe_time_and_declares_itself(app, client):
    card = client.get("/.well-known/agent-env.json").json()
    page = client.get("/admin")

    uris = [ext["uri"] for ext in card["capabilities"]["extensions"]]
    site = next(ext for ext in card["capabilities"]["extensions"] if ext["uri"] == WEBSITE_EXTENSION_URI)
    assert CLOCK_EXTENSION_URI not in uris and site["params"]["clock"] == "real"
    assert UNIVERSE_NOW_HEADER not in page.headers


def test_the_conformance_check_skips_the_clock_checks_of_a_site_with_a_real_clock_and_passes_it(app, client):
    results = {result.name: result for result in conformance.check("http://testserver", client=client)}

    assert (results["clock"].status, results["universe_now"].status) == ("skipped", "skipped")
    assert "real clock" in results["clock"].detail
    assert all(result.status in ("passed", "skipped") for result in results.values()), {
        name: (result.status, result.detail) for name, result in results.items()}


def test_a_plain_site_keeps_its_built_in_routes_and_the_universe_clock_by_default():
    @website(domains=["notes.test"])
    class Notes(WebsiteEnvironment):
        pass

    with TestClient(Notes().create_app()) as client:
        card = client.get("/.well-known/agent-env.json").json()
        session = client.get("/api/session")
        page = client.get("/missing")

    uris = [ext["uri"] for ext in card["capabilities"]["extensions"]]
    site = next(ext for ext in card["capabilities"]["extensions"] if ext["uri"] == WEBSITE_EXTENSION_URI)
    assert CLOCK_EXTENSION_URI in uris and "clock" not in site["params"] and site["params"]["session_path"]
    assert session.status_code == 200 and UNIVERSE_NOW_HEADER in page.headers


def test_without_built_in_routes_api_session_is_the_apps_path():
    @website(domains=["notes.test"], builtin_routes=False)
    class Bare(WebsiteEnvironment):
        pass

    with TestClient(Bare().create_app()) as client:
        assert client.get("/api/session").status_code == 404


@pytest.mark.parametrize("declared, problem", [
    ({"clock": "virtual"}, "clock is 'universe' or 'real'"),
    ({"clock": "real", "clock_patch": True}, "a site with a real clock"),
    ({"builtin_routes": "no"}, "builtin_routes is True or False"),
], ids=["unknown-clock", "patch-on-real", "routes-not-bool"])
def test_website_refuses_a_clock_or_routes_switch_it_cant_honor(declared, problem):
    with pytest.raises(ValueError, match=problem):
        website(domains=["notes.test"], **declared)
