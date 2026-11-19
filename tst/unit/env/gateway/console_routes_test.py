"""The env's SQL consoles (pgweb, db-mcp) are served behind /svc/<key>/ only with the deploy's control token, which never
reaches them, and not at all by a gateway started without one."""
from __future__ import annotations

import asyncio
import logging
import time
from urllib.parse import quote, unquote, urlsplit

import httpx
import pytest
from uvicorn.logging import AccessFormatter, DefaultFormatter

from agent_env.env.gateway import rest_proxy as rest_proxy_module
from agent_env.env.gateway.console_session import COOKIE_NAME, ConsoleSession, LinkRedaction
from agent_env.env.gateway.constants import CONTROL_ROUTES
from agent_env.env.gateway.gateway import Gateway

TOKEN = "s3cret"
CONSOLES = {"pgweb": "http://pgweb:8081", "db-mcp": "http://db-mcp:8000"}
BACKING = {"mcp-svc": "http://svc:18765", "shop": "http://shop-backend:8000"}
_RealAsyncClient = httpx.AsyncClient

# Every spelling that reaches a console: its root, its query API, db-mcp's MCP endpoint, and the same paths with dot
# segments, percent-encoding, doubled slashes or no trailing slash.
CONSOLE_PATHS = [("GET", "/svc/pgweb/"), ("GET", "/svc/pgweb"), ("GET", "/svc/pgweb/api/query?query=SELECT%201"),
                 ("POST", "/svc/pgweb/api/query"), ("GET", "/svc/pgweb/static/js/app.js"),
                 ("GET", "/svc/pgweb/x/../api/query"), ("GET", "/svc/%70gweb/api/query"),
                 ("GET", "/svc/pgweb//api/query"), ("GET", "/svc/pgweb/public%2Fx/../api/query"),
                 ("POST", "/svc/db-mcp/mcp"), ("GET", "/svc/db-mcp/mcp"), ("DELETE", "/svc/db-mcp/mcp"),
                 ("POST", "/svc/db-mcp/./mcp"), ("POST", "/svc/db-%6dcp/mcp"), ("POST", "/svc/db-mcp/%6dcp")]


class _Upstream:
    """Every service behind the gateway: answers each request as an MCP server would, and keeps what reached it."""

    def __init__(self, monkeypatch):
        self.requests: list[tuple[str, str, str | None]] = []
        self.cookies: list[str | None] = []
        transport = httpx.MockTransport(self._answer)
        monkeypatch.setattr(rest_proxy_module.httpx, "AsyncClient", lambda **kw: _RealAsyncClient(transport=transport, **kw))

    def _answer(self, request: httpx.Request) -> httpx.Response:
        self.requests.append((request.url.host, request.url.path, request.headers.get("authorization")))
        self.cookies.append(request.headers.get("cookie"))

        async def body():
            yield b'event: message\ndata: {"jsonrpc": "2.0", "id": 1, "result": {}}\n\n'

        return httpx.Response(200, headers={"content-type": "text/event-stream", "mcp-session-id": "abc"}, content=body())

    @property
    def console_hosts(self) -> list[str]:
        return [host for host, _, _ in self.requests if host in ("pgweb", "db-mcp")]


async def _send(gw: Gateway, method: str, path: str, headers: dict[str, str]) -> tuple[int, dict[bytes, bytes]]:
    """One request through the gateway's served app with its path as uvicorn hands it: percent-decoded, its dot
    segments and doubled slashes kept. Returns the status and the response headers."""
    url = urlsplit(path)
    scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"}, "http_version": "1.1", "method": method,
             "scheme": "http", "path": unquote(url.path), "raw_path": url.path.encode(),
             "query_string": url.query.encode(), "root_path": "", "server": ("gateway", 80), "client": ("agent", 1),
             "headers": [(b"content-type", b"application/json"),
                         *((name.lower().encode(), value.encode()) for name, value in headers.items())]}
    received, started = False, []

    async def receive():
        nonlocal received
        if not received:
            received = True
            return {"type": "http.request", "body": b'{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}',
                    "more_body": False}
        await asyncio.Event().wait()

    async def send(message):
        if message["type"] == "http.response.start":
            started.append((message["status"], dict(message.get("headers", ()))))

    await gw._asgi_app()(scope, receive, send)
    return started[0]


def _mint(token: str = TOKEN, shift: float = 0) -> ConsoleSession:
    """A deployment's session minter on a clock ``shift`` seconds off the gateway's."""
    return ConsoleSession(token, now=lambda: time.time() + shift)


def _cookie(session: ConsoleSession | None = None) -> str:
    return (session or _mint()).set_cookie().split(";")[0]


SAME_ORIGIN = {"Sec-Fetch-Site": "same-origin", "Sec-Fetch-Mode": "cors"}


def _gateway(token: str | None, consoles: dict[str, str] | None = None) -> Gateway:
    return Gateway(host="127.0.0.1", port=0, server_name="unit", internal_mcp_servers=[], control_token=token,
                   rest_proxy_urls=dict(BACKING), website_urls={"shop": BACKING["shop"]},
                   console_urls=CONSOLES if consoles is None else consoles)


@pytest.mark.asyncio
@pytest.mark.parametrize("method, path", CONSOLE_PATHS)
@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}, {"Authorization": TOKEN}],
                         ids=["none", "wrong", "not-bearer"])
async def test_a_console_refuses_a_request_without_the_token(monkeypatch, method, path, headers):
    upstream = _Upstream(monkeypatch)
    status, _ = await _send(_gateway(TOKEN), method, path, headers)
    assert status == 401
    assert upstream.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("method, path, host, sent", [
    ("GET", "/svc/pgweb/", "pgweb", "/"), ("GET", "/svc/pgweb/api/query", "pgweb", "/api/query"),
    ("GET", "/svc/pgweb/static/js/app.js", "pgweb", "/static/js/app.js"),
    ("GET", "/svc/pgweb/x/../api/query", "pgweb", "/api/query"),
    ("POST", "/svc/db-mcp/mcp", "db-mcp", "/mcp"), ("DELETE", "/svc/db-mcp/mcp", "db-mcp", "/mcp")])
async def test_a_console_answers_a_request_with_the_token_which_it_never_sees(monkeypatch, method, path, host, sent):
    upstream = _Upstream(monkeypatch)
    status, headers = await _send(_gateway(TOKEN), method, path, {"Authorization": f"Bearer {TOKEN}"})
    assert status == 200
    assert upstream.requests == [(host, sent, None)]
    assert headers[b"mcp-session-id"] == b"abc"


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/svc/mcp-svc/../pgweb/api/query", "/svc/shop/../db-mcp/mcp",
                                  "/svc/mcp-svc/x%2F..%2F..%2Fpgweb/api/query", "/svc/PGWEB/api/query",
                                  "/svc//pgweb/api/query", "/website/shop/../../svc/pgweb/api/query",
                                  "/svc/pgweb%2Fapi/query"])
@pytest.mark.parametrize("headers", [{}, {"Authorization": f"Bearer {TOKEN}"}], ids=["none", "token"])
async def test_no_other_route_reaches_a_console(monkeypatch, path, headers):
    upstream = _Upstream(monkeypatch)
    await _send(_gateway(TOKEN), "POST", path, headers)
    assert upstream.console_hosts == []


@pytest.mark.asyncio
@pytest.mark.parametrize("method, path", CONSOLE_PATHS)
async def test_a_gateway_started_without_a_token_serves_no_console(monkeypatch, method, path):
    upstream = _Upstream(monkeypatch)
    status, _ = await _send(_gateway(None), method, path, {})
    assert status in (404, 307)  # a bare /svc/pgweb is redirected to /svc/pgweb/, which is 404 too
    assert upstream.requests == []


@pytest.mark.asyncio
async def test_the_backing_servers_routes_are_unchanged_beside_the_consoles(monkeypatch):
    upstream = _Upstream(monkeypatch)
    status, _ = await _send(_gateway(TOKEN), "GET", "/svc/shop/api/orders", {})
    assert status == 200
    assert upstream.requests == [("shop-backend", "/api/orders", None)]


@pytest.mark.asyncio
async def test_consoles_alone_get_the_svc_route(monkeypatch):
    upstream = _Upstream(monkeypatch)
    gw = Gateway(host="127.0.0.1", port=0, server_name="unit", internal_mcp_servers=[], control_token=TOKEN,
                 console_urls=CONSOLES)
    assert (await _send(gw, "POST", "/svc/db-mcp/mcp", {}))[0] == 401
    assert (await _send(gw, "POST", "/svc/db-mcp/mcp", {"Authorization": f"Bearer {TOKEN}"}))[0] == 200
    assert upstream.requests == [("db-mcp", "/mcp", None)]


def test_a_console_cannot_share_a_backing_servers_key():
    with pytest.raises(ValueError, match="shop"):
        _gateway(TOKEN, {"shop": "http://pgweb:8081"})


@pytest.mark.asyncio
async def test_a_link_sets_a_pgweb_only_cookie_and_sends_the_browser_to_the_console(monkeypatch):
    upstream = _Upstream(monkeypatch)
    status, headers = await _send(_gateway(TOKEN), "GET", f"/svc/pgweb/?pgweb_link={_mint().link()}", {})
    assert status == 303
    assert headers[b"location"] == b"./"
    cookie = headers[b"set-cookie"].decode()
    assert cookie.startswith(f"{COOKIE_NAME}=")
    assert {"Path=/", "Secure", "HttpOnly", "SameSite=None", "Partitioned", "Max-Age=3600"} <= {
        part.strip() for part in cookie.split(";")}
    assert upstream.requests == []


def _in_arabic_indic_digits(link: str) -> str:
    """``link`` with its expiry's digits written as Arabic-Indic digits, which Python's int() reads as the same number."""
    expiry, _, signature = link.partition(".")
    return "".join(chr(0x0660 + int(digit)) for digit in expiry) + "." + signature


@pytest.mark.asyncio
@pytest.mark.parametrize("link", [
    pytest.param(lambda: _mint("another deployment's token").link(), id="other-deployment"),
    pytest.param(lambda: _mint(shift=-200).link(120), id="expired"),
    pytest.param(lambda: _mint(shift=1000).link(120), id="too-far-ahead"),
    pytest.param(lambda: _mint().link()[:-1] + "A", id="forged"),
    pytest.param(lambda: _cookie().partition("=")[2], id="a-cookie-is-not-a-link"),
    pytest.param(lambda: "", id="empty"),
    pytest.param(lambda: _in_arabic_indic_digits(_mint().link()), id="non-ascii-digits")])
async def test_a_link_this_deployment_did_not_mint_or_that_expired_is_refused(monkeypatch, link):
    upstream = _Upstream(monkeypatch)
    status, headers = await _send(_gateway(TOKEN), "GET", f"/svc/pgweb/?pgweb_link={quote(link())}", {})
    assert status == 401
    assert b"set-cookie" not in headers
    assert upstream.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("method, path, sent", [
    ("GET", "/svc/pgweb/", "/"), ("GET", "/svc/pgweb/api/query?query=SELECT%201", "/api/query"),
    ("POST", "/svc/pgweb/api/query", "/api/query"), ("GET", "/svc/pgweb/static/js/app.js", "/static/js/app.js")])
async def test_the_cookie_opens_pgweb_from_its_own_page_and_never_reaches_it(monkeypatch, method, path, sent):
    upstream = _Upstream(monkeypatch)
    status, _ = await _send(_gateway(TOKEN), method, path, {**SAME_ORIGIN, "Cookie": f"theme=dark; {_cookie()}"})
    assert status == 200
    assert upstream.requests == [("pgweb", sent, None)]
    assert upstream.cookies == ["theme=dark"]


@pytest.mark.asyncio
async def test_a_navigation_to_the_console_index_takes_the_cookie_from_any_site(monkeypatch):
    upstream = _Upstream(monkeypatch)
    status, _ = await _send(_gateway(TOKEN), "GET", "/svc/pgweb/",
                            {"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "navigate", "Cookie": _cookie()})
    assert status == 200
    assert upstream.requests == [("pgweb", "/", None)]
    assert upstream.cookies == [None]


@pytest.mark.asyncio
@pytest.mark.parametrize("method, path, fetch", [
    ("POST", "/svc/pgweb/api/query", {"Sec-Fetch-Site": "same-site", "Sec-Fetch-Mode": "cors"}),
    ("POST", "/svc/pgweb/api/query", {"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "no-cors"}),
    ("POST", "/svc/pgweb/api/query", {}),
    ("GET", "/svc/pgweb/api/query?query=DROP%20TABLE%20t", {"Sec-Fetch-Site": "same-site", "Sec-Fetch-Mode": "navigate"}),
    ("POST", "/svc/pgweb/", {"Sec-Fetch-Site": "same-site", "Sec-Fetch-Mode": "navigate"})])
async def test_the_cookie_does_not_count_on_a_request_another_site_sends(monkeypatch, method, path, fetch):
    """Every Modal tunnel is same-site with every other, so another deployment's page could otherwise write here."""
    upstream = _Upstream(monkeypatch)
    status, _ = await _send(_gateway(TOKEN), method, path, {**fetch, "Cookie": _cookie()})
    assert status == 401
    assert upstream.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("cookie", [
    pytest.param(lambda: _cookie(_mint("another deployment's token")), id="other-deployment"),
    pytest.param(lambda: _cookie(_mint(shift=-4000)), id="expired"),
    pytest.param(lambda: f"{COOKIE_NAME}={_mint().link()}", id="a-link-is-not-a-cookie"),
    pytest.param(lambda: f"{COOKIE_NAME}=1.{'A' * 43}", id="forged")])
async def test_a_cookie_this_deployment_did_not_set_or_that_expired_opens_nothing(monkeypatch, cookie):
    upstream = _Upstream(monkeypatch)
    status, _ = await _send(_gateway(TOKEN), "POST", "/svc/pgweb/api/query", {**SAME_ORIGIN, "Cookie": cookie()})
    assert status == 401
    assert upstream.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("method, path", [
    *[(method, path) for method, path in CONSOLE_PATHS if "db-" in path],
    *[("POST", route) for route in sorted(CONTROL_ROUTES)],
    ("POST", "/svc/db-mcp/../pgweb/api/query")])
async def test_the_cookie_opens_no_route_but_pgweb(monkeypatch, method, path):
    upstream = _Upstream(monkeypatch)
    status, _ = await _send(_gateway(TOKEN), method, path, {**SAME_ORIGIN, "Cookie": _cookie()})
    assert status in (401, 404)
    assert upstream.console_hosts == []


@pytest.mark.asyncio
async def test_the_bearer_token_still_opens_pgweb_and_the_cookie_still_does_not_reach_it(monkeypatch):
    upstream = _Upstream(monkeypatch)
    status, _ = await _send(_gateway(TOKEN), "POST", "/svc/pgweb/api/query",
                            {"Authorization": f"Bearer {TOKEN}", "Cookie": _cookie()})
    assert status == 200
    assert upstream.requests == [("pgweb", "/api/query", None)]
    assert upstream.cookies == [None]


@pytest.mark.asyncio
@pytest.mark.parametrize("token, consoles", [(TOKEN, {"db-mcp": CONSOLES["db-mcp"]}), (None, CONSOLES)],
                         ids=["no-pgweb", "no-token"])
async def test_a_gateway_with_no_pgweb_behind_its_token_takes_no_link(monkeypatch, token, consoles):
    upstream = _Upstream(monkeypatch)
    status, headers = await _send(_gateway(token, consoles), "GET", f"/svc/pgweb/?pgweb_link={_mint().link()}", {})
    assert status in (401, 404)
    assert b"set-cookie" not in headers
    assert upstream.console_hosts == []


@pytest.mark.parametrize("logger, fmt, args, line", [
    pytest.param("uvicorn.access", '%s - "%s %s HTTP/%s" %d',
                 lambda link: ("agent:1", "GET", f"/svc/pgweb/?pgweb_link={link}&x=1", "1.1", 303),
                 'INFO:     agent:1 - "GET /svc/pgweb/?pgweb_link=<redacted>&x=1 HTTP/1.1" 303 See Other',
                 id="uvicorn-access"),
    pytest.param("httpx", 'HTTP Request: %s %s "%s %d %s"',
                 lambda link: ("GET", httpx.URL(f"http://pgweb:8081/?pgweb_link={link}"), "HTTP/1.1", 200, "OK"),
                 'INFO:     HTTP Request: GET http://pgweb:8081/?pgweb_link=<redacted> "HTTP/1.1 200 OK"', id="httpx")])
def test_the_log_names_no_link_and_still_formats_its_line(logger, fmt, args, line):
    """Through the formatters uvicorn's log config gives these loggers: its access formatter reads the request line
    from the record's arguments."""
    link = _mint().link()
    record = logging.LogRecord(logger, logging.INFO, __file__, 0, fmt, args(link), None)
    assert LinkRedaction().filter(record)
    formatter = (AccessFormatter('%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s', use_colors=False)
                 if logger == "uvicorn.access" else DefaultFormatter("%(levelprefix)s %(message)s", use_colors=False))
    formatted = formatter.format(record)
    assert link not in formatted
    assert formatted == line