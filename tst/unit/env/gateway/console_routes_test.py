"""The env's SQL consoles (pgweb, db-mcp) are served behind /svc/<key>/ only with the deploy's control token, which never
reaches them, or on pgweb a session minted from it, and not at all by a gateway started without one."""
from __future__ import annotations

import asyncio
import logging
import time
from urllib.parse import quote, unquote, urlsplit

import httpx
import pytest
from uvicorn.logging import AccessFormatter, DefaultFormatter

from agent_env.env.gateway import rest_proxy as rest_proxy_module
from agent_env.env.gateway.console_session import COOKIE_PREFIX, SESSION_SECONDS, ConsoleSession, LinkRedaction
from agent_env.env.gateway.constants import CONSOLE_URLS_ENV, CONTROL_ROUTES
from agent_env.env.gateway.entrypoint import parse_frame_ancestors, parse_name_url_map
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


def _gateway(token: str | None, consoles: dict[str, str] | None = None, ancestors: tuple[str, ...] = ()) -> Gateway:
    return Gateway(host="127.0.0.1", port=0, server_name="unit", internal_mcp_servers=[], control_token=token,
                   rest_proxy_urls=dict(BACKING), website_urls={"shop": BACKING["shop"]},
                   console_urls=CONSOLES if consoles is None else consoles, console_frame_ancestors=ancestors)


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
async def test_a_write_through_a_console_opens_no_write_window(monkeypatch):
    """Only a holder of the control token reaches a console, so what it writes there is credited to no role; a backing
    server's write beside it still is."""
    upstream = _Upstream(monkeypatch)
    gw = _gateway(TOKEN)
    changelog, events = iter(range(1, 100)), []

    async def log_event(event):
        events.append(event)
        return "event"

    monkeypatch.setattr(gw, "_query_changelog_id", lambda: next(changelog))
    monkeypatch.setattr(gw, "_log_event", log_event)
    for path in ("/svc/db-mcp/mcp", "/svc/pgweb/api/query"):
        assert (await _send(gw, "POST", path, {"Authorization": f"Bearer {TOKEN}"}))[0] == 200
    assert (await _send(gw, "POST", "/svc/shop/api/orders", {}))[0] == 200
    assert [(e["event_type"], e["service"]) for e in events] == [("http_request", "shop")]
    assert upstream.console_hosts == ["db-mcp", "pgweb"]


@pytest.mark.parametrize("value, error", [("pgweb", "Invalid CONSOLE_URLS entry: 'pgweb'"),
                                          ("pgweb=http://a:1,pgweb=http://b:2", "Duplicate service name in CONSOLE_URLS")])
def test_a_bad_console_entry_names_its_own_variable(monkeypatch, value, error):
    monkeypatch.setenv(CONSOLE_URLS_ENV, value)
    with pytest.raises(ValueError, match=f"^{error}"):
        parse_name_url_map(CONSOLE_URLS_ENV)


@pytest.mark.asyncio
async def test_a_link_sets_a_pgweb_only_cookie_and_sends_the_browser_to_the_console(monkeypatch):
    upstream = _Upstream(monkeypatch)
    status, headers = await _send(_gateway(TOKEN), "GET", f"/svc/pgweb/?pgweb_link={_mint().link()}", {})
    assert status == 303
    assert headers[b"location"] == b"./"
    cookie = headers[b"set-cookie"].decode()
    assert cookie.startswith(f"{_mint().cookie_name}=") and _mint().cookie_name.startswith(COOKIE_PREFIX)
    assert {"Path=/", "Secure", "HttpOnly", "SameSite=None", "Partitioned", f"Max-Age={SESSION_SECONDS}"} <= {
        part.strip() for part in cookie.split(";")}
    assert headers[b"content-security-policy"] == b"frame-ancestors 'self'"
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
    pytest.param(lambda: (link := _mint().link())[:-1] + ("B" if link.endswith("A") else "A"), id="forged"),
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
    pytest.param(lambda: f"{_mint().cookie_name}={_mint().link()}", id="a-link-is-not-a-cookie"),
    pytest.param(lambda: f"{_mint().cookie_name}=1.{'A' * 43}", id="forged"),
    pytest.param(lambda: f"{COOKIE_PREFIX}00000000={_cookie().partition('=')[2]}", id="this-deploys-value-another-name")])
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


@pytest.mark.asyncio
async def test_a_link_opens_once_however_its_expiry_is_spelled():
    gw, link = _gateway(TOKEN), _mint().link()
    assert (await _send(gw, "GET", f"/svc/pgweb/?pgweb_link={link}", {}))[0] == 303
    for again in (link, f"0{link}", f"00{link}"):
        status, headers = await _send(gw, "GET", f"/svc/pgweb/?pgweb_link={again}", {})
        assert status == 401 and b"set-cookie" not in headers, again


@pytest.mark.asyncio
async def test_a_link_to_the_console_without_its_slash_opens_it_relative_to_the_request(monkeypatch):
    upstream = _Upstream(monkeypatch)
    status, headers = await _send(_gateway(TOKEN), "GET", f"/svc/pgweb?pgweb_link={_mint().link()}", {})
    assert (status, headers[b"location"]) == (303, b"pgweb/")
    assert headers[b"set-cookie"].decode().startswith(f"{_mint().cookie_name}=")
    assert upstream.requests == []


@pytest.mark.asyncio
async def test_a_link_counts_only_under_its_literal_name(monkeypatch):
    """A spelling the access log's redaction doesn't know never opens the console, so no live link sits in the log."""
    upstream = _Upstream(monkeypatch)
    status, headers = await _send(_gateway(TOKEN), "GET", f"/svc/pgweb/?pgweb%5flink={_mint().link()}", {})
    assert status == 401 and b"set-cookie" not in headers
    assert upstream.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("credentials", [
    pytest.param(lambda: {"Sec-Fetch-Site": "none", "Sec-Fetch-Mode": "navigate", "Cookie": _cookie()}, id="cookie"),
    pytest.param(lambda: {"Authorization": f"Bearer {TOKEN}"}, id="token")])
async def test_a_stale_link_on_a_request_that_carries_a_session_or_the_token_opens_the_console(monkeypatch, credentials):
    upstream = _Upstream(monkeypatch)
    status, headers = await _send(_gateway(TOKEN), "GET", f"/svc/pgweb/?pgweb_link={_mint(shift=-200).link()}",
                                  credentials())
    assert status == 200 and b"set-cookie" not in headers
    assert upstream.requests == [("pgweb", "/", None)]


@pytest.mark.asyncio
async def test_a_session_past_half_its_life_gets_a_fresh_cookie_and_a_young_one_does_not(monkeypatch):
    _Upstream(monkeypatch)
    gw = _gateway(TOKEN)
    old = _cookie(_mint(shift=-(SESSION_SECONDS // 2 + 10)))
    status, headers = await _send(gw, "POST", "/svc/pgweb/api/query", {**SAME_ORIGIN, "Cookie": old})
    assert status == 200
    assert headers[b"set-cookie"].decode().startswith(f"{_mint().cookie_name}=")
    status, headers = await _send(gw, "POST", "/svc/pgweb/api/query", {**SAME_ORIGIN, "Cookie": _cookie()})
    assert status == 200 and b"set-cookie" not in headers


@pytest.mark.asyncio
async def test_two_deploys_sessions_live_side_by_side_and_neither_reaches_pgweb(monkeypatch):
    """Cookies ignore ports, so every local gateway's session sits in the same jar on localhost."""
    upstream = _Upstream(monkeypatch)
    other = _mint("another deployment's token")
    assert other.cookie_name != _mint().cookie_name
    jar = f"theme=dark; {_cookie(other)}; {_cookie()}"
    assert (await _send(_gateway(TOKEN), "POST", "/svc/pgweb/api/query", {**SAME_ORIGIN, "Cookie": jar}))[0] == 200
    assert (await _send(_gateway("another deployment's token"), "POST", "/svc/pgweb/api/query",
                        {**SAME_ORIGIN, "Cookie": jar}))[0] == 200
    assert upstream.cookies == ["theme=dark", "theme=dark"]


@pytest.mark.asyncio
@pytest.mark.parametrize("method, path, headers", [
    ("GET", "/svc/pgweb/", lambda: {"Authorization": f"Bearer {TOKEN}"}),
    ("POST", "/svc/pgweb/api/query", lambda: {}),
    ("GET", "/svc/pgweb/", lambda: {"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "navigate", "Cookie": _cookie()}),
    ("GET", "/svc/pgweb/?pgweb_link=x", lambda: {})], ids=["token", "refused", "navigation", "bad-link"])
async def test_every_pgweb_answer_names_the_origins_that_may_frame_it(monkeypatch, method, path, headers):
    _Upstream(monkeypatch)
    _, answered = await _send(_gateway(TOKEN, ancestors=("https://app.example.com", "https://*.example.org")),
                              method, path, headers())
    assert answered[b"content-security-policy"] == b"frame-ancestors 'self' https://app.example.com https://*.example.org"


@pytest.mark.asyncio
async def test_no_other_route_takes_the_consoles_frame_policy(monkeypatch):
    _Upstream(monkeypatch)
    _, answered = await _send(_gateway(TOKEN), "POST", "/svc/db-mcp/mcp", {"Authorization": f"Bearer {TOKEN}"})
    assert b"content-security-policy" not in answered


@pytest.mark.parametrize("spec, origins", [
    ("", ()), ("https://app.example.com  http://localhost:4100", ("https://app.example.com", "http://localhost:4100")),
    ("https://*.example.com", ("https://*.example.com",))])
def test_frame_ancestors_are_space_separated_origins(spec, origins):
    assert parse_frame_ancestors(spec) == origins


@pytest.mark.parametrize("spec", ["https://app.example.com/path", "app.example.com", "https://a; script-src *",
                                  "'none'", "https://exa mple.com", "javascript:alert(1)"])
def test_a_frame_ancestor_that_is_not_an_origin_stops_the_gateway(spec):
    with pytest.raises(ValueError, match="CONSOLE_FRAME_ANCESTORS"):
        parse_frame_ancestors(spec)


@pytest.mark.parametrize("seconds", [0, -1, 301])
def test_a_link_lasts_one_to_three_hundred_seconds(seconds):
    with pytest.raises(ValueError, match="1 to 300 seconds"):
        _mint().link(seconds)
    assert _mint().opens(_mint().link(300))


@pytest.mark.asyncio
async def test_two_links_minted_in_one_second_differ_and_each_opens_once():
    session = ConsoleSession(TOKEN, now=lambda: 1_800_000_000)
    first, second = session.link(), session.link()
    assert first != second
    gateway = _gateway(TOKEN)
    gateway._console_session._now = lambda: 1_800_000_000
    codes = [(await _send(gateway, "GET", f"/svc/pgweb/?pgweb_link={link}", {}))[0] for link in (first, second, first)]
    assert codes == [303, 303, 401]
