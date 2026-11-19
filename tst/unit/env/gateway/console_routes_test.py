"""The env's SQL consoles (pgweb, db-mcp) are served behind /svc/<key>/ only with the deploy's control token, which never
reaches them, and not at all by a gateway started without one."""
from __future__ import annotations

import asyncio
from urllib.parse import unquote, urlsplit

import httpx
import pytest

from agent_env.env.gateway import rest_proxy as rest_proxy_module
from agent_env.env.gateway.constants import CONSOLE_URLS_ENV
from agent_env.env.gateway.entrypoint import parse_name_url_map
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
        transport = httpx.MockTransport(self._answer)
        monkeypatch.setattr(rest_proxy_module.httpx, "AsyncClient", lambda **kw: _RealAsyncClient(transport=transport, **kw))

    def _answer(self, request: httpx.Request) -> httpx.Response:
        self.requests.append((request.url.host, request.url.path, request.headers.get("authorization")))

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
