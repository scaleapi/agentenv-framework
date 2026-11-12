"""The /svc and /website reverse proxy streams upstream bodies through instead of buffering them.

Drives the gateway's ASGI app directly with a mocked upstream (no sockets): the fake upstream
yields its body lazily, so a chunk reaching the downstream `send` before the upstream has
finished is the streaming property itself."""

from __future__ import annotations

import asyncio
import doctest
import json
from urllib.parse import unquote, urlsplit

import httpx
import pytest
from starlette.testclient import TestClient
from agentenv_protocol.website import ApiContext, WebsiteEnvironment, api, website

from agent_env.env.gateway import AGENT_ENV_ROLE_META_KEY, AGENT_ENV_SESSION_META_KEY
from agent_env.env.gateway import gateway as gateway_module
from agent_env.env.gateway import rest_proxy as rest_proxy_module
from agent_env.env.gateway.gateway import Gateway

UPSTREAM = "http://upstream"
_RealAsyncClient = httpx.AsyncClient  # captured before any test patches the module attribute


def _gateway(**kw) -> Gateway:
    return Gateway(host="127.0.0.1", port=0, server_name="t", internal_mcp_servers=[],
                   rest_proxy_urls={"mcp-svc": UPSTREAM}, **kw)


class _Upstream:
    """A MockTransport upstream whose body is produced lazily and records what it saw."""

    def __init__(self, chunks, *, headers=None, status=200, gate: asyncio.Event | None = None, error=None):
        self.chunks, self.headers, self.status, self.gate, self.error = chunks, headers or {}, status, gate, error
        if isinstance(self.headers, dict):
            self.headers = list(self.headers.items())
        self.requests: list[httpx.Request] = []
        self.produced = 0
        self.closed = False

    @property
    def forwarded(self) -> list[httpx.Request]:
        """What it saw but the card reads the gateway makes of a website it fronts, for the paths its card reserves."""
        return [request for request in self.requests if request.url.path != "/.well-known/agent-env.json"]

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.error:
            raise self.error

        async def body():
            try:
                for i, chunk in enumerate(self.chunks):
                    if i == 1 and self.gate is not None:
                        await self.gate.wait()  # hold the rest of the body until the test says so
                    self.produced += 1
                    yield chunk
            finally:
                self.closed = True

        return httpx.Response(self.status, headers=self.headers, content=body())

    def install(self, monkeypatch):
        transport = httpx.MockTransport(self.handler)
        up = self
        self.clients_closed = 0
        self.responses: list[httpx.Response] = []

        class _RecordingClient(_RealAsyncClient):
            async def aclose(self):
                up.clients_closed += 1
                await super().aclose()

        real_handler = self.handler

        def handler(request):
            resp = real_handler(request)
            up.responses.append(resp)
            return resp

        transport = httpx.MockTransport(handler)
        monkeypatch.setattr(rest_proxy_module.httpx, "AsyncClient",
                            lambda **kw: _RecordingClient(transport=transport, **kw))


async def _call(gw: Gateway, path: str, *, method="GET", headers=None, body=b"", disconnect_after: int | None = None,
                disconnect_before_response=False, raw_headers=None, root_path=""):
    """Run one request through the ASGI app; return (status, headers, chunks). `disconnect_after` = number of body
    chunks after which the downstream client goes away; `disconnect_before_response` = it is already gone when the
    response starts (and the status-line send suspends, as a real server's can under write flow control).
    `scope["path"]` is percent-decoded and `raw_path` is not, exactly as uvicorn builds them."""
    app = gw._mcp.streamable_http_app()
    url = urlsplit(path)
    scope = {
        "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"}, "http_version": "1.1",
        "method": method, "scheme": "http", "path": unquote(url.path), "raw_path": url.path.encode(),
        "query_string": url.query.encode(), "root_path": root_path, "server": ("gateway", 80), "client": ("c", 1),
        "headers": raw_headers or [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
    }
    sent: list[dict] = []
    chunks: list[bytes] = []
    disconnected = asyncio.Event()
    request_sent = False

    async def receive():
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        if not disconnect_before_response:
            await disconnected.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if disconnect_before_response:  # suspend so Starlette's cancel lands here, before the body iterator starts
            await asyncio.sleep(0)
        sent.append(message)
        if message["type"] == "http.response.body" and message.get("body"):
            chunks.append(message["body"])
            if disconnect_after is not None and len(chunks) >= disconnect_after:
                disconnected.set()

    try:
        await app(scope, receive, send)
    finally:
        disconnected.set()
    start = next((m for m in sent if m["type"] == "http.response.start"), None)
    if start is None:  # cancelled before the status line went out
        return None, {}, chunks
    headers_out = {k.decode(): v.decode("latin-1") for k, v in start["headers"]}
    headers_out["__raw__"] = [(k.decode(), v) for k, v in start["headers"]]
    return start["status"], headers_out, chunks


@pytest.mark.asyncio
async def test_first_chunk_reaches_the_client_before_the_upstream_has_finished(monkeypatch):
    gate = asyncio.Event()
    up = _Upstream([b"a" * 1024, b"b" * 1024, b"c" * 1024], gate=gate,
                   headers={"content-length": "3072", "content-type": "application/zip"})
    up.install(monkeypatch)
    gw = _gateway()
    app = gw._mcp.streamable_http_app()

    seen_first = asyncio.Event()
    got: list[bytes] = []

    async def receive():
        await asyncio.sleep(3600)

    async def send(message):
        if message["type"] == "http.response.body" and message.get("body"):
            got.append(message["body"])
            seen_first.set()

    scope = {"type": "http", "asgi": {"spec_version": "2.3"}, "http_version": "1.1", "method": "GET", "scheme": "http",
             "path": "/svc/mcp-svc/export-snapshot", "raw_path": b"/svc/mcp-svc/export-snapshot", "query_string": b"stream=1",
             "root_path": "", "server": ("g", 80), "client": ("c", 1), "headers": []}

    async def first_request():
        return {"type": "http.request", "body": b"", "more_body": False}

    calls = [first_request]

    async def receive2():
        if calls:
            return await calls.pop()()
        await asyncio.sleep(3600)

    task = asyncio.create_task(app(scope, receive2, send))
    await asyncio.wait_for(seen_first.wait(), 5)
    assert up.produced == 1 and got == [b"a" * 1024]  # first byte out while the upstream body is still open
    gate.set()
    await asyncio.wait_for(task, 5)
    assert b"".join(got) == b"a" * 1024 + b"b" * 1024 + b"c" * 1024
    assert up.closed


@pytest.mark.asyncio
async def test_body_is_forwarded_byte_for_byte_with_length_and_encoding(monkeypatch):
    gz = bytes(range(256)) * 4
    up = _Upstream([gz[:512], gz[512:]], headers={"content-length": str(len(gz)), "content-encoding": "gzip",
                                                    "content-type": "application/zip", "x-upstream": "yes",
                                                    "connection": "keep-alive", "transfer-encoding": "chunked"})
    up.install(monkeypatch)
    status, headers, chunks = await _call(_gateway(), "/svc/mcp-svc/export-snapshot?stream=1")
    assert status == 200
    assert b"".join(chunks) == gz  # undecoded
    assert headers["content-length"] == str(len(gz)) and headers["content-encoding"] == "gzip"
    assert headers["content-type"] == "application/zip" and headers["x-upstream"] == "yes"
    assert "transfer-encoding" not in headers and "connection" not in headers  # hop-by-hop dropped
    req = up.requests[0]
    assert str(req.url) == f"{UPSTREAM}/export-snapshot?stream=1"
    assert "date" not in headers and "server" not in headers  # uvicorn adds the gateway's own


@pytest.mark.asyncio
async def test_upstream_is_only_asked_for_encodings_the_caller_accepts(monkeypatch):
    up = _Upstream([b"{}"], headers={"content-type": "application/json"})
    up.install(monkeypatch)
    gw = _gateway()
    await _call(gw, "/svc/mcp-svc/agentenv")
    assert up.requests[0].headers["accept-encoding"] == "identity"  # caller sent none: no gzip surprise
    await _call(gw, "/svc/mcp-svc/agentenv", headers={"accept-encoding": "gzip"})
    assert up.requests[1].headers["accept-encoding"] == "gzip"  # caller's own preference passes through
    assert "host" not in {k.lower() for k in up.requests[1].headers} or up.requests[1].headers["host"] == "upstream"


@pytest.mark.asyncio
async def test_request_body_method_and_status_pass_through(monkeypatch):
    up = _Upstream([b'{"ok":true}'], status=201, headers={"content-type": "application/json"})
    up.install(monkeypatch)
    status, headers, chunks = await _call(_gateway(), "/svc/mcp-svc/api/reset", method="POST", body=b'{"x":1}',
                                          headers={"content-type": "application/json"})
    assert status == 201 and b"".join(chunks) == b'{"ok":true}'
    req = up.requests[0]
    assert req.method == "POST" and req.content == b'{"x":1}' and req.headers["content-type"] == "application/json"


@pytest.mark.asyncio
async def test_connect_failure_before_headers_is_a_502(monkeypatch):
    up = _Upstream([], error=httpx.ConnectError("refused"))
    up.install(monkeypatch)
    status, headers, chunks = await _call(_gateway(), "/svc/mcp-svc/export-snapshot")
    assert status == 502 and b"refused" in b"".join(chunks) and headers["content-type"] == "application/json"


@pytest.mark.asyncio
async def test_unknown_service_is_a_404_without_contacting_anything(monkeypatch):
    up = _Upstream([b"x"])
    up.install(monkeypatch)
    status, _, chunks = await _call(_gateway(), "/svc/nope/anything")
    assert status == 404 and b"Unknown service" in b"".join(chunks) and up.requests == []


@pytest.mark.asyncio
async def test_downstream_disconnect_closes_the_upstream_stream(monkeypatch):
    gate = asyncio.Event()
    up = _Upstream([b"a" * 1024, b"b" * 1024, b"c" * 1024], gate=gate, headers={"content-length": "3072"})
    up.install(monkeypatch)
    # The client goes away after the first chunk while the upstream is still holding the rest.
    task = asyncio.create_task(_call(_gateway(), "/svc/mcp-svc/export-snapshot", disconnect_after=1))
    await asyncio.wait_for(task, 5)
    assert up.produced == 1  # the remaining chunks were never pulled
    assert up.clients_closed == 1 and up.responses[0].is_closed  # the proxy's own finally ran: response and client closed


@pytest.mark.asyncio
async def test_website_routes_use_the_same_handler(monkeypatch):
    up = _Upstream([b"<html>hi</html>"], headers={"content-type": "text/html; charset=utf-8", "content-length": "15"})
    up.install(monkeypatch)
    gw = _gateway(website_urls={"web": "http://web"})
    status, headers, chunks = await _call(gw, "/website/web/assets/app.js")
    assert status == 200 and b"".join(chunks) == b"<html>hi</html>" and headers["content-type"].startswith("text/html")
    assert str(up.forwarded[0].url) == "http://web/assets/app.js"
    status, _, _ = await _call(gw, "/website/web")
    assert status == 200 and str(up.forwarded[1].url) == "http://web"


@pytest.mark.asyncio
async def test_repeated_and_non_latin1_headers_are_forwarded_intact(monkeypatch):
    """Two Set-Cookie lines stay two lines, and a UTF-8 header value neither raises nor is mangled."""
    up = _Upstream([b"x"], headers=[("set-cookie", "a=1; Path=/"), ("set-cookie", "b=2; Path=/"),
                                     (b"content-disposition", 'attachment; filename="报告.zip"'.encode("utf-8")),  # what a real upstream puts on the wire
                                     ("date", "Sun, 20 Sep 2026 00:00:00 GMT")])
    up.install(monkeypatch)
    status, headers, _ = await _call(_gateway(), "/svc/mcp-svc/x")
    assert status == 200
    cookies = [v for k, v in headers["__raw__"] if k == "set-cookie"]
    assert cookies == [b"a=1; Path=/", b"b=2; Path=/"]
    assert dict(headers["__raw__"])["content-disposition"] == 'attachment; filename="报告.zip"'.encode("utf-8")
    assert "date" not in dict(headers["__raw__"])


@pytest.mark.asyncio
async def test_query_string_and_repeated_request_headers_pass_through_verbatim(monkeypatch):
    up = _Upstream([b"{}"])
    up.install(monkeypatch)
    scope_headers = {"cookie": "a=1"}
    gw = _gateway()
    await _call(gw, "/svc/mcp-svc/search?id=1&id=2&q=%2Fx%3F", headers=scope_headers)
    assert str(up.requests[0].url) == f"{UPSTREAM}/search?id=1&id=2&q=%2Fx%3F"
    # repeated request headers keep their multiplicity on the way upstream
    await _call(gw, "/svc/mcp-svc/x", raw_headers=[(b"cookie", b"a=1"), (b"cookie", b"b=2")])
    assert up.requests[1].headers.get_list("cookie") == ["a=1", "b=2"]


@pytest.mark.asyncio
async def test_timeout_before_headers_names_the_error_class(monkeypatch):
    up = _Upstream([], error=httpx.ReadTimeout(""))  # str() of an httpx timeout is empty
    up.install(monkeypatch)
    status, _, chunks = await _call(_gateway(), "/svc/mcp-svc/agentenv")
    assert status == 502 and b"ReadTimeout" in b"".join(chunks)


@pytest.mark.asyncio
async def test_mid_body_upstream_failure_is_logged_with_bytes_and_propagates(monkeypatch, caplog):
    class _Boom(Exception):
        pass

    up = _Upstream([b"a" * 1024, b"b" * 1024], headers={"content-length": "2048"})

    def handler(request):
        async def body():
            yield b"a" * 1024
            raise httpx.RemoteProtocolError("peer closed")
        return httpx.Response(200, headers=[("content-length", "2048")], content=body())
    up.handler = handler
    up.install(monkeypatch)
    with pytest.raises(httpx.RemoteProtocolError):
        await _call(_gateway(), "/svc/mcp-svc/export-snapshot")  # the cut body is the only signal HTTP allows after headers
    assert any("failed after 1024 bytes" in r.getMessage() and "RemoteProtocolError" in r.getMessage() for r in caplog.records)
    assert up.clients_closed == 1


@pytest.mark.asyncio
async def test_disconnect_before_the_first_chunk_still_closes_the_upstream(monkeypatch):
    """The disconnect is already queued when the response starts, so Starlette's cancel lands in the status-line
    send, before the iterator's first step; a never-started generator runs no `finally`, so the close cannot live only there."""
    gate = asyncio.Event()
    up = _Upstream([b"a" * 1024, b"b" * 1024], gate=gate, headers={"content-length": "2048"})
    up.install(monkeypatch)
    status, _, chunks = await asyncio.wait_for(_call(_gateway(), "/svc/mcp-svc/export-snapshot", disconnect_before_response=True), 5)
    assert status is None and chunks == [] and up.produced == 0  # cancelled before the body iterator ever ran
    assert up.clients_closed == 1 and up.responses[0].is_closed  # closed by the response's own lifetime


@pytest.mark.asyncio
async def test_percent_encoded_segments_reach_the_upstream_intact(monkeypatch):
    """uvicorn decodes `scope["path"]`; a target built from it would go out as `/files/a/b?c#d...`."""
    up = _Upstream([b"ok"])
    up.install(monkeypatch)
    status, _, _ = await _call(_gateway(), "/svc/mcp-svc/files/a%2Fb%3Fc%23d%C3%A9%2f?q=a%2Fb&q=c")
    assert status == 200
    assert up.requests[0].url.raw_path == b"/files/a%2Fb%3Fc%23d%C3%A9%2f?q=a%2Fb&q=c"


@pytest.mark.asyncio
async def test_website_paths_keep_their_encoding_and_the_bare_root_still_maps_to_the_base_url(monkeypatch):
    up = _Upstream([b"ok"])
    up.install(monkeypatch)
    gw = _gateway(website_urls={"web": "http://web"})
    await _call(gw, "/website/web/a%2Fb/c%20d")
    await _call(gw, "/website/web")
    await _call(gw, "/website/web/")
    assert [r.url.raw_path for r in up.forwarded] == [b"/a%2Fb/c%20d", b"/", b"/"]


@pytest.mark.asyncio
async def test_mounted_under_a_root_path_strips_only_the_route_prefix(monkeypatch):
    """`root_path` is decoded text while `raw_path` keeps its escapes, so the mount prefix is skipped by segment count."""
    up = _Upstream([b"ok"])
    up.install(monkeypatch)
    await _call(_gateway(), "/gw/svc/mcp-svc/x%2Fy", root_path="/gw")
    await _call(_gateway(), "/gw%20x/v1/svc/mcp-svc/x%2Fy", root_path="/gw x/v1")
    assert [r.url.raw_path for r in up.requests] == [b"/x%2Fy", b"/x%2Fy"]


@pytest.mark.asyncio
async def test_an_escaped_slash_in_the_route_prefix_is_refused_not_misrouted(monkeypatch):
    """Starlette routes `/svc/mcp-svc%2Fx/y` to `mcp-svc` with path `x/y`, but the raw split would cut at the wrong slash
    (`main` forwarded the decoded `/x/y`). The decoded remainder no longer matches, so the gateway answers 400."""
    up = _Upstream([b"ok"])
    up.install(monkeypatch)
    gw = _gateway(website_urls={"web": "http://web"})
    for path in ("/svc/mcp-svc%2Fx/y", "/website/web%2Findex.html"):
        status, _, chunks = await _call(gw, path)
        assert status == 400 and path.encode() in b"".join(chunks)  # names the request-target
    assert up.forwarded == []  # nothing was forwarded


def test_gateway_module_doctests_pass():
    """CI does not collect doctests, and `raw_remainder`'s example is its explanation, so keep it true."""
    results = [doctest.testmod(module) for module in (gateway_module, rest_proxy_module)]
    assert sum(r.attempted for r in results) >= 1 and all(r.failed == 0 for r in results)


@pytest.mark.asyncio
async def test_upstream_requests_carry_a_short_connect_timeout_and_the_data_plane_read_ceiling(monkeypatch):
    up = _Upstream([b"ok"])
    up.install(monkeypatch)
    await _call(_gateway(), "/svc/mcp-svc/x")
    ceiling = Gateway.REST_PROXY_TIMEOUT_S
    assert up.requests[0].extensions["timeout"] == {
        "connect": Gateway.REST_PROXY_CONNECT_TIMEOUT_S, "read": ceiling, "write": ceiling, "pool": ceiling,
    }


@pytest.mark.asyncio
async def test_connect_timeout_maps_to_the_502_body_naming_the_error(monkeypatch):
    up = _Upstream([], error=httpx.ConnectTimeout("blackholed upstream"))
    up.install(monkeypatch)
    status, _, chunks = await _call(_gateway(), "/svc/mcp-svc/x")
    assert status == 502 and b"ConnectTimeout" in b"".join(chunks)


# --- The role header and the JSON-RPC `_meta` are the gateway's to write, not the client's ---


@pytest.mark.asyncio
async def test_the_client_s_role_headers_are_replaced_by_the_gateway_s_attribution(monkeypatch):
    up = _Upstream([b"{}"])
    up.install(monkeypatch)
    status, _, _ = await _call(_gateway(), "/svc/mcp-svc/x", raw_headers=[
        (b"agentenv-role", b"alice@example.com"), (b"AgentEnv-Role", b"bob@example.com"), (b"cookie", b"a=1")])
    assert status == 200
    assert up.requests[0].headers.get_list("agentenv-role") == ["alice@example.com"]  # first value wins; one entry survives
    assert up.requests[0].headers["cookie"] == "a=1"


@pytest.mark.asyncio
async def test_an_absent_or_blank_role_header_is_the_default_role_as_step_attributes_it(monkeypatch):
    up = _Upstream([b"{}"])
    up.install(monkeypatch)
    gw = _gateway()
    await _call(gw, "/svc/mcp-svc/x")
    await _call(gw, "/svc/mcp-svc/x", headers={"AgentEnv-Role": "   "})
    assert [r.headers.get_list("agentenv-role") for r in up.requests] == [["default"], ["default"]]
    seen: list[str] = []
    monkeypatch.setattr(gw, "_step_list_tools", lambda role: (seen.append(role), gw._json_response({"tools": []}, 200))[1])
    try:
        status, _, _ = await _call(gw, "/step", method="POST", body=b'{"action": "list_tools"}', headers={"AgentEnv-Role": "   "})
    finally:
        await gw._close_child_sessions()
    assert status == 200 and seen == ["default"]


@pytest.mark.asyncio
async def test_the_website_proxy_stamps_the_role_too(monkeypatch):
    up = _Upstream([b"ok"])
    up.install(monkeypatch)
    gw = _gateway(website_urls={"web": "http://web"})
    await _call(gw, "/website/web/assets/app.js", headers={"AgentEnv-Role": "viewer"})
    await _call(gw, "/website/web")
    assert [r.headers.get_list("agentenv-role") for r in up.forwarded] == [["viewer"], ["default"]]


def _rpc(**params) -> dict:
    return {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": params}


@pytest.mark.asyncio
async def test_a_proxied_child_mcp_post_carries_the_gateway_s_role_in_every_request_s_meta(monkeypatch):
    up = _Upstream([b"{}"])
    up.install(monkeypatch)
    gw = _gateway()
    forged = {AGENT_ENV_ROLE_META_KEY: "victim@example.com", AGENT_ENV_SESSION_META_KEY: "forged"}
    one = json.dumps(_rpc(name="t", arguments={}, _meta=forged)).encode()
    batch = json.dumps([_rpc(name="t", arguments={}), {"jsonrpc": "2.0", "method": "notifications/initialized"}]).encode()
    for path, body in [("/svc/mcp-svc/mcp", one), ("/svc/mcp-svc/mcp/", batch), ("/svc/mcp-svc/x/../mcp", one)]:
        status, _, _ = await _call(gw, path, method="POST", body=body, headers={
            "AgentEnv-Role": "alice@example.com", "content-type": "application/json", "content-length": str(len(body))})
        assert status == 200
    sent = [json.loads(r.content) for r in up.requests]
    stamp = {AGENT_ENV_ROLE_META_KEY: "alice@example.com"}
    assert sent[0]["params"]["_meta"] == stamp  # the forged role and session key are gone
    assert sent[1][0]["params"]["_meta"] == stamp and "params" not in sent[1][1]  # each request of a batch
    assert sent[2]["params"]["_meta"] == stamp and up.requests[2].url.path == "/mcp"  # dot segments collapse upstream
    assert all(r.headers["content-length"] == str(len(r.content)) for r in up.requests)  # re-serialised, re-measured
    assert all(r.headers.get_list("agentenv-role") == ["alice@example.com"] for r in up.requests)


@pytest.mark.asyncio
async def test_bodies_that_are_not_a_child_mcp_request_are_forwarded_untouched(monkeypatch):
    up = _Upstream([b"{}"])
    up.install(monkeypatch)
    gw = Gateway(host="127.0.0.1", port=0, server_name="t", internal_mcp_servers=[],
                 rest_proxy_urls={"mcp-svc": UPSTREAM, "backend": UPSTREAM})
    rpc = json.dumps(_rpc(name="t", arguments={}, _meta={AGENT_ENV_ROLE_META_KEY: "x"})).encode()
    cases = [
        ("/svc/mcp-svc/mcp", "POST", b"not json"),  # unparsable: as it came
        ("/svc/mcp-svc/mcp", "GET", b""),  # not a POST
        ("/svc/mcp-svc/mcpx", "POST", rpc),  # not the MCP endpoint
        ("/svc/mcp-svc/agentenv", "POST", rpc),  # the data plane keeps its body
        ("/svc/backend/mcp", "POST", rpc),  # not an mcp-* service
        ("/svc/mcp-svc/mcp", "POST", b'{"jsonrpc": "2.0", "id": 1, "result": {}}'),  # no params: nothing to stamp
    ]
    for path, method, body in cases:
        status, _, _ = await _call(gw, path, method=method, body=body, headers={"AgentEnv-Role": "alice@example.com"})
        assert status == 200, path
    assert [r.content for r in up.requests] == [body for _, _, body in cases]


def _forwarded(request: httpx.Request) -> tuple[list[str], list[str], list[str]]:
    return tuple(request.headers.get_list(name) for name in ("x-forwarded-host", "x-forwarded-proto", "x-forwarded-prefix"))


@pytest.mark.asyncio
async def test_the_upstream_learns_the_host_scheme_and_route_prefix_the_client_reached_it_at(monkeypatch):
    up = _Upstream([b"ok"])
    up.install(monkeypatch)
    gw = _gateway(website_urls={"web": "http://web"})
    await _call(gw, "/website/web/api/x/", headers={"host": "gw.example", "x-forwarded-prefix": "/elsewhere"})
    await _call(gw, "/website/web", headers={"host": "gw.example"})
    await _call(gw, "/website/web/", headers={"host": "gw.example"})
    await _call(gw, "/svc/mcp-svc/x%2Fy", headers={"host": "gw.example"})
    await _call(gw, "/gw%20x/svc/mcp-svc/x", headers={"host": "gw.example"}, root_path="/gw x")
    assert [_forwarded(r) for r in up.forwarded] == [
        (["gw.example"], ["http"], ["/website/web"]),
        (["gw.example"], ["http"], ["/website/web"]),
        (["gw.example"], ["http"], ["/website/web"]),
        (["gw.example"], ["http"], ["/svc/mcp-svc"]),
        (["gw.example"], ["http"], ["/gw%20x/svc/mcp-svc"]),
    ]


@pytest.mark.asyncio
async def test_a_client_serving_the_route_at_its_own_root_is_told_no_prefix_and_its_header_stays_with_the_gateway(
        monkeypatch):
    up = _Upstream([b"ok"])
    up.install(monkeypatch)
    gw = _gateway(website_urls={"web": "http://web"})
    await _call(gw, "/website/web/api/x", headers={"host": "gw.example", "x-forwarded-host": "notes.test",
                                                   "x-forwarded-proto": "https", "agentenv-site-root": "1"})
    await _call(gw, "/website/web/api/x", headers={"host": "gw.example", "agentenv-site-root": ""})
    assert [_forwarded(r) for r in up.forwarded] == [(["notes.test"], ["https"], []),
                                                    (["gw.example"], ["http"], ["/website/web"])]
    assert all("agentenv-site-root" not in r.headers for r in up.forwarded)


@pytest.mark.asyncio
@pytest.mark.parametrize("root, expected", [
    ("https://notes.test", (["notes.test"], ["https"], [])),
    ("http://notes.test:8080/", (["notes.test:8080"], ["http"], [])),
    ("javascript://notes.test", (["front.example"], ["https"], [])),
    ("https://notes.test/elsewhere", (["front.example"], ["https"], [])),
    ("https://user@notes.test", (["front.example"], ["https"], [])),
    ("https://]notes.test[", (["front.example"], ["https"], [])),
    ("https://nötes.test", (["front.example"], ["https"], [])),
], ids=["origin", "port", "not-http", "a-path", "userinfo", "unbalanced-bracket", "not-ascii"])
async def test_a_site_root_naming_an_origin_is_the_address_though_a_front_rewrote_the_forwarded_host(
        monkeypatch, root, expected):
    """A front between a computer's site proxy and the gateway, as a provider's ingress is, may set its own
    X-Forwarded-Host; the origin the proxy names in AgentEnv-Site-Root is the site's address all the same."""
    up = _Upstream([b"ok"])
    up.install(monkeypatch)
    gw = _gateway(website_urls={"web": "http://web"})
    await _call(gw, "/website/web/api/x", raw_headers=[
        (b"host", b"gw.example"), (b"x-forwarded-host", b"front.example"), (b"x-forwarded-proto", b"https"),
        (b"agentenv-site-root", root.encode())])
    assert [_forwarded(r) for r in up.forwarded] == [expected]


@pytest.mark.asyncio
async def test_a_host_that_isnt_printable_ascii_names_nothing_and_the_request_still_goes_through(monkeypatch):
    up = _Upstream([b"ok"])
    up.install(monkeypatch)
    gw = _gateway(website_urls={"web": "http://web"}, site_names={"web": ["shop.test"]})
    answered = [(await _call(gw, "/website/web/x", raw_headers=raw))[0] for raw in (
        [(b"host", "shöp.test".encode())],
        [(b"host", b"shop.test"), (b"x-forwarded-host", "shöp.test".encode())],
        [(b"host", b"gw.example"), (b"x-forwarded-host", b"shop.test\x7f")])]
    assert answered == [200, 200, 200]
    assert [(r.headers["host"], _forwarded(r)[0]) for r in up.forwarded] == [
        ("web", []), ("shop.test", ["shop.test"]), ("web", ["gw.example"])]


@pytest.mark.asyncio
async def test_a_front_s_own_forwarded_host_and_scheme_win_and_reach_the_upstream_once(monkeypatch):
    up = _Upstream([b"ok"])
    up.install(monkeypatch)
    gw = _gateway(website_urls={"web": "http://web"})
    await _call(gw, "/website/web/x", raw_headers=[
        (b"host", b"gateway.internal"), (b"x-forwarded-host", b"env.example, proxy"), (b"x-forwarded-proto", b"https"),
        (b"x-forwarded-proto", b"http"), (b"x-forwarded-for", b"10.0.0.1")])
    await _call(gw, "/website/web/x", headers={"host": "gw.example", "x-forwarded-proto": "gopher"})
    assert [_forwarded(r) for r in up.forwarded] == [
        (["env.example"], ["https"], ["/website/web"]),
        (["gw.example"], ["http"], ["/website/web"]),
    ]
    assert up.forwarded[0].headers.get_list("x-forwarded-for") == ["10.0.0.1"]


@pytest.mark.asyncio
@pytest.mark.parametrize("method, headers, host", [
    ("GET", {"agentenv-site-root": "https://shop.test"}, "shop.test"),
    ("HEAD", {"agentenv-site-root": "https://shop.test"}, "shop.test"),
    ("POST", {"agentenv-site-root": "https://Shop.TEST:8443"}, "Shop.TEST:8443"),
    ("GET", {"x-forwarded-host": "www.shop.test, gw.example", "x-forwarded-proto": "https"}, "www.shop.test"),
    ("GET", {"host": "shop.test"}, "shop.test"),
    ("GET", {"agentenv-site-root": "https://api.shop.test"}, "api.shop.test"),
    ("GET", {"agentenv-site-root": "https://a.b.shop.test"}, "web"),
    ("GET", {"agentenv-site-root": "https://evil.example"}, "web"),
    ("GET", {"x-forwarded-host": "shop.test.evil.example"}, "web"),
    ("GET", {"host": "evil.example"}, "web"),
    ("GET", {"x-forwarded-host": "shop.test:65536"}, "web"),
    ("GET", {"x-forwarded-host": "shop.test@evil.example"}, "web"),
    ("GET", {"x-forwarded-host": "shop..test"}, "web"),
    ("GET", {"x-forwarded-host": "[::1]:80"}, "web"),
], ids=["site-root", "head", "case-and-port", "front", "host", "wildcard", "two-labels-under-a-wildcard", "forged-root",
        "suffix", "forged-host", "bad-port", "userinfo", "empty-label", "address"])
async def test_a_site_reached_at_one_of_its_names_takes_it_as_its_host_and_at_any_other_keeps_its_own(
        monkeypatch, method, headers, host):
    up = _Upstream([b"ok"])
    up.install(monkeypatch)
    gw = _gateway(website_urls={"web": "http://web"}, site_names={"web": ["shop.test", "www.shop.test", "*.shop.test"]})
    await _call(gw, "/website/web/api/x", method=method, headers={"host": "gw.example", **headers})
    assert [r.headers["host"] for r in up.forwarded] == [host]


@pytest.mark.asyncio
async def test_a_site_with_no_names_and_a_svc_route_keep_their_own_host_wherever_the_client_was(monkeypatch):
    up = _Upstream([b"ok"])
    up.install(monkeypatch)
    gw = Gateway(host="127.0.0.1", port=0, server_name="t", internal_mcp_servers=[],
                 website_urls={"web": "http://web", "docs": "http://docs"}, rest_proxy_urls={"web": "http://web"},
                 site_names={"web": ["shop.test"]})
    for path in ("/website/docs/x", "/svc/web/x"):
        await _call(gw, path, headers={"host": "shop.test", "agentenv-site-root": "https://shop.test"})
    assert [r.headers["host"] for r in up.forwarded] == ["docs", "web"]


def _card_site(seen: list[str], declared: dict[str, list[str]]):
    """A site at each host of ``declared`` whose card declares those names; ``seen`` takes each other request's Host,
    and a host ``declared`` doesn't name serves no card."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/.well-known/agent-env.json":
            if request.url.host not in declared:
                return httpx.Response(404)
            return httpx.Response(200, json={"name": request.url.host, "capabilities": {"extensions": [
                {"uri": "urn:agentenv:website/v1", "params": {"domains": declared[request.url.host]}}]}})
        seen.append(request.headers["host"])

        async def body():
            yield b"ok"
        return httpx.Response(200, content=body())
    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_a_site_whose_env_stores_no_names_takes_its_host_from_the_names_its_card_declares(monkeypatch):
    seen: list[str] = []
    transport = _card_site(seen, {"notes": ["Notes.Test"], "kept": ["notes.test"]})
    monkeypatch.setattr(rest_proxy_module.httpx, "AsyncClient",
                        lambda **kw: _RealAsyncClient(transport=transport, **kw))
    gw = _gateway(website_urls={"notes": "http://notes", "kept": "http://kept"}, site_names={"kept": ["kept.test"]})
    for path, root in (("notes", "https://notes.test"), ("kept", "https://notes.test"), ("kept", "https://kept.test")):
        await _call(gw, f"/website/{path}/x", headers={"host": "gw.example", "agentenv-site-root": root})
    assert seen == ["notes.test", "kept", "kept.test"]


@pytest.mark.parametrize("upstream, destination", [
    ("https://notes.internal", {"host": "notes.internal", "port": 443, "server_hostname": "notes.internal"}),
    ("http://notes.internal", {"host": "notes.internal", "port": 80, "server_hostname": "unset"}),
    ("https://notes.internal:8443", {"host": "notes.internal", "port": 8443, "server_hostname": "notes.internal"}),
], ids=["https", "http", "https-port"])
def test_a_websocket_to_a_site_reached_by_name_connects_to_its_upstream_and_checks_the_upstreams_certificate(
        monkeypatch, upstream, destination):
    """Straight to the upstream, never through a proxy the environment names, which would tunnel to the site's
    name: a real domain for a site named after one."""
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:3128")
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.example:3128")
    transport = _card_site([], {})
    monkeypatch.setattr(rest_proxy_module.httpx, "AsyncClient",
                        lambda **kw: _RealAsyncClient(transport=transport, **kw))
    asked = []

    async def connect(uri, **kwargs):
        asked.append((uri, {key: kwargs.get(key, "unset") for key in ("host", "port", "server_hostname", "proxy")}))
        raise OSError("refused")
    monkeypatch.setattr(rest_proxy_module, "connect_websocket", connect)
    gw = _gateway(website_urls={"notes": upstream}, site_names={"notes": ["notes.test"]})
    with pytest.raises(Exception):
        with TestClient(gw._asgi_app()).websocket_connect("/website/notes/live",
                                                         headers={"AgentEnv-Site-Root": "https://notes.test"}):
            pass
    scheme = "wss" if upstream.startswith("https") else "ws"
    assert asked == [(f"{scheme}://notes.test/live", {**destination, "proxy": None})]


@pytest.mark.asyncio
async def test_list_website_urls_gives_a_site_listed_by_name_at_its_first_name_and_any_other_behind_the_gateway(
        monkeypatch):
    transport = _card_site([], {"carded": ["carded.test"]})
    monkeypatch.setattr(rest_proxy_module.httpx, "AsyncClient",
                        lambda **kw: _RealAsyncClient(transport=transport, **kw))
    sites = ("shop", "notes", "wild", "bare", "carded", "other")
    gw = _gateway(website_urls={site: f"http://{site}" for site in sites},
                  site_names={"shop": ["*.shop.test", "www.shop.test", "shop.test"], "notes": ["notes.localhost"],
                              "wild": ["*.wild.test"], "other": ["other.test"]},
                  listed_by_name=["shop", "notes", "wild", "bare", "carded"])
    listed = await gw._mcp._tool_manager.get_tool("list_website_urls").fn()
    assert listed == ["https://www.shop.test/", "http://notes.localhost/", "http://wild", "http://bare",
                      "https://carded.test/", "http://other"]


@pytest.mark.asyncio
async def test_an_sdk_site_behind_the_gateway_names_the_gateway_s_address(monkeypatch):
    @website(domains=["notes.example"])
    class Notes(WebsiteEnvironment):
        @api.get("/api/whoami")
        def whoami(self, ctx: ApiContext) -> dict:
            return {"base_url": ctx.base_url}

    transport = httpx.ASGITransport(app=Notes().create_app())
    monkeypatch.setattr(rest_proxy_module.httpx, "AsyncClient",
                        lambda **kw: _RealAsyncClient(transport=transport, **kw))
    gw = _gateway(website_urls={"notes": "http://notes"})

    async def whoami(path: str) -> tuple[int, dict, bytes]:
        status, answered, chunks = await _call(gw, path, headers={"host": "gw.example"})
        return status, answered, b"".join(chunks)

    _, _, answer = await whoami("/website/notes/api/whoami")
    status, redirect, _ = await whoami("/website/notes/api/whoami/")
    assert json.loads(answer) == {"base_url": "http://gw.example/website/notes"}
    assert (status, redirect["location"]) == (307, "http://gw.example/website/notes/api/whoami")


def test_the_route_an_sdk_card_names_reaches_its_nested_and_typed_paths():
    @website(domains=["files.example"])
    class Files(WebsiteEnvironment):
        @api.get("/files/{rest:path}")
        def files(self, rest: str) -> dict:
            return {"rest": rest}

        @api.get("/api/notes/{note_id:int}")
        def get_note(self, note_id: int) -> dict:
            return {"id": note_id}

    card = TestClient(Files().create_app()).get("/.well-known/agent-env.json").json()
    routes = rest_proxy_module.website_routes(card)
    assert rest_proxy_module.route_name(routes, "GET", "/files/a/b.txt") == "files"
    assert rest_proxy_module.route_name(routes, "GET", "/api/notes/7") == "get_note"
    assert rest_proxy_module.route_name(routes, "GET", "/api/notes/seven") is None


def test_a_request_a_built_in_route_answers_is_named_by_no_card_route():
    @website(domains=["any.example"])
    class AnyApi(WebsiteEnvironment):
        @api.get("/api/{rest:path}")
        def any_api(self, rest: str) -> dict:
            return {"rest": rest}

        @api.delete("/{rest:path}")
        def any_delete(self, rest: str) -> dict:
            return {"rest": rest}

    card = TestClient(AnyApi().create_app()).get("/.well-known/agent-env.json").json()
    routes = rest_proxy_module.website_routes(card)
    assert rest_proxy_module.route_name(routes, "GET", "/api/session") is None
    assert rest_proxy_module.route_name(routes, "DELETE", "/api/session") is None
    assert rest_proxy_module.route_name(routes, "GET", "/api/health") == "any_api"
    assert rest_proxy_module.route_name(routes, "GET", "/api/notes") == "any_api"
    assert rest_proxy_module.route_name(routes, "DELETE", "/api/health") == "any_delete"


def test_a_site_route_at_the_session_path_replaces_the_built_in_one():
    @website(domains=["own.example"])
    class OwnSession(WebsiteEnvironment):
        @api.post("/api/session")
        def sign_in(self) -> dict:
            return {}

        @api.get("/api/{rest:path}")
        def any_api(self, rest: str) -> dict:
            return {"rest": rest}

    card = TestClient(OwnSession().create_app()).get("/.well-known/agent-env.json").json()
    routes = rest_proxy_module.website_routes(card)
    assert rest_proxy_module.route_name(routes, "POST", "/api/session") == "sign_in"
    assert rest_proxy_module.route_name(routes, "GET", "/api/session") == "any_api"


@pytest.mark.parametrize("path, method, target, expected", [
    ("/api/notes/{note_id:int}", "GET", "/api/notes/7", "route"),
    ("/api/notes/{note_id:int}", "GET", "/api/notes/seven", None),
    ("/api/notes/{note_id}", "GET", "/api/notes/seven", "route"),
    ("/api/notes/{note_id}", "GET", "/api/notes/7/extra", None),
    ("/files/{rest:path}", "GET", "/files/a/b.txt", "route"),
    ("/api/price/{p:float}", "GET", "/api/price/3.50", "route"),
    ("/api/x/{id:uuid}", "GET", "/api/x/123e4567-e89b-12d3-a456-426614174000", "route"),
    ("/api/a.b", "GET", "/api/aXb", None),
    ("/api/notes", "HEAD", "/api/notes", "route"),
    ("/api/notes", "POST", "/api/notes", None),
])
def test_a_request_is_named_by_the_card_route_its_method_and_path_reach(path, method, target, expected):
    card = {"capabilities": {"extensions": [{"uri": "urn:agentenv:website/v1", "params": {
        "routes": [{"method": "GET", "path": path, "name": "route"}]}}]}}
    assert rest_proxy_module.route_name(rest_proxy_module.website_routes(card), method, target) == expected
