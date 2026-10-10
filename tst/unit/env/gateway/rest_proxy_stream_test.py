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

from agent_env.env.gateway import AGENT_ENV_ROLE_META_KEY, AGENT_ENV_SESSION_META_KEY
from agent_env.env.gateway import gateway as gateway_module
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
        monkeypatch.setattr(gateway_module.httpx, "AsyncClient",
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
    assert str(up.requests[0].url) == "http://web/assets/app.js"
    status, _, _ = await _call(gw, "/website/web")
    assert status == 200 and str(up.requests[1].url) == "http://web"


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
    real = up.handler

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
    assert [r.url.raw_path for r in up.requests] == [b"/a%2Fb/c%20d", b"/", b"/"]


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
    assert up.requests == []  # nothing was forwarded


def test_gateway_module_doctests_pass():
    """CI does not collect doctests, and `_raw_remainder`'s example is its explanation, so keep it true."""
    result = doctest.testmod(gateway_module)
    assert result.attempted >= 1 and result.failed == 0


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
    assert [r.headers.get_list("agentenv-role") for r in up.requests] == [["viewer"], ["default"]]


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
