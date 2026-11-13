"""The site proxy a computer reaches websites by their names through: what it forwards to a site's route on the
gateway and how, what it answers with, its certificates by name, its DNS for wildcard names, and what its install
writes. The proxy is a network program, so these tests open sockets, on loopback alone."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import dataclasses
import email.message
import io
import json
import os
import socket
import ssl
import struct
import subprocess
import sys
import time
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest
import uvicorn
from agentenv_protocol.website import ApiContext, WebsiteEnvironment, api, website
from websockets.asyncio.client import connect
from websockets.exceptions import InvalidStatus

from agent_env.env.envs.computer.utils import site_proxy, sites
from agent_env.env.envs.computer.utils.params import Site as SiteParams
from agent_env.env.envs.computer.utils.site_proxy import Config, Site, SiteProxy
from agent_env.env.gateway.constants import AGENT_ENV_SITE_ROOT_HEADER
from agent_env.env.gateway.gateway import Gateway
from tst.unit.env.envs.test_computer_stand_in_proxy import _ask, _logged
from tst.unit.env.envs.test_computer_stand_in_proxy import _serving as _stand_in_serving
from tst.unit.env.gateway.website_websocket_test import EchoSite

pytestmark = pytest.mark.usefixtures("socket_enabled")


def _ok(body=b"hello", extra=()):
    async def respond(start, headers, request_body, reader, writer):
        lines = ["HTTP/1.1 200 OK", f"Content-Length: {len(body)}", *(f"{k}: {v}" for k, v in extra)]
        writer.write(("\r\n".join(lines) + "\r\n\r\n").encode() + body)
    return respond


class Upstream:
    """The gateway as the proxy sees it: each request recorded with its decoded body, answered by ``respond``."""

    def __init__(self, respond=None):
        self.requests, self.respond, self.server = [], respond or _ok(), None

    async def __aenter__(self):
        self.server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.port = self.server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *exc):
        self.server.close()

    async def _handle(self, reader, writer):
        head = await reader.readuntil(b"\r\n\r\n")
        start, headers = site_proxy.parse_head(head)
        body = await _body(reader, headers)
        self.requests.append((start, dict((k.lower(), v) for k, v in headers), body))
        await self.respond(start, headers, body, reader, writer)
        await writer.drain()
        writer.close()


async def _body(reader, headers):
    if "chunked" in site_proxy.tokens(headers, "transfer-encoding"):
        data = b""
        while (size := int((await reader.readuntil(b"\r\n")).split(b";")[0], 16)):
            data += (await reader.readexactly(size + 2))[:-2]
        await reader.readuntil(b"\r\n")
        return data
    length = site_proxy.header(headers, "content-length")
    return await reader.readexactly(int(length)) if length else b""


@contextlib.asynccontextmanager
async def _proxy(config, scheme="https", network=None):
    proxy = SiteProxy(config, network=network)
    context = site_proxy.tls_context(config) if scheme == "https" else None
    server = await asyncio.start_server(lambda r, w: proxy.handle(r, w, scheme), "127.0.0.1", 0, ssl=context)
    try:
        yield server.sockets[0].getsockname()[1]
    finally:
        server.close()


@contextlib.asynccontextmanager
async def _forward_port(config, network=None):
    proxy = SiteProxy(config, tls=site_proxy.tls_context(config), network=network)
    server = await asyncio.start_server(proxy.handle_forward, "127.0.0.1", 0)
    try:
        yield server.sockets[0].getsockname()[1]
    finally:
        server.close()


async def _exchange(port, request: bytes, *, server_hostname=None, cadata=None, reader_writer=None):
    if reader_writer is None:
        context = None
        if cadata is not None:
            context = ssl.create_default_context(cadata=cadata)
        reader, writer = await asyncio.open_connection("127.0.0.1", port, ssl=context,
                                                       server_hostname=server_hostname if context else None)
    else:
        reader, writer = reader_writer
    writer.write(request)
    await writer.drain()
    head = await reader.readuntil(b"\r\n\r\n")
    start, headers = site_proxy.parse_head(head)
    body = await _body(reader, headers)
    return start, dict((k.lower(), v) for k, v in headers), body, (reader, writer)


def _issued(tmp_path, routes):
    issued = sites.issue(routes, label="test")
    files = {}
    for name, (cert, key) in issued.leaves.items():
        (tmp_path / f"{name}.crt").write_text(cert)
        (tmp_path / f"{name}.key").write_text(key)
        files[name] = (str(tmp_path / f"{name}.crt"), str(tmp_path / f"{name}.key"))
    return issued, files


def _config(tmp_path, upstream_port, **domains_by_site):
    domains_by_site = domains_by_site or {"notes": ["notes.test"]}
    routes = [sites.Route(name, f"http://127.0.0.1:{upstream_port}/website/{name}/", tuple(domains))
              for name, domains in domains_by_site.items()]
    issued, files = _issued(tmp_path, routes)
    config = Config(user=None, sites=[Site(r.site, r.url, list(r.domains), *files.get(r.site, (None, None)))
                                      for r in routes])
    return config, issued


@pytest.mark.asyncio
async def test_a_request_goes_to_the_sites_route_with_the_name_and_scheme_the_browser_used(tmp_path):
    async with Upstream() as upstream:
        config, issued = _config(tmp_path, upstream.port)
        async with _proxy(config) as port:
            start, headers, body, conn = await _exchange(
                port, b"GET /notes/1?full=yes HTTP/1.1\r\nHost: notes.test\r\nX-Forwarded-Host: evil.example\r\n"
                      b"X-Forwarded-Prefix: /elsewhere\r\nAgentEnv-Site-Root: 0\r\n"
                      b"Cookie: s=1\r\nConnection: keep-alive\r\n\r\n", server_hostname="notes.test",
                cadata=issued.ca_pem)
            again = await _exchange(port, b"GET / HTTP/1.1\r\nHost: notes.test\r\n\r\n", reader_writer=conn)
            conn[1].close()

    (first_start, first, _), (second_start, _, _) = upstream.requests
    assert first_start == "GET /website/notes/notes/1?full=yes HTTP/1.1" and second_start.startswith(
        "GET /website/notes/ ")
    assert first["host"] == f"127.0.0.1:{upstream.port}" and first["x-forwarded-host"] == "notes.test"
    assert (first["x-forwarded-proto"], first["cookie"], first["connection"]) == ("https", "s=1", "close")
    assert "x-forwarded-prefix" not in first and first[AGENT_ENV_SITE_ROOT_HEADER.lower()] == "https://notes.test"
    assert site_proxy.SITE_ROOT_HEADER == AGENT_ENV_SITE_ROOT_HEADER
    assert (start, body, headers["connection"]) == ("HTTP/1.1 200 OK", b"hello", "keep-alive")
    assert again[2] == b"hello"


@pytest.mark.asyncio
async def test_the_providers_headers_go_upstream_in_place_of_the_browsers_and_never_back(tmp_path):
    async with Upstream() as upstream:
        config, issued = _config(tmp_path, upstream.port)
        config.sites[0].upstream_headers = {"CF-Access-Client-Secret": "s3cret"}
        async with _proxy(config) as port:
            start, headers, body, conn = await _exchange(
                port, b"GET / HTTP/1.1\r\nHost: notes.test\r\ncf-access-client-secret: forged\r\n\r\n",
                server_hostname="notes.test", cadata=issued.ca_pem)
            conn[1].close()

    ((_, sent, _),) = upstream.requests
    assert sent["cf-access-client-secret"] == "s3cret"
    assert "s3cret" not in json.dumps(headers) and body == b"hello"


async def _serve(app) -> tuple[uvicorn.Server, int]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    config = uvicorn.Config(app, lifespan="off", log_level="critical", access_log=False)
    config.load()
    server = uvicorn.Server(config)
    server.lifespan = config.lifespan_class(config)
    await server.startup(sockets=[sock])
    return server, sock.getsockname()[1]


@pytest.mark.asyncio
async def test_an_sdk_site_behind_the_real_gateway_names_its_own_root_and_keeps_its_card_hidden(tmp_path):
    @website(domains=["notes.test"])
    class Notes(WebsiteEnvironment):
        @api.get("/api/whoami")
        def whoami(self, ctx: ApiContext) -> dict:
            return {"base_url": ctx.base_url}

    site, site_port = await _serve(Notes().create_app())
    gateway = Gateway(host="127.0.0.1", port=0, server_name="t", internal_mcp_servers=[],
                      rest_proxy_urls={"notes": f"http://127.0.0.1:{site_port}"},
                      website_urls={"notes": f"http://127.0.0.1:{site_port}"})
    gw, gw_port = await _serve(gateway._mcp.streamable_http_app())
    answers = []
    try:
        config, issued = _config(tmp_path, gw_port)
        async with _proxy(config) as port:
            for path in ("/api/whoami", "/api/whoami/", "/.well-known/agent-env.json"):
                start, headers, body, conn = await _exchange(
                    port, f"GET {path} HTTP/1.1\r\nHost: notes.test\r\nConnection: close\r\n\r\n".encode(),
                    server_hostname="notes.test", cadata=issued.ca_pem)
                conn[1].close()
                answers.append((start, headers.get("location"), body))
    finally:
        await gw.shutdown()
        await site.shutdown()

    (whoami, redirect, card) = answers
    assert json.loads(whoami[2]) == {"base_url": "https://notes.test"}
    assert redirect[:2] == ("HTTP/1.1 307 Temporary Redirect", "https://notes.test/api/whoami")
    assert card[0].startswith("HTTP/1.1 404")


@pytest.mark.asyncio
async def test_a_redirect_or_cookie_naming_the_route_is_rewritten_onto_the_sites_name(tmp_path):
    async def respond(start, headers, body, reader, writer):
        location = f"http://127.0.0.1:{upstream.port}/website/notes/signed-in?next=%2F#top"
        writer.write(f"HTTP/1.1 302 Found\r\nLocation: {location}\r\nSet-Cookie: s=1; Domain=127.0.0.1; "
                     f"Path=/website/notes/app; HttpOnly\r\nAlt-Svc: h3=\":443\"\r\nContent-Length: 0\r\n\r\n".encode())

    async with Upstream(respond) as upstream:
        config, issued = _config(tmp_path, upstream.port)
        async with _proxy(config) as port:
            _, headers, _, conn = await _exchange(port, b"POST /login HTTP/1.1\r\nHost: notes.test\r\n"
                                                        b"Content-Length: 4\r\n\r\nuser", server_hostname="notes.test",
                                                  cadata=issued.ca_pem)
            conn[1].close()

    assert headers["location"] == "https://notes.test/signed-in?next=%2F#top"
    assert headers["set-cookie"] == "s=1; Path=/app; HttpOnly" and "alt-svc" not in headers
    assert upstream.requests[0][2] == b"user"


@pytest.mark.parametrize("location, expected", [
    ("/website/notes/a", "/a"),
    ("/website/notes", "/"),
    ("http://gw:8080/website/notes/a?b=1", "https://notes.test/a?b=1"),
    ("http://gw:8080/website/other/a", "http://gw:8080/website/other/a"),
    ("http://gw:8080/svc/notes/x", "http://gw:8080/svc/notes/x"),
    ("https://gw:8080/website/notes/a", "https://gw:8080/website/notes/a"),
    ("https://notes.test/a", "https://notes.test/a"),
    ("/a", "/a"),
    ("a/b", "a/b"),
    ("//gw:8080/website/notes/a", "https://notes.test/a"),
    ("//other.example/website/notes/a", "//other.example/website/notes/a"),
    ("https://notes.test/website/notes/app/", "https://notes.test/app/"),
    ("https://notes.test:443/website/notes", "https://notes.test/"),
    ("http://notes.test/website/notes/a", "http://notes.test/website/notes/a"),
], ids=["path", "prefix-alone", "gateway-url", "another-site", "outside-routes", "other-scheme", "site-url",
        "site-path", "relative", "scheme-relative", "scheme-relative-elsewhere", "site-url-under-prefix", "site-url-default-port",
        "site-name-other-scheme"])
def test_only_a_location_on_the_sites_route_is_moved_onto_its_name(location, expected):
    site = Site("notes", "http://gw:8080/website/notes/", ["notes.test"])

    assert site_proxy.public_location(location, site, "https", "notes.test") == expected


@pytest.mark.parametrize(("cookie", "expected"), [
    ("s=1; Path = /website/notes/x", "s=1; Path=/x"),
    ("s=1; path=/website/notes", "s=1; Path=/"),
    ("s=1;PATH= /website/notes/a/b ; HttpOnly", "s=1; Path=/a/b; HttpOnly"),
    ("s=1; Path=/website/other", "s=1; Path=/website/other"),
], ids=["spaced", "lower-case", "upper-case-spaced", "another-route"])
def test_a_cookie_path_on_the_route_is_rooted_however_it_is_spelled(cookie, expected):
    site = Site("notes", "http://gw:8080/website/notes/", ["notes.test"])

    assert site_proxy.public_cookie(cookie, site) == expected


@pytest.mark.asyncio
async def test_a_chunked_upload_and_a_chunked_answer_are_relayed_as_framed(tmp_path):
    async def respond(start, headers, body, reader, writer):
        writer.write(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nevent\r\n3\r\n: 1\r\n0\r\n\r\n")

    async with Upstream(respond) as upstream:
        config, issued = _config(tmp_path, upstream.port)
        async with _proxy(config) as port:
            _, headers, body, conn = await _exchange(
                port, b"POST /api HTTP/1.1\r\nHost: notes.test\r\nTransfer-Encoding: chunked\r\n\r\n"
                      b"4\r\nabcd\r\n2\r\nef\r\n0\r\n\r\n", server_hostname="notes.test", cadata=issued.ca_pem)
            conn[1].close()

    assert upstream.requests[0][2] == b"abcdef" and upstream.requests[0][1]["transfer-encoding"] == "chunked"
    assert (headers["transfer-encoding"], body) == ("chunked", b"event: 1")


@pytest.mark.asyncio
async def test_an_answer_with_no_length_is_relayed_to_its_end_and_closes_the_connection(tmp_path):
    async def respond(start, headers, body, reader, writer):
        writer.write(b"HTTP/1.1 200 OK\r\n\r\nall of it")

    async with Upstream(respond) as upstream:
        config, issued = _config(tmp_path, upstream.port)
        async with _proxy(config) as port:
            reader, writer = await asyncio.open_connection("127.0.0.1", port, ssl=ssl.create_default_context(
                cadata=issued.ca_pem), server_hostname="notes.test")
            writer.write(b"GET / HTTP/1.1\r\nHost: notes.test\r\n\r\n")
            answer = await reader.read()
            writer.close()

    assert answer.startswith(b"HTTP/1.1 200 OK\r\n") and b"Connection: close" in answer
    assert answer.endswith(b"\r\n\r\nall of it")


@pytest.mark.asyncio
async def test_the_forward_port_tunnels_a_connect_to_a_sites_name_through_its_certificate_to_its_route(tmp_path):
    async with Upstream() as upstream:
        config, issued = _config(tmp_path, upstream.port)
        async with _forward_port(config) as port:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(b"CONNECT notes.test:443 HTTP/1.1\r\nHost: notes.test:443\r\n\r\n")
            established = await reader.readuntil(b"\r\n\r\n")
            await writer.start_tls(ssl.create_default_context(cadata=issued.ca_pem), server_hostname="notes.test")
            first = await _exchange(port, b"GET /a?b=1 HTTP/1.1\r\nHost: notes.test\r\n\r\n",
                                    reader_writer=(reader, writer))
            second = await _exchange(port, b"GET /c HTTP/1.1\r\nHost: notes.test\r\n\r\n",
                                     reader_writer=(reader, writer))
            writer.close()

    assert established.startswith(b"HTTP/1.1 200 ")
    assert (first[0], first[2], second[0]) == ("HTTP/1.1 200 OK", b"hello", "HTTP/1.1 200 OK")
    assert [request[0] for request in upstream.requests] == ["GET /website/notes/a?b=1 HTTP/1.1",
                                                             "GET /website/notes/c HTTP/1.1"]
    assert (upstream.requests[0][1]["x-forwarded-host"], upstream.requests[0][1]["x-forwarded-proto"]) == (
        "notes.test", "https")


@pytest.mark.asyncio
async def test_a_request_in_a_sites_tunnel_naming_another_host_is_refused_and_logged_as_the_port_refuses(tmp_path,
                                                                                                        capsys):
    async with Upstream() as upstream:
        config, issued = _config(tmp_path, upstream.port)
        async with _forward_port(config) as port:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(b"CONNECT notes.test:443 HTTP/1.1\r\nHost: notes.test:443\r\n\r\n")
            await reader.readuntil(b"\r\n\r\n")
            await writer.start_tls(ssl.create_default_context(cadata=issued.ca_pem), server_hostname="notes.test")
            start, headers, body, conn = await _exchange(port, b"GET / HTTP/1.1\r\nHost: evil.test\r\n\r\n",
                                                         reader_writer=(reader, writer))
            writer.close()

    assert start == "HTTP/1.1 403 Forbidden" and headers["x-agentenv-site-proxy"] == "failed"
    assert upstream.requests == []
    assert _logged(capsys.readouterr().err, "refused") == [
        {"method": "GET", "host": "evil.test", "reason": "no site of this computer"}]


@pytest.mark.asyncio
async def test_the_forward_port_serves_a_localhost_name_named_by_its_url_and_sends_a_real_name_to_https(tmp_path):
    async with Upstream() as upstream:
        config, _ = _config(tmp_path, upstream.port, notes=["notes.localhost"], shop=["shop.test"])
        async with _forward_port(config) as port:
            local = await _exchange(port, b"GET http://notes.localhost/a?b=1 HTTP/1.1\r\nHost: notes.localhost\r\n\r\n")
            local[3][1].close()
            real = await _exchange(port, b"GET http://shop.test/cart?q=book&n=2 HTTP/1.1\r\nHost: shop.test\r\n\r\n")
            real[3][1].close()

    assert (local[0], local[2]) == ("HTTP/1.1 200 OK", b"hello")
    assert upstream.requests[0][0] == "GET /website/notes/a?b=1 HTTP/1.1"
    assert upstream.requests[0][1]["x-forwarded-proto"] == "http" and len(upstream.requests) == 1
    assert (real[0], real[1]["location"]) == ("HTTP/1.1 308 Permanent Redirect", "https://shop.test/cart?q=book&n=2")


@pytest.mark.asyncio
@pytest.mark.parametrize("request_line, named", [
    (b"CONNECT example.com:443 HTTP/1.1", "example.com:443"),
    (b"CONNECT 127.0.0.1:18765 HTTP/1.1", "127.0.0.1:18765"),
    (b"GET http://127.0.0.1:18765/ HTTP/1.1", "127.0.0.1:18765"),
    (b"GET http://example.com/ HTTP/1.1", "example.com"),
], ids=["connect", "connect-loopback", "url-loopback", "url"])
async def test_the_forward_port_refuses_any_other_host_at_once_and_logs_it(tmp_path, capsys, request_line, named):
    async with Upstream() as upstream:
        config, _ = _config(tmp_path, upstream.port)
        async with _forward_port(config) as port:
            start, headers, body, conn = await asyncio.wait_for(_exchange(
                port, request_line + b"\r\nHost: " + named.encode() + b"\r\n\r\n"), 5)
            conn[1].close()

    assert start == "HTTP/1.1 403 Forbidden" and headers["x-agentenv-site-proxy"] == "failed"
    assert body == f"{named} is no site of this computer, and the browser reaches its sites alone\n".encode()
    assert upstream.requests == []
    method = request_line.split(b" ")[0].decode()
    assert _logged(capsys.readouterr().err, "refused") == [
        {"method": method, "host": named, "reason": "no site of this computer"}]


@contextlib.contextmanager
def _live_hosts_at(monkeypatch, upstream_port):
    """Connections the proxy opens to a live site's host landing on ``upstream_port`` on loopback, and each host it
    opened one to recorded."""
    opened, real = [], asyncio.open_connection

    async def open_connection(host, port, **kwargs):
        if host == "127.0.0.1":
            return await real(host, port, **kwargs)
        opened.append((host, port))
        return await real("127.0.0.1", upstream_port, **kwargs)

    monkeypatch.setattr(site_proxy.asyncio, "open_connection", open_connection)
    yield opened


@pytest.mark.asyncio
async def test_the_forward_port_tunnels_a_live_sites_host_straight_to_it_and_sends_its_url_on_as_a_path(
        tmp_path, monkeypatch, capsys):
    async with Upstream() as upstream, Upstream() as live:
        config, _ = _config(tmp_path, upstream.port)
        config.direct = ["github.com", "*.githubassets.com"]
        with _live_hosts_at(monkeypatch, live.port) as opened:
            async with _forward_port(config) as port:
                reader, writer = await asyncio.open_connection("127.0.0.1", port)
                writer.write(b"CONNECT github.githubassets.com:8443 HTTP/1.1\r\nHost: github.githubassets.com\r\n\r\n")
                established = await reader.readuntil(b"\r\n\r\n")
                tunnelled = await _exchange(port, b"GET /x HTTP/1.1\r\nHost: github.githubassets.com\r\n\r\n",
                                            reader_writer=(reader, writer))
                writer.close()
                plain = await _exchange(port, b"GET http://github.com/a?b=1 HTTP/1.1\r\nHost: github.com\r\n"
                                              b"Proxy-Connection: keep-alive\r\n\r\n")
                plain[3][1].close()
                await asyncio.sleep(0.3)  # each connection is logged as it ends

    assert established.startswith(b"HTTP/1.1 200 ") and opened == [("github.githubassets.com", 8443), ("github.com", 80)]
    assert (tunnelled[0], tunnelled[2], plain[0], plain[2]) == ("HTTP/1.1 200 OK", b"hello", "HTTP/1.1 200 OK", b"hello")
    assert [request[0] for request in live.requests] == ["GET /x HTTP/1.1", "GET /a?b=1 HTTP/1.1"]
    assert live.requests[1][1]["connection"] == "close" and "proxy-connection" not in live.requests[1][1]
    assert live.requests[1][1]["host"] == "github.com" and upstream.requests == []
    logged = _logged(capsys.readouterr().err, "connection")
    assert [(entry["host"], entry["port"], entry["address"]) for entry in logged] == [
        ("github.githubassets.com", 8443, "127.0.0.1"), ("github.com", 80, "127.0.0.1")]
    assert all(entry["bytes_sent"] > 0 and entry["bytes_received"] > 0 and entry["duration_s"] >= 0 for entry in logged)


@pytest.mark.asyncio
async def test_a_live_host_that_doesnt_answer_in_time_is_a_502_and_logged(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(site_proxy, "UPSTREAM_TIMEOUT_S", 0.5)
    real_open = asyncio.open_connection

    async def hangs(host, port, **kwargs):
        if host == "127.0.0.1":
            return await real_open(host, port, **kwargs)
        await asyncio.sleep(60)

    async with Upstream() as upstream:
        config, _ = _config(tmp_path, upstream.port)
        config.direct = ["github.com"]
        monkeypatch.setattr(site_proxy.asyncio, "open_connection", hangs)
        async with _forward_port(config) as port:
            began = time.monotonic()
            answer = await _exchange(port, b"CONNECT github.com:443 HTTP/1.1\r\nHost: github.com:443\r\n\r\n")
            took = time.monotonic() - began

    assert answer[0] == "HTTP/1.1 502 Bad Gateway" and answer[1]["x-agentenv-site-proxy"] == "failed" and took < 1.2
    [failed] = _logged(capsys.readouterr().err, "failed")
    assert failed.pop("started").endswith("Z") and failed == {
        "method": "CONNECT", "host": "github.com", "port": 443, "reason": "no answer", "error": "TimeoutError"}
    assert answer[2] == b"github.com:443 didn't answer\n"


@pytest.mark.asyncio
async def test_both_proxies_log_a_connection_and_each_way_one_fails_in_the_same_json_lines(tmp_path, monkeypatch,
                                                                                          capsys):
    """The stand-in a browser alone uses and the forward port's tunnel to a live site's hosts write the same records:
    each line JSON, a ``connection`` with the same fields, and a ``failed`` with the same fields and reason, whichever
    proxy a run took."""
    requests = [b"GET http://site.test/ HTTP/1.1\r\nHost: site.test\r\n\r\n",
                b"CONNECT nowhere.test:443 HTTP/1.1\r\nHost: nowhere.test:443\r\n\r\n",
                b"CONNECT slow.test:443 HTTP/1.1\r\nHost: slow.test:443\r\n\r\n"]
    with _stand_in_serving(monkeypatch, names={"nowhere.test": [], "slow.test": ["140.82.112.9"]},
                           acts={"140.82.112.9": "hangs"}, CONNECT_TIMEOUT_S=0.3) as stand_in:
        for request in requests:
            _ask(stand_in.port, request)
    stand_in_log = capsys.readouterr().out

    monkeypatch.setattr(site_proxy, "UPSTREAM_TIMEOUT_S", 0.3)
    real_open = asyncio.open_connection

    async def open_connection(host, port, **kwargs):
        if host == "nowhere.test":
            raise socket.gaierror(-2, "Name or service not known")
        if host == "slow.test":
            await asyncio.sleep(60)
        return await real_open("127.0.0.1", port if host == "127.0.0.1" else live.port, **kwargs)

    async with Upstream() as upstream, Upstream() as live:
        config, _ = _config(tmp_path, upstream.port)
        config.direct = ["site.test", "nowhere.test", "slow.test"]
        monkeypatch.setattr(site_proxy.asyncio, "open_connection", open_connection)
        async with _forward_port(config) as port:
            for request in requests:
                answer = await _exchange(port, request)
                answer[3][1].close()
            await asyncio.sleep(0.3)  # each connection is logged as it ends
    shared_log = capsys.readouterr().err

    def shape(log: str) -> list:
        return [(kind, sorted(record), record.get("reason")) for entry in _logged(log) for kind, record in entry.items()]

    assert sorted(shape(stand_in_log)) == sorted(shape(shared_log)) == [
        ("connection", ["address", "bytes_received", "bytes_sent", "duration_s", "host", "port", "started"], None),
        ("failed", ["error", "host", "method", "port", "reason", "started"], "doesn't resolve"),
        ("failed", ["error", "host", "method", "port", "reason", "started"], "no answer")]


@pytest.mark.asyncio
async def test_a_sites_name_within_a_live_hosts_wildcard_is_the_sites_and_no_address_is_ever_taken_straight(
        tmp_path, monkeypatch, capsys):
    async with Upstream() as upstream, Upstream() as live:
        config, issued = _config(tmp_path, upstream.port, gist=["gist.github.com"])
        config.direct = ["*.github.com", "127.0.0.1", "github.localhost"]
        with _live_hosts_at(monkeypatch, live.port) as opened:
            async with _forward_port(config) as port:
                reader, writer = await asyncio.open_connection("127.0.0.1", port)
                writer.write(b"CONNECT gist.github.com:443 HTTP/1.1\r\nHost: gist.github.com:443\r\n\r\n")
                await reader.readuntil(b"\r\n\r\n")
                await writer.start_tls(ssl.create_default_context(cadata=issued.ca_pem),
                                       server_hostname="gist.github.com")
                site = await _exchange(port, b"GET / HTTP/1.1\r\nHost: gist.github.com\r\n\r\n",
                                       reader_writer=(reader, writer))
                writer.close()
                refused = [await _exchange(port, line + b"\r\n\r\n") for line in (
                    b"CONNECT 127.0.0.1:18765 HTTP/1.1\r\nHost: 127.0.0.1:18765",
                    b"GET http://github.localhost/ HTTP/1.1\r\nHost: github.localhost")]

    assert site[0] == "HTTP/1.1 200 OK" and upstream.requests[0][0] == "GET /website/gist/ HTTP/1.1"
    assert [answer[0] for answer in refused] == ["HTTP/1.1 403 Forbidden"] * 2
    assert opened == [] and live.requests == []
    assert [Config(user=None, sites=[], direct=["*.github.com", "10.0.0.1"]).is_direct(host) for host in (
        "api.github.com", "github.com", "a.b.github.com", "10.0.0.1")] == [True, False, False, False]


@pytest.mark.asyncio
async def test_the_forward_port_streams_an_event_stream_as_it_comes(tmp_path):
    release = asyncio.Event()

    async def respond(start, headers, body, reader, writer):
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n"
                     b"d\r\ndata: first\n\n\r\n")
        await writer.drain()
        await release.wait()
        writer.write(b"0\r\n\r\n")

    async with Upstream(respond) as upstream:
        config, _ = _config(tmp_path, upstream.port, notes=["notes.localhost"])
        async with _forward_port(config) as port:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(b"GET http://notes.localhost/events HTTP/1.1\r\nHost: notes.localhost\r\n\r\n")
            head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 5)
            first = await asyncio.wait_for(reader.readuntil(b"\n\n\r\n"), 5)
            release.set()
            rest = await asyncio.wait_for(reader.readuntil(b"0\r\n\r\n"), 5)
            writer.close()

    assert head.startswith(b"HTTP/1.1 200 OK\r\n") and b"text/event-stream" in head
    assert first == b"d\r\ndata: first\n\n\r\n" and rest == b"0\r\n\r\n"


@pytest.mark.asyncio
async def test_a_websocket_upgrade_is_relayed_both_ways(tmp_path):
    async def respond(start, headers, body, reader, writer):
        writer.write(b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                     b"Sec-WebSocket-Accept: x\r\n\r\n")
        await writer.drain()
        data = await reader.readexactly(4)
        writer.write(data[::-1])

    async with Upstream(respond) as upstream:
        config, issued = _config(tmp_path, upstream.port)
        async with _proxy(config) as port:
            reader, writer = await asyncio.open_connection("127.0.0.1", port, ssl=ssl.create_default_context(
                cadata=issued.ca_pem), server_hostname="notes.test")
            writer.write(b"GET /live HTTP/1.1\r\nHost: notes.test\r\nConnection: Upgrade\r\nUpgrade: websocket\r\n"
                         b"Sec-WebSocket-Key: k\r\n\r\n")
            head = await reader.readuntil(b"\r\n\r\n")
            writer.write(b"ping")
            echoed = await asyncio.wait_for(reader.readexactly(4), 5)
            writer.close()

    request = upstream.requests[0][1]
    assert (request["connection"], request["upgrade"], request["sec-websocket-key"]) == ("Upgrade", "websocket", "k")
    assert head.startswith(b"HTTP/1.1 101 ") and b"Connection: Upgrade" in head and echoed == b"gnip"


@pytest.mark.asyncio
async def test_a_browsers_websocket_reaches_the_site_through_the_real_gateway_under_the_sites_name(tmp_path):
    site = EchoSite()
    site_server, site_port = await _serve(site.app)
    gateway = Gateway(host="127.0.0.1", port=0, server_name="t", internal_mcp_servers=[],
                      website_urls={"notes": f"http://127.0.0.1:{site_port}"})
    gw, gw_port = await _serve(gateway._mcp.streamable_http_app())
    try:
        config, issued = _config(tmp_path, gw_port)
        async with _proxy(config) as port:
            context = ssl.create_default_context(cadata=issued.ca_pem)
            async with connect("wss://notes.test/live", host="127.0.0.1", port=port, ssl=context) as websocket:
                await websocket.send("hello")
                echoed = await asyncio.wait_for(websocket.recv(), 5)
            with pytest.raises(InvalidStatus) as refused:
                async with connect("wss://notes.test/agentenv", host="127.0.0.1", port=port, ssl=context):
                    pass
    finally:
        await gw.shutdown()
        await site_server.shutdown()

    ((path, headers),) = site.handshakes
    assert echoed == "HELLO" and path == "/live" and refused.value.response.status_code == 404
    assert (headers["x-forwarded-host"], headers["x-forwarded-proto"]) == ("notes.test", "https")
    assert "x-forwarded-prefix" not in headers


@pytest.mark.asyncio
async def test_http_redirects_a_name_served_over_https_and_serves_a_localhost_name_as_it_is(tmp_path):
    async with Upstream() as upstream:
        config, _ = _config(tmp_path, upstream.port, notes=["notes.test", "notes.localhost"])
        async with _proxy(config, scheme="http") as port:
            redirected = await _exchange(port, b"GET /a?b=1 HTTP/1.1\r\nHost: notes.test\r\n\r\n")
            served = await _exchange(port, b"GET /a HTTP/1.1\r\nHost: notes.localhost:80\r\n\r\n")
            for answer in (redirected, served):
                answer[3][1].close()

    assert redirected[0].startswith("HTTP/1.1 308 ") and redirected[1]["location"] == "https://notes.test/a?b=1"
    ((_, request, _),) = upstream.requests
    assert (request["x-forwarded-host"], request["x-forwarded-proto"]) == ("notes.localhost:80", "http")
    assert served[2] == b"hello"


@pytest.mark.asyncio
async def test_a_name_no_site_has_is_misdirected_and_an_unreachable_site_is_a_bad_gateway(tmp_path):
    with socket.socket() as closed:
        closed.bind(("127.0.0.1", 0))
        dead = closed.getsockname()[1]
    config, _ = _config(tmp_path, dead, notes=["notes.localhost"])
    async with _proxy(config, scheme="http") as port:
        unknown = await _exchange(port, b"GET / HTTP/1.1\r\nHost: other.localhost\r\n\r\n")
        unreachable = await _exchange(port, b"GET / HTTP/1.1\r\nHost: notes.localhost\r\n\r\n")

    assert unknown[0].startswith("HTTP/1.1 421 ") and b"other.localhost is no site" in unknown[2]
    assert unreachable[0].startswith("HTTP/1.1 502 ") and b"the site notes can't be reached" in unreachable[2]
    assert unknown[1]["x-agentenv-site-proxy"] == unreachable[1]["x-agentenv-site-proxy"] == "failed"


def test_a_check_answered_by_the_proxys_own_bad_gateway_fails_and_a_sites_own_error_status_loads(monkeypatch):
    answers = {"notes.test": site_proxy.FAILED, "chat.test": ()}

    class Opener:
        def open(self, url, timeout):
            name = url.split("/")[2]
            headers = email.message.Message()
            for key, value in answers[name]:
                headers[key] = value
            raise urllib.error.HTTPError(url, 502, "Bad Gateway", headers, io.BytesIO(b"the site can't be reached\n"))

    monkeypatch.setattr(site_proxy.urllib.request, "build_opener", lambda *handlers: Opener())
    monkeypatch.setattr(site_proxy, "check_runtimes", lambda url: {"node": {"url": url, "status": 502}})
    config = Config(user=None, sites=[Site("notes", "http://gw/website/notes/", ["notes.test"]),
                                      Site("chat", "http://gw/website/chat/", ["chat.test"])])

    assert site_proxy.check(config) == {"sites": {
        "notes.test": {"url": "https://notes.test/", "error": "502 Bad Gateway: the site can't be reached"},
        "chat.test": {"url": "https://chat.test/", "status": 502}},
        "runtimes": {"node": {"url": "https://notes.test/", "status": 502}}}


def test_a_check_loads_a_site_through_the_forward_port_and_names_one_that_doesnt_answer(monkeypatch):
    opened = []

    class Opener:
        def __init__(self, proxies):
            self.proxies = proxies

        def open(self, url, timeout):
            opened.append((url, self.proxies))
            if self.proxies:
                raise urllib.error.URLError(ConnectionRefusedError(111, "Connection refused"))
            return contextlib.nullcontext(SimpleNamespace(status=200))

    monkeypatch.setattr(site_proxy.urllib.request, "build_opener", lambda handler: Opener(handler.proxies))
    monkeypatch.setattr(site_proxy, "check_runtimes", lambda url: {})
    sites_ = [Site("notes", "http://gw/website/notes/", ["notes.localhost"]),
              Site("chat", "http://gw/website/chat/", ["chat.test"])]

    alone = site_proxy.check(Config(user=None, sites=sites_, proxy_port=3129))
    desktop = site_proxy.check(Config(user=None, sites=sites_))

    proxy = "http://127.0.0.1:3129"
    assert alone["forward_port"] == {"proxy": proxy, "url": "https://chat.test/",
                                     "error": "[Errno 111] Connection refused"}
    assert opened[2] == ("https://chat.test/", {"http": proxy, "https": proxy})
    assert alone["sites"] == desktop["sites"] and "forward_port" not in desktop


@pytest.mark.asyncio
async def test_each_name_is_served_the_certificate_of_its_own_site_and_a_wildcard_covers_one_label(tmp_path):
    async with Upstream() as upstream:
        config, issued = _config(tmp_path, upstream.port, notes=["notes.test"], chat=["chat.test", "*.chat.test"])
        async with _proxy(config) as port:
            seen = {}
            for name in ("notes.test", "chat.test", "eu.chat.test"):
                reader, writer = await asyncio.open_connection("127.0.0.1", port, ssl=ssl.create_default_context(
                    cadata=issued.ca_pem), server_hostname=name)
                seen[name] = writer.get_extra_info("peercert")["subjectAltName"][0][1]
                writer.close()
            with pytest.raises(ssl.SSLCertVerificationError):
                await asyncio.open_connection("127.0.0.1", port, ssl=ssl.create_default_context(cadata=issued.ca_pem),
                                              server_hostname="a.eu.chat.test")

    assert seen == {"notes.test": "notes.test", "chat.test": "chat.test", "eu.chat.test": "chat.test"}


def test_a_site_is_found_by_its_exact_name_before_a_wildcard_and_a_wildcard_covers_one_label():
    config = Config(user=None, sites=[Site("wild", "http://gw/website/wild/", ["*.chat.test"]),
                                      Site("eu", "http://gw/website/eu/", ["eu.chat.test"])])

    assert [getattr(config.site_for(name), "name", None) for name in ("eu.chat.test", "us.chat.test", "chat.test",
                                                                      "a.us.chat.test", "US.chat.test.")] == [
        "eu", "wild", None, None, "wild"]


def _query(name: str, qtype: int) -> bytes:
    labels = b"".join(bytes([len(part)]) + part.encode() for part in name.split("."))
    return struct.pack("!HHHHHH", 0x1234, 0x0100, 1, 0, 0, 0) + labels + b"\0" + struct.pack("!HH", qtype, 1)


@pytest.mark.asyncio
async def test_dns_answers_a_sites_wildcard_name_with_loopback_and_passes_every_other_question_on():
    passed = []

    class Nameserver(asyncio.DatagramProtocol):
        def connection_made(self, transport):
            self.transport = transport

        def datagram_received(self, data, addr):
            passed.append(data)
            self.transport.sendto(data[:2] + b"\x81\x80" + data[4:], addr)

    loop = asyncio.get_running_loop()
    upstream, _ = await loop.create_datagram_endpoint(Nameserver, local_addr=("127.0.0.1", 0))
    resolver = site_proxy.Resolver(Config(user=None, sites=[Site("chat", "http://gw/website/chat/", ["*.chat.test"])],
                                          nameservers=["127.0.0.1"]))
    answered = []
    resolver.transport = type("Transport", (), {"sendto": lambda self, data, addr: answered.append(data)})()
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(site_proxy, "NAMESERVER_PORT", upstream.get_extra_info("sockname")[1])
            for name, qtype in (("eu.chat.test", 1), ("eu.chat.test", 28), ("example.org", 1)):
                await resolver._answer(_query(name, qtype), None)
    finally:
        upstream.close()

    a, aaaa, other = answered
    assert struct.unpack("!HHHH", a[:8]) == (0x1234, 0x8580, 1, 1) and a.endswith(socket.inet_aton("127.0.0.1"))
    assert struct.unpack("!HHHH", aaaa[:8]) == (0x1234, 0x8580, 1, 0)
    assert passed == [_query("example.org", 1)] and other[2:4] == b"\x81\x80"


@pytest.mark.asyncio
async def test_dns_passes_on_an_answer_larger_than_a_classic_datagram_whole():
    class Nameserver(asyncio.DatagramProtocol):
        def connection_made(self, transport):
            self.transport = transport

        def datagram_received(self, data, addr):
            self.transport.sendto(data[:2] + b"\x81\x80" + data[4:] + b"\0" * 6000, addr)

    loop = asyncio.get_running_loop()
    upstream, _ = await loop.create_datagram_endpoint(Nameserver, local_addr=("127.0.0.1", 0))
    resolver = site_proxy.Resolver(Config(user=None, sites=[], nameservers=["127.0.0.1"]))
    answered = []
    resolver.transport = type("Transport", (), {"sendto": lambda self, data, addr: answered.append(data)})()
    query = _query("example.org", 16)
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(site_proxy, "NAMESERVER_PORT", upstream.get_extra_info("sockname")[1])
            await resolver._answer(query, None)
    finally:
        upstream.close()

    assert [len(reply) for reply in answered] == [len(query) + 6000]


@pytest.mark.asyncio
async def test_dns_over_tcp_answers_a_sites_name_and_passes_every_other_question_on_over_tcp():
    passed = []

    async def nameserver(reader, writer):
        (size,) = struct.unpack("!H", await reader.readexactly(2))
        query = await reader.readexactly(size)
        passed.append(query)
        reply = query[:2] + b"\x81\x80" + query[4:]
        writer.write(struct.pack("!H", len(reply)) + reply)
        await writer.drain()
        writer.close()

    upstream = await asyncio.start_server(nameserver, "127.0.0.1", 0)
    resolver = site_proxy.Resolver(Config(user=None, sites=[Site("chat", "http://gw/website/chat/", ["*.chat.test"])],
                                          nameservers=["127.0.0.1"]))
    server = await asyncio.start_server(resolver.handle_stream, "127.0.0.1", 0)
    answered = []
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(site_proxy, "NAMESERVER_PORT", upstream.sockets[0].getsockname()[1])
            reader, writer = await asyncio.open_connection("127.0.0.1", server.sockets[0].getsockname()[1])
            for name in ("eu.chat.test", "example.org"):
                query = _query(name, 1)
                writer.write(struct.pack("!H", len(query)) + query)
                await writer.drain()
                (size,) = struct.unpack("!H", await reader.readexactly(2))
                answered.append(await reader.readexactly(size))
            writer.close()
    finally:
        server.close()
        upstream.close()

    local, other = answered
    assert struct.unpack("!HHHH", local[:8]) == (0x1234, 0x8580, 1, 1)
    assert local.endswith(socket.inet_aton("127.0.0.1"))
    assert passed == [_query("example.org", 1)] and other[2:4] == b"\x81\x80"


@pytest.mark.asyncio
async def test_dns_with_no_nameserver_to_pass_on_to_still_answers_a_sites_name_and_fails_every_other_at_once():
    resolver = site_proxy.Resolver(Config(user=None, sites=[Site("chat", "http://gw/website/chat/", ["*.chat.test"])]))

    local = await resolver.answer(_query("eu.chat.test", 1), stream=False)
    other = await resolver.answer(_query("example.org", 1), stream=True)

    assert struct.unpack("!HHHH", local[:8]) == (0x1234, 0x8580, 1, 1)
    assert struct.unpack("!HHHH", other[:8]) == (0x1234, 0x8182, 1, 0) and other[12:] == _query("example.org", 1)[12:]


def test_the_hosts_block_is_replaced_not_added_to_and_an_empty_one_is_removed():
    first = site_proxy.hosts_text("127.0.0.1 localhost\n", ["notes.test", "chat.test"])
    again = site_proxy.hosts_text(first + "10.0.0.1 db\n", ["notes.test"])

    assert first == ("127.0.0.1 localhost\n# agentenv sites\n127.0.0.1 notes.test\n127.0.0.1 chat.test\n"
                     "# end agentenv sites\n")
    assert again == "127.0.0.1 localhost\n10.0.0.1 db\n# agentenv sites\n127.0.0.1 notes.test\n# end agentenv sites\n"
    assert site_proxy.hosts_text(again, []) == "127.0.0.1 localhost\n10.0.0.1 db\n"


def test_resolv_conf_asks_loopback_first_and_keeps_its_nameservers_to_pass_the_rest_to():
    text, passed = site_proxy.resolv_text("search corp\nnameserver 127.0.0.11\nnameserver 8.8.8.8\noptions ndots:0\n")

    assert text == "nameserver 127.0.0.1\nsearch corp\noptions ndots:0\n" and passed == ["127.0.0.11", "8.8.8.8"]


def test_firefoxs_policies_gain_the_ca_and_keep_every_other_policy():
    merged = json.loads(site_proxy.firefox_policies(json.dumps({"policies": {"DisableTelemetry": True}}), "/ca.pem"))

    assert merged == {"policies": {"DisableTelemetry": True, "Certificates": {"Install": ["/ca.pem"]}}}
    assert json.loads(site_proxy.firefox_policies(json.dumps(merged), "/ca.pem")) == merged


def test_install_writes_the_sites_files_names_and_trust_then_starts_the_proxy_on_them(tmp_path, monkeypatch):
    root, folder = tmp_path / "root", tmp_path / "sites"
    (root / "etc").mkdir(parents=True)
    folder.mkdir()
    (root / "etc/hosts").write_text("127.0.0.1 localhost\n")
    (root / "etc/resolv.conf").write_text("nameserver 127.0.0.11\n")
    (root / "etc/pki/tls/certs").mkdir(parents=True)
    (root / "etc/pki/tls/certs/ca-bundle.crt").write_text("RED HAT ROOTS\n")
    folder.chmod(0o700)
    table = sites.site_table([SiteParams(name="notes", url="http://gw/website/notes/", domains=["notes.test"]),
                              SiteParams(name="chat", url="http://gw/website/chat/", domains=["*.chat.test"]),
                              SiteParams(name="bare", url="http://gw/website/bare/")])
    issued = sites.issue(table.routes, label="t")
    (folder / "config.json").write_text(json.dumps(sites.proxy_config(table, issued, user=None, os_type="linux",
                                                                      forward_port=True, direct=["github.com"])))
    started = []
    monkeypatch.setattr(site_proxy, "start", lambda path, log: started.append(Config.load(path)))
    monkeypatch.setattr(site_proxy, "checks_as", lambda user, path: {"sites": {"notes.test": {"status": 200}},
                                                                     "runtimes": {"node": {"status": 200}}})
    monkeypatch.setattr(site_proxy.shutil, "which", lambda tool: None)

    report = site_proxy.install(str(folder / "config.json"), site_proxy.Paths(root))

    (config,) = started
    assert [(s.name, s.domains, bool(s.cert)) for s in config.sites] == [
        ("notes", ["notes.test"], True), ("chat", ["*.chat.test"], True), ("bare", ["bare.localhost"], False)]
    assert (config.dns_port, config.nameservers) == (53, ["127.0.0.11"])
    assert (config.proxy_port, config.direct) == (sites.FORWARD_PROXY_PORT, ["github.com"])
    assert sites.FORWARD_PROXY_PORT == 3129
    assert [(folder / name).stat().st_mode & 0o777 for name in ("notes.key", "config.json")] == [0o600, 0o600]
    assert folder.stat().st_mode & 0o777 == 0o755
    assert "key_pem" not in (folder / "serve.json").read_text()
    assert (root / "etc/hosts").read_text().splitlines()[2:4] == ["127.0.0.1 notes.test", "127.0.0.1 bare.localhost"]
    assert (root / "etc/resolv.conf").read_text() == "nameserver 127.0.0.1\n"
    assert json.loads((root / "etc/firefox/policies/policies.json").read_text())["policies"]["Certificates"] == {
        "Install": [str(folder / "ca.pem")]}
    assert (folder / "ca.pem").read_text() == issued.ca_pem
    for policy in ("etc/opt/chrome_for_testing", "etc/chromium", "etc/chromium-browser", "etc/opt/chrome"):
        (der,) = json.loads((root / policy / "policies/managed/agentenv-sites.json").read_text())["CACertificates"]
        assert base64.b64decode(der) == ssl.PEM_cert_to_DER_cert(issued.ca_pem)
    assert report["trust"]["system"] is None and report["trust"]["nss"] == [] and report["trust"]["java"] == {}
    assert len(report["trust"]["chromium"]) == 4
    assert report["checks"] == {"notes.test": {"status": 200}} and report["runtimes"] == {"node": {"status": 200}}
    assert (root / "etc/agentenv/ca.pem").read_text() == issued.ca_pem.strip() + "\n"
    assert (root / "etc/agentenv/ca-bundle.pem").read_text() == "RED HAT ROOTS\n" + issued.ca_pem.strip() + "\n"
    for added in ("etc/agentenv/ca.pem", "etc/agentenv/ca-bundle.pem"):
        assert (root / added).stat().st_mode & 0o777 == 0o644
    assert report["trust"]["added"] == [str(root / "etc/agentenv/ca.pem"), str(root / "etc/agentenv/ca-bundle.pem")]
    assert (root / "etc/profile.d/agentenv-sites.sh").read_text() == "".join(
        f"export {name}={value}\n" for name, value in sites.TRUST_ENVIRONMENTS["linux"].items())


def test_every_field_of_a_sites_config_is_served_as_given_or_kept_private_and_the_proxy_loads_each():
    """A site's config entry, with every field proxy_config writes, holds only fields install serves as given or keeps
    out of what the session user reads; and the proxy's Site takes each served field, and the files and headers made
    of the private ones, so a field added to one and not the others fails here, not silently."""
    table = sites.site_table([SiteParams(name="notes", url="http://gw/website/notes/", domains=["notes.test"])])
    given = sites.proxy_config(table, sites.issue(table.routes, label="t"), user=None, os_type="linux",
                               forward_port=True, upstream_headers={"notes": {"X-Gateway": "g"}},
                               sign_in_headers={"notes": ["X-Auto-Login"]})
    (entry,) = given["sites"]
    loaded = {f.name for f in dataclasses.fields(site_proxy.Site) if f.init}

    assert set(entry) == set(site_proxy.SERVED_SITE_FIELDS) | set(site_proxy.PRIVATE_SITE_FIELDS), sorted(entry)
    assert loaded == set(site_proxy.SERVED_SITE_FIELDS) | {"cert", "key", "upstream_headers"}, sorted(loaded)


def test_install_keeps_a_sites_sign_in_headers_for_the_proxy_it_starts_whose_network_log_takes_their_values_out(
        tmp_path, monkeypatch):
    root, folder = tmp_path / "root", tmp_path / "sites"
    (root / "etc/ssl/certs").mkdir(parents=True)
    (root / "etc/ssl/certs/ca-certificates.crt").write_text("PUBLIC ROOTS\n")
    folder.mkdir()
    folder.chmod(0o700)
    table = sites.site_table([SiteParams(name="shop", url="http://gw/website/shop/", domains=["shop.localhost"]),
                              SiteParams(name="notes", url="http://gw/website/notes/", domains=["notes.test"])])
    given = sites.proxy_config(table, sites.issue(table.routes, label="t"), user=None, os_type="linux",
                               forward_port=True, sign_in_headers={"shop": ["X-Auto-Login"]})
    (folder / "config.json").write_text(json.dumps(given))
    started = []
    monkeypatch.setattr(site_proxy, "start", lambda path, log: started.append(Config.load(path)))
    monkeypatch.setattr(site_proxy, "checks_as", lambda user, path: {"sites": {}, "runtimes": {}})
    monkeypatch.setattr(site_proxy.shutil, "which", lambda tool: None)

    site_proxy.install(str(folder / "config.json"), site_proxy.Paths(root))

    (config,) = started
    served = {site.name: site for site in config.sites}
    assert (served["shop"].sign_in_headers, served["notes"].sign_in_headers) == (["X-Auto-Login"], [])
    assert site_proxy.logged_headers([("x-auto-login", "admin:test")], served["shop"].sign_in_headers) == [
        ["x-auto-login", site_proxy.REMOVED]]


def test_install_keeps_the_upstream_headers_in_a_file_root_alone_reads_and_serve_json_names_it(tmp_path, monkeypatch):
    root, folder = tmp_path / "root", tmp_path / "sites"
    (root / "etc/ssl/certs").mkdir(parents=True)
    (root / "etc/ssl/certs/ca-certificates.crt").write_text("PUBLIC ROOTS\n")
    folder.mkdir()
    folder.chmod(0o700)
    table = sites.site_table([SiteParams(name="notes", url="http://gw/website/notes/", domains=["notes.test"])])
    issued = sites.issue(table.routes, label="t")
    given = sites.proxy_config(table, issued, user=None, os_type="linux", forward_port=False,
                               upstream_headers={"notes": {"CF-Access-Client-Secret": "s3cret"}})
    (folder / "config.json").write_text(json.dumps(given))
    started = []
    monkeypatch.setattr(site_proxy, "start", lambda path, log: started.append(Config.load(path)))
    monkeypatch.setattr(site_proxy, "checks_as", lambda user, path: {"sites": {}, "runtimes": {}})
    monkeypatch.setattr(site_proxy.shutil, "which", lambda tool: None)

    site_proxy.install(str(folder / "config.json"), site_proxy.Paths(root))

    (config,) = started
    assert config.sites[0].upstream_headers == {"CF-Access-Client-Secret": "s3cret"}
    private = folder / site_proxy.UPSTREAM_HEADERS_FILE
    assert private.stat().st_mode & 0o777 == 0o600 and (folder / "config.json").stat().st_mode & 0o777 == 0o600
    serving = (folder / "serve.json").read_text()
    assert "s3cret" not in serving and json.loads(serving)["upstream_headers"] == str(private)
    read_text = Path.read_text

    def refused(path, *args, **kwargs):  # what the session user meets; root, which CI may run as, reads any mode
        if path == private:
            raise PermissionError(13, "Permission denied", str(path))
        return read_text(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_text", refused)
        assert Config.load(str(folder / "serve.json")).sites[0].upstream_headers == {}


def test_the_checks_are_given_each_sites_timeout_in_turn(tmp_path, monkeypatch):
    serve_path = tmp_path / "serve.json"
    serve_path.write_text(json.dumps({"user": None, "sites": [
        {"name": f"s{i}", "upstream": f"http://gw/website/s{i}/", "domains": [f"s{i}.localhost"]} for i in range(7)]}))
    given = []

    def run(argv, **kwargs):
        given.append(kwargs["timeout"])
        return site_proxy.subprocess.CompletedProcess(argv, 0, stdout="{}", stderr="")

    monkeypatch.setattr(site_proxy.subprocess, "run", run)
    site_proxy.checks_as(None, str(serve_path))

    assert given[0] >= site_proxy.CHECK_TIMEOUT_S * 7


def test_the_sites_ca_joins_the_trust_a_provider_added_and_a_bundle_made_here_starts_from_the_systems(tmp_path):
    paths, ca_pem = site_proxy.Paths(tmp_path), "-----BEGIN CERTIFICATE-----\nU0lURVM=\n-----END CERTIFICATE-----\n"
    files = {"ca": "/etc/agentenv/ca.pem", "bundle": "/etc/agentenv/ca-bundle.pem"}
    (tmp_path / "etc/agentenv").mkdir(parents=True)
    (tmp_path / "etc/agentenv/ca.pem").write_text("LOCAL CA\n")
    (tmp_path / "etc/agentenv/ca-bundle.pem").write_text("PUBLIC ROOTS\nLOCAL CA\n")
    (tmp_path / "etc/agentenv").chmod(0o700)
    other = site_proxy.Paths(tmp_path / "other")
    (tmp_path / "other/etc/ssl/certs").mkdir(parents=True)
    (tmp_path / "other/etc/ssl/certs/ca-certificates.crt").write_text("SYSTEM ROOTS\n")

    site_proxy.add_trust_files(paths, files, ca_pem)
    site_proxy.add_trust_files(paths, files, ca_pem)
    site_proxy.add_trust_files(other, files, ca_pem)

    assert (tmp_path / "etc/agentenv/ca.pem").read_text() == "LOCAL CA\n" + ca_pem
    assert (tmp_path / "etc/agentenv/ca-bundle.pem").read_text() == "PUBLIC ROOTS\nLOCAL CA\n" + ca_pem
    assert (tmp_path / "etc/agentenv").stat().st_mode & 0o777 == 0o755
    assert (tmp_path / "other/etc/agentenv/ca.pem").read_text() == ca_pem
    assert (tmp_path / "other/etc/agentenv/ca-bundle.pem").read_text() == "SYSTEM ROOTS\n" + ca_pem


def test_the_bundle_is_made_again_from_the_system_store_with_a_ca_an_app_added_after_the_image_was_built(tmp_path):
    paths, files = site_proxy.Paths(tmp_path), {"ca": "/etc/agentenv/ca.pem", "bundle": "/etc/agentenv/ca-bundle.pem"}
    app, local, sites_ca = (f"-----BEGIN CERTIFICATE-----\n{name}\n-----END CERTIFICATE-----\n"
                            for name in ("QVBQ", "TE9DQUw=", "U0lURVM="))
    (tmp_path / "etc/agentenv").mkdir(parents=True)
    (tmp_path / "etc/agentenv/ca.pem").write_text(local)
    (tmp_path / "etc/agentenv/ca-bundle.pem").write_text("ROOTS AT BUILD\n")
    (tmp_path / "etc/ssl/certs").mkdir(parents=True)
    (tmp_path / "etc/ssl/certs/ca-certificates.crt").write_text("ROOTS AT BUILD\n" + app + sites_ca)

    site_proxy.add_trust_files(paths, files, sites_ca)
    site_proxy.add_trust_files(paths, files, sites_ca)

    assert (tmp_path / "etc/agentenv/ca.pem").read_text() == local + sites_ca
    assert (tmp_path / "etc/agentenv/ca-bundle.pem").read_text() == "ROOTS AT BUILD\n" + app + sites_ca + local


def test_a_bundle_with_no_public_roots_to_start_from_is_refused_before_either_file_is_written(tmp_path):
    paths, ca_pem = site_proxy.Paths(tmp_path), "-----BEGIN CERTIFICATE-----\nU0lURVM=\n-----END CERTIFICATE-----\n"
    files = {"ca": "/etc/agentenv/ca.pem", "bundle": "/etc/agentenv/ca-bundle.pem"}
    (tmp_path / "etc/agentenv").mkdir(parents=True)
    (tmp_path / "etc/agentenv/ca-bundle.pem").write_text("\n")

    with pytest.raises(SystemExit, match="there are no public roots to make /etc/agentenv/ca-bundle.pem from: none "
                                         "of .*etc/ssl/certs/ca-certificates.crt, .*etc/pki/tls/certs/ca-bundle.crt"):
        site_proxy.add_trust_files(paths, files, ca_pem)

    assert not (tmp_path / "etc/agentenv/ca.pem").exists()
    assert (tmp_path / "etc/agentenv/ca-bundle.pem").read_text() == "\n"


# A keytool whose store is a text file of the SHA-256 fingerprints it holds: -importcert adds the file's, -list -v
# prints them, -delete does nothing; each call's operation is logged beside the store. With FAIL in the store,
# -importcert fails as a store with another password does.
FAKE_KEYTOOL = """#!{python}
import base64, hashlib, sys
args = sys.argv[1:]
store = args[args.index("-keystore") + 1]
with open(store + ".calls", "a") as calls:
    calls.write(args[0] + "\\n")
if args[0] == "-importcert":
    if "FAIL" in open(store).read():
        sys.exit("keytool error: java.io.IOException: keystore password was incorrect")
    pem = open(args[args.index("-file") + 1]).read()
    body = "".join(line.strip() for line in pem.splitlines() if line.strip() and not line.startswith("-----"))
    digest = hashlib.sha256(base64.b64decode(body)).digest()
    with open(store, "a") as f:
        f.write("SHA256: " + ":".join(f"{{byte:02X}}" for byte in digest) + "\\n")
elif args[0] == "-list":
    print(open(store).read())
"""


def _jdk(home: Path, store_text: str = "", keytool: bool = True) -> Path:
    """A JDK at ``home``: its store, holding ``store_text``, and with ``keytool``, a fake one in its bin."""
    store = home / "lib/security/cacerts"
    store.parent.mkdir(parents=True)
    store.write_text(store_text)
    if keytool:
        (home / "bin").mkdir()
        (home / "bin/keytool").write_text(FAKE_KEYTOOL.format(python=sys.executable))
        (home / "bin/keytool").chmod(0o755)
    return store


def _calls(store: Path) -> list[str]:
    calls = Path(f"{store}.calls")
    return calls.read_text().split() if calls.exists() else []


def test_every_jdk_trusts_the_ca_an_apt_one_through_the_system_hook_the_rest_by_keytool(tmp_path, monkeypatch):
    issued = sites.issue(sites.site_table([SiteParams(name="notes", url="http://gw/website/notes/",
                                                      domains=["notes.test"])]).routes, label="t")
    fingerprint = site_proxy.sha256_fingerprint(issued.ca_pem)
    ca = tmp_path / "ca.pem"
    ca.write_text(issued.ca_pem)
    root = tmp_path / "root"
    hook = root / "etc/ssl/certs/java/cacerts"
    hook.parent.mkdir(parents=True)
    hook.write_text(f"SHA256: {fingerprint}\n")
    temurin = _jdk(root / "usr/lib/jvm/temurin-21")
    apt = _jdk(root / "usr/lib/jvm/java-21-openjdk-amd64")
    apt.unlink()
    apt.symlink_to(hook)
    (root / "usr/lib/jvm/default-java").symlink_to(root / "usr/lib/jvm/java-21-openjdk-amd64")
    bundled = _jdk(root / "opt/agentenv/apps/ide/idea/jbr", keytool=False)
    locked = _jdk(root / "opt/locked/jdk", "FAIL\n")
    skipped = _jdk(root / "opt/app/node_modules/jre")
    monkeypatch.setattr(site_proxy.shutil, "which",
                        lambda tool: str(root / "usr/lib/jvm/temurin-21/bin/keytool") if tool == "keytool" else None)

    trusted = site_proxy.trust_java(site_proxy.Paths(root), str(ca), issued.ca_pem)

    assert trusted == {str(temurin): {"trusted": True}, str(apt): {"trusted": True},
                       str(bundled): {"trusted": True},
                       str(locked): {"error": "keytool error: java.io.IOException: keystore password was incorrect"}}
    assert _calls(temurin) == _calls(bundled) == ["-delete", "-importcert", "-list"]
    assert _calls(apt) == ["-list"] and _calls(skipped) == []
    assert fingerprint in temurin.read_text() and fingerprint in bundled.read_text()
    assert site_proxy.trust_java(site_proxy.Paths(tmp_path / "bare"), str(ca), issued.ca_pem) == {}


def test_a_jdk_store_the_ca_isnt_listed_in_after_routing_is_named(tmp_path, monkeypatch):
    root = tmp_path / "root"
    hook = root / "etc/ssl/certs/java/cacerts"
    hook.parent.mkdir(parents=True)
    hook.write_text("SHA256: 00:11\n")
    apt = _jdk(root / "usr/lib/jvm/java-21-openjdk-amd64")
    apt.unlink()
    apt.symlink_to(hook)
    issued = sites.issue(sites.site_table([SiteParams(name="notes", url="http://gw/website/notes/",
                                                      domains=["notes.test"])]).routes, label="t")

    trusted = site_proxy.trust_java(site_proxy.Paths(root), str(tmp_path / "ca.pem"), issued.ca_pem)

    assert trusted == {str(apt): {"error": "keytool -list doesn't list the CA in it"}}


def test_each_runtime_loads_the_first_https_site_through_its_own_trust_and_says_why_one_couldnt(tmp_path, monkeypatch):
    programs = {"node": "#!/bin/sh\necho 200\n",
                "curl": "#!/bin/sh\necho 'curl: (60) SSL certificate problem: unable to get local issuer "
                        "certificate' >&2\nexit 60\n"}
    for name, text in programs.items():
        (tmp_path / name).write_text(text)
        (tmp_path / name).chmod(0o755)
    monkeypatch.setattr(site_proxy.shutil, "which", lambda tool: str(tmp_path / tool) if tool in programs else None)
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    unset = site_proxy.check_runtimes("https://notes.test/")
    issued = sites.issue(sites.site_table([SiteParams(name="notes", url="http://gw/website/notes/",
                                                      domains=["notes.test"])]).routes, label="t")
    (tmp_path / "bundle.pem").write_text(issued.ca_pem)
    monkeypatch.setenv("SSL_CERT_FILE", str(tmp_path / "bundle.pem"))
    opened = []

    class Opener:
        def open(self, url, timeout):
            opened.append(url)
            return contextlib.nullcontext(SimpleNamespace(status=200))

    monkeypatch.setattr(site_proxy.urllib.request, "build_opener", lambda *handlers: Opener())
    set_ = site_proxy.check_runtimes("https://notes.test/")

    assert unset == {
        "python SSL_CERT_FILE": {"url": "https://notes.test/",
                                 "error": "SSL_CERT_FILE isn't set in the computer's environment"},
        "node": {"url": "https://notes.test/", "status": 200},
        "curl": {"url": "https://notes.test/",
                 "error": "curl: (60) SSL certificate problem: unable to get local issuer certificate"}}
    assert set_["python SSL_CERT_FILE"] == {"url": "https://notes.test/", "status": 200}
    assert opened == ["https://notes.test/"]


@pytest.mark.parametrize("resolv", [None, "", "nameserver 127.0.0.1\n"], ids=["missing", "empty", "loopback-only"])
def test_a_wildcard_name_starts_dns_even_when_resolv_conf_names_no_other_nameserver(tmp_path, monkeypatch, resolv):
    root, folder = tmp_path / "root", tmp_path / "sites"
    (root / "etc/ssl/certs").mkdir(parents=True)
    (root / "etc/ssl/certs/ca-certificates.crt").write_text("PUBLIC ROOTS\n")
    folder.mkdir()
    if resolv is not None:
        (root / "etc/resolv.conf").write_text(resolv)
    table = sites.site_table([SiteParams(name="chat", url="http://gw/website/chat/", domains=["*.chat.test"])])
    issued = sites.issue(table.routes, label="t")
    (folder / "config.json").write_text(json.dumps(sites.proxy_config(table, issued, user=None, os_type="linux",
                                                                      forward_port=False)))
    started = []
    monkeypatch.setattr(site_proxy, "start", lambda path, log: started.append(Config.load(path)))
    monkeypatch.setattr(site_proxy, "checks_as", lambda user, path: {"sites": {}, "runtimes": {}})
    monkeypatch.setattr(site_proxy.shutil, "which", lambda tool: None)

    site_proxy.install(str(folder / "config.json"), site_proxy.Paths(root))

    (config,) = started
    assert (config.dns_port, config.nameservers) == (53, [])
    assert (root / "etc/resolv.conf").read_text() == "nameserver 127.0.0.1\n"


def test_the_proxy_serves_as_nobody_where_the_base_has_it_else_as_the_session_user(tmp_path, monkeypatch):
    monkeypatch.setattr(site_proxy.pwd, "getpwnam", lambda name: SimpleNamespace(pw_name=name))
    assert site_proxy.serve_user("agent") == "nobody"

    def missing(name):
        raise KeyError(name)

    monkeypatch.setattr(site_proxy.pwd, "getpwnam", missing)
    assert site_proxy.serve_user("agent") == "agent"
    serve_path = tmp_path / "serve.json"
    serve_path.write_text(json.dumps({"user": "agent", "serve_user": "nobody", "sites": []}))
    assert (Config.load(str(serve_path)).user, Config.load(str(serve_path)).serve_user) == ("agent", "nobody")


def test_the_proxy_imports_when_pwd_is_unavailable():
    subprocess.run([sys.executable, "-c", "import sys; sys.modules['pwd'] = None; "
                    "from agent_env.env.envs.computer.utils import site_proxy, sites; assert site_proxy.pwd is None"],
                   check=True)


def test_keep_starts_serve_again_each_time_it_ends(monkeypatch):
    runs = []

    def run(argv, **kwargs):
        runs.append(argv[-2:])
        return site_proxy.subprocess.CompletedProcess(argv, 1)

    def sleep(seconds):
        if len(runs) == 3:
            raise StopIteration

    monkeypatch.setattr(site_proxy.subprocess, "run", run)
    monkeypatch.setattr(site_proxy.time, "sleep", sleep)
    with pytest.raises(StopIteration):
        site_proxy.keep("/etc/agentenv/sites/serve.json")

    assert runs == [["serve", "/etc/agentenv/sites/serve.json"]] * 3


def test_the_log_is_root_owned_beside_the_config_and_a_link_at_its_path_is_refused(tmp_path):
    assert sites.proxy_config(sites.site_table([]), SimpleNamespace(ca_pem=None, leaves={}), user=None,
                              forward_port=False, os_type="linux")["log"] == str(sites.FOLDER / "site-proxy.log")
    target, log = tmp_path / "elsewhere", tmp_path / "site-proxy.log"
    target.write_text("kept")
    log.symlink_to(target)
    with pytest.raises(OSError):
        site_proxy.open_log(str(log))
    assert target.read_text() == "kept"
    os.close(site_proxy.open_log(str(tmp_path / "logs" / "fresh.log")))
    assert (tmp_path / "logs" / "fresh.log").stat().st_mode & 0o777 == 0o600


def test_the_proxy_is_standard_library_alone_so_the_bases_own_python_runs_it():
    source = Path(site_proxy.__file__).read_text()
    imported = {line.split()[1].split(".")[0] for line in source.splitlines()
                if line.startswith(("import ", "from ")) and not line.startswith("from __future__")}

    assert imported <= {"asyncio", "base64", "datetime", "hashlib", "ipaddress", "itertools", "json", "os", "pwd", "re",
                        "shutil", "socket", "ssl", "struct", "subprocess", "sys", "time", "urllib", "dataclasses",
                        "pathlib", "typing"}
