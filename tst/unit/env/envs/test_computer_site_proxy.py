"""The site proxy a computer reaches websites by their names through: what it forwards to a site's route on the
gateway and how, what it answers with, its certificates by name, its DNS for wildcard names, and what its install
writes. The proxy is a network program, so these tests open sockets, on loopback alone."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import email.message
import io
import json
import os
import socket
import ssl
import struct
import urllib.error
from datetime import timedelta
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
async def _proxy(config, scheme="https"):
    proxy = SiteProxy(config)
    context = site_proxy.tls_context(config) if scheme == "https" else None
    server = await asyncio.start_server(lambda r, w: proxy.handle(r, w, scheme), "127.0.0.1", 0, ssl=context)
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
    issued = sites.issue(routes, valid_for=timedelta(hours=1), label="test")
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
    config = Config(user=None, sites=[Site("notes", "http://gw/website/notes/", ["notes.test"]),
                                      Site("chat", "http://gw/website/chat/", ["chat.test"])])

    assert site_proxy.check(config) == {
        "notes.test": {"url": "https://notes.test/", "error": "502 Bad Gateway: the site can't be reached"},
        "chat.test": {"url": "https://chat.test/", "status": 502}}


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
    folder.chmod(0o700)
    table = sites.site_table([SiteParams(name="notes", url="http://gw/website/notes/", domains=["notes.test"]),
                              SiteParams(name="chat", url="http://gw/website/chat/", domains=["*.chat.test"]),
                              SiteParams(name="bare", url="http://gw/website/bare/")])
    issued = sites.issue(table.routes, valid_for=timedelta(hours=1), label="t")
    (folder / "config.json").write_text(json.dumps(sites.proxy_config(table, issued, user=None)))
    started = []
    monkeypatch.setattr(site_proxy, "start", lambda path, log: started.append(Config.load(path)))
    monkeypatch.setattr(site_proxy, "checks_as", lambda user, path: {"notes.test": {"status": 200}})
    monkeypatch.setattr(site_proxy.shutil, "which", lambda tool: None)

    report = site_proxy.install(str(folder / "config.json"), site_proxy.Paths(root))

    (config,) = started
    assert [(s.name, s.domains, bool(s.cert)) for s in config.sites] == [
        ("notes", ["notes.test"], True), ("chat", ["*.chat.test"], True), ("bare", ["bare.localhost"], False)]
    assert (config.dns_port, config.nameservers) == (53, ["127.0.0.11"])
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
    assert report["trust"]["system"] is None and report["trust"]["nss"] == []
    assert len(report["trust"]["chromium"]) == 4
    assert report["checks"] == {"notes.test": {"status": 200}}


def test_install_keeps_the_upstream_headers_in_a_file_root_alone_reads_and_serve_json_names_it(tmp_path, monkeypatch):
    root, folder = tmp_path / "root", tmp_path / "sites"
    (root / "etc").mkdir(parents=True)
    folder.mkdir()
    folder.chmod(0o700)
    table = sites.site_table([SiteParams(name="notes", url="http://gw/website/notes/", domains=["notes.test"])])
    issued = sites.issue(table.routes, valid_for=timedelta(hours=1), label="t")
    given = sites.proxy_config(table, issued, user=None, upstream_headers={"notes": {"CF-Access-Client-Secret": "s3cret"}})
    (folder / "config.json").write_text(json.dumps(given))
    started = []
    monkeypatch.setattr(site_proxy, "start", lambda path, log: started.append(Config.load(path)))
    monkeypatch.setattr(site_proxy, "checks_as", lambda user, path: {})
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


@pytest.mark.parametrize("resolv", [None, "", "nameserver 127.0.0.1\n"], ids=["missing", "empty", "loopback-only"])
def test_a_wildcard_name_starts_dns_even_when_resolv_conf_names_no_other_nameserver(tmp_path, monkeypatch, resolv):
    root, folder = tmp_path / "root", tmp_path / "sites"
    (root / "etc").mkdir(parents=True)
    folder.mkdir()
    if resolv is not None:
        (root / "etc/resolv.conf").write_text(resolv)
    table = sites.site_table([SiteParams(name="chat", url="http://gw/website/chat/", domains=["*.chat.test"])])
    issued = sites.issue(table.routes, valid_for=timedelta(hours=1), label="t")
    (folder / "config.json").write_text(json.dumps(sites.proxy_config(table, issued, user=None)))
    started = []
    monkeypatch.setattr(site_proxy, "start", lambda path, log: started.append(Config.load(path)))
    monkeypatch.setattr(site_proxy, "checks_as", lambda user, path: {})
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
    assert sites.proxy_config(sites.site_table([]), SimpleNamespace(ca_pem=None, leaves={}), user=None)["log"] == str(sites.FOLDER / "site-proxy.log")
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

    assert imported <= {"asyncio", "json", "os", "pwd", "shutil", "socket", "ssl", "struct", "subprocess", "sys",
                        "time", "urllib", "dataclasses", "pathlib", "typing"}
