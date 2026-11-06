"""The site proxy: inside a computer, it serves each website of the deployment under the names the site declares
(``https://notes.test/``), forwarding every request to the site's route on the env gateway
(``<gateway>/website/notes/``), the only address the computer reaches the site at. It runs on the base's own
``python3``, so it uses the standard library alone.

``install``, as root, maps each exact name to loopback in the hosts file, trusts the deployment's CA in the system
store, Firefox's policies, Chromium's managed policies and the session user's NSS databases, starts ``serve`` and checks, as the session user, that
each name loads. ``serve`` listens on loopback: on 443 for a name served over HTTPS, on 80 to redirect those to HTTPS
and to serve ``.localhost`` names over HTTP, and on 53 for DNS, over UDP and TCP, when a name is a wildcard, which a
hosts file can't list; it then drops to ``SERVE_USER`` (``nobody``), else the session user, so the agent, which runs as
the session user, can't stop it, and ``keep``, as root, starts it again should it end. The site sees itself at ``/`` under its own name: the path goes upstream under
the route's prefix, ``X-Forwarded-Host`` and ``X-Forwarded-Proto`` say the name and scheme the browser used,
``AgentEnv-Site-Root`` has the gateway tell the site no prefix, so the links it makes are rooted at ``/`` too, and a
``Location`` or cookie naming the route is rewritten back to the name. Hop-by-hop headers and the browser's own
``X-Forwarded-*`` never go upstream; ``Transfer-Encoding`` does, since bodies are relayed as framed, and an
``Alt-Svc`` offering HTTP/3, which the proxy doesn't serve, never reaches the browser. A WebSocket's upgrade is
relayed, for an upstream that answers it."""

from __future__ import annotations

import asyncio
import json
import os
import pwd
import shutil
import socket
import ssl
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit, urlunsplit

HOSTS_BEGIN = "# agentenv sites"
HOSTS_END = "# end agentenv sites"
LOOPBACK = "127.0.0.1"
CA_NAME = "agentenv-sites"
NSS_NICKNAME = "agentenv sites"
HOP_BY_HOP = frozenset({"connection", "keep-alive", "proxy-connection", "proxy-authorization", "proxy-authenticate",
                        "te", "trailer", "upgrade", "host", "expect"})
SITE_ROOT_HEADER = "AgentEnv-Site-Root"
FORWARDED = frozenset({"x-forwarded-host", "x-forwarded-proto", "x-forwarded-for", "x-forwarded-port",
                       "x-forwarded-prefix", "forwarded", SITE_ROOT_HEADER.lower()})
RESPONSE_DROP = frozenset({"connection", "keep-alive", "proxy-connection", "alt-svc"})
UPSTREAM_HEADERS_FILE = "upstream_headers.json"
HEAD_LIMIT = 1 << 20
CHUNK = 1 << 16
UPSTREAM_TIMEOUT_S = 30.0
DNS_TIMEOUT_S = 4.0
NAMESERVER_PORT = 53
DATAGRAM_MAX = 65535
LISTEN_TIMEOUT_S = 15.0
# How long one try to connect to the proxy waiting to listen may take, the wait between tries, and how much of its
# log the failure shows.
LISTEN_PROBE_S, LISTEN_POLL_S, LOG_TAIL_CHARS = 1.0, 0.2, 2000
CHECK_TIMEOUT_S = 15.0
RESTART_DELAY_S = 1.0
SERVE_USER = "nobody"
FAILED = (("X-Agentenv-Site-Proxy", "failed"),)


class BadRequest(Exception):
    """A request the proxy can't read."""


def is_local(name: str) -> bool:
    """A ``.localhost`` name, served over HTTP: browsers treat it as secure without TLS."""
    return name == "localhost" or name.endswith(".localhost")


def matches(pattern: str, host: str) -> bool:
    """Whether ``host`` is the name ``pattern`` names: itself, or for ``*.example.com`` one label more, as a
    certificate's wildcard matches."""
    host = host.lower().rstrip(".")
    if pattern.startswith("*."):
        head, dot, rest = host.partition(".")
        return bool(head) and bool(dot) and rest == pattern[2:]
    return host == pattern


@dataclass
class Site:
    """One site as the proxy serves it: its names, the route on the gateway it forwards to, and its certificate and key
    files, for a site with a name served over HTTPS."""

    name: str
    upstream: str
    domains: list[str]
    cert: Optional[str] = None
    key: Optional[str] = None
    upstream_headers: dict[str, str] = field(default_factory=dict)
    scheme: str = field(init=False)
    host: str = field(init=False)
    port: int = field(init=False)
    prefix: str = field(init=False)

    def __post_init__(self) -> None:
        parts = urlsplit(self.upstream)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError(f"site {self.name}: its upstream {self.upstream!r} isn't an http(s) URL")
        self.scheme, self.host = parts.scheme, parts.hostname
        self.port = parts.port or (443 if parts.scheme == "https" else 80)
        self.prefix = parts.path.rstrip("/")

    @property
    def netloc(self) -> str:
        default = 443 if self.scheme == "https" else 80
        return self.host if self.port == default else f"{self.host}:{self.port}"

    def tls(self, name: str) -> bool:
        return not is_local(name)


@dataclass
class Config:
    """What ``serve`` reads: the session user, the user it drops to (``serve_user``, else the session user), its
    ports, the sites and, with a wildcard name, the nameservers it forwards every other question to."""

    user: Optional[str]
    sites: list[Site]
    serve_user: Optional[str] = None
    listen: str = LOOPBACK
    https_port: int = 443
    http_port: int = 80
    dns_port: Optional[int] = None
    nameservers: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: str) -> Config:
        """The config at ``path``, with each site's upstream headers from the private file it names, which root alone
        reads: ``serve`` reads it before it drops privileges, and ``check``, as the session user, goes without."""
        data = json.loads(Path(path).read_text())
        headers: dict = {}
        if data.get("upstream_headers"):
            try:
                headers = json.loads(Path(data["upstream_headers"]).read_text())
            except PermissionError:
                headers = {}
        sites = [Site(name=s["name"], upstream=s["upstream"], domains=list(s["domains"]), cert=s.get("cert"),
                      key=s.get("key"), upstream_headers=dict(headers.get(s["name"]) or {})) for s in data["sites"]]
        return cls(user=data.get("user"), sites=sites, serve_user=data.get("serve_user"),
                   listen=data.get("listen", LOOPBACK),
                   https_port=data.get("https_port", 443), http_port=data.get("http_port", 80),
                   dns_port=data.get("dns_port"), nameservers=list(data.get("nameservers") or []))

    def site_for(self, host: str) -> Optional[Site]:
        """The site ``host`` names: one naming it exactly before one whose wildcard covers it."""
        host = host.lower().rstrip(".")
        for exact in (True, False):
            for site in self.sites:
                if any(matches(name, host) for name in site.domains if name.startswith("*.") != exact):
                    return site
        return None


def header(headers: list[tuple[str, str]], name: str) -> Optional[str]:
    return next((value for key, value in headers if key.lower() == name), None)


def tokens(headers: list[tuple[str, str]], name: str) -> set[str]:
    return {token.strip().lower() for key, value in headers if key.lower() == name for token in value.split(",")}


def parse_head(data: bytes) -> tuple[str, list[tuple[str, str]]]:
    """A message head's start line and its headers, in order."""
    lines = data.decode("latin-1").removesuffix("\r\n\r\n").split("\r\n")
    headers = []
    for line in lines[1:]:
        key, colon, value = line.partition(":")
        if not colon or not key or key != key.strip():
            raise BadRequest(f"a malformed header line {line!r}")
        headers.append((key, value.strip()))
    return lines[0], headers


def split_host(host: str) -> str:
    """The name of a ``Host`` value, without its port."""
    if host.startswith("["):
        return host[1:].partition("]")[0]
    return host.rpartition(":")[0] if host.count(":") == 1 else host


def origin(scheme: str, netloc: str) -> tuple[str, str, int]:
    """A URL's scheme, name and port, its port the scheme's own when it names none."""
    parts = urlsplit(f"//{netloc}")
    scheme = scheme.lower()
    return scheme, (parts.hostname or "").lower(), parts.port or (443 if scheme == "https" else 80)


def public_location(location: str, site: Site, scheme: str, host: str) -> str:
    """A ``Location`` the upstream sent, as the browser follows it: one on the site's route, under the gateway's address,
    under the site's own name (as a site that ignores ``AgentEnv-Site-Root`` may send) or as a path, is rewritten onto
    the site's name; any other is left alone. A scheme-relative one is read with the scheme of the page it was sent
    to: the route's for the gateway's address, the browser's for the site's name."""
    parts = urlsplit(location)
    if parts.netloc:
        pages = ((site.scheme, site.netloc), (scheme, host))
        if not any(origin(parts.scheme or own, parts.netloc) == origin(own, netloc) for own, netloc in pages):
            return location
    elif parts.scheme or not location.startswith("/"):
        return location
    path = parts.path
    if not (path == site.prefix or path.startswith(site.prefix + "/")) or not site.prefix:
        return location
    rest = path[len(site.prefix):] or "/"
    origin_parts = (scheme, host) if parts.netloc else ("", "")
    return urlunsplit((*origin_parts, rest, parts.query, parts.fragment))


def public_cookie(value: str, site: Site) -> str:
    """A ``Set-Cookie`` the upstream sent, with any ``Domain`` naming the gateway dropped and a ``Path`` under the route
    rooted at ``/``, so the browser keeps it for the site's name."""
    first, *attributes = value.split(";")
    kept = [first]
    for attribute in attributes:
        key, _, attr_value = attribute.strip().partition("=")
        key, attr_value = key.strip(), attr_value.strip()
        if key.lower() == "domain" and attr_value.strip().lstrip(".").lower() == site.host.lower():
            continue
        if key.lower() == "path" and site.prefix and (attr_value == site.prefix
                                                      or attr_value.startswith(site.prefix + "/")):
            attribute = f" Path={attr_value[len(site.prefix):] or '/'}"
        kept.append(attribute)
    return ";".join(kept)


def upstream_request(method: str, target: str, headers: list[tuple[str, str]], site: Site, *, scheme: str,
                     host: str, peer: str, upgrade: Optional[str]) -> bytes:
    """The request head sent to the site's route: the path under its prefix, the gateway as ``Host``, the browser's
    name and scheme in ``X-Forwarded-Host`` and ``X-Forwarded-Proto``, ``AgentEnv-Site-Root`` naming the same origin, so
    the site is told it is at ``/`` there even through a front before the gateway that rewrites the others, the headers the gateway's sandbox provider asks for (``Site.upstream_headers``) in place of any the
    browser sent by those names, and a connection that closes after it, or upgrades when the browser asked to."""
    if not target.startswith("/"):
        parts = urlsplit(target)
        target = urlunsplit(("", "", parts.path or "/", parts.query, ""))
    provided = {key.lower() for key in site.upstream_headers}
    kept = [(key, value) for key, value in headers if key.lower() not in HOP_BY_HOP | FORWARDED | provided]
    kept += [("Host", site.netloc), ("X-Forwarded-Host", host), ("X-Forwarded-Proto", scheme),
             ("X-Forwarded-For", peer), (SITE_ROOT_HEADER, f"{scheme}://{host}"), *site.upstream_headers.items()]
    kept += [("Connection", "Upgrade"), ("Upgrade", upgrade)] if upgrade else [("Connection", "close")]
    lines = [f"{method} {site.prefix}{target} HTTP/1.1", *(f"{key}: {value}" for key, value in kept)]
    return ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1")


def response_head(status_line: str, headers: list[tuple[str, str]], site: Site, *, scheme: str, host: str,
                  connection: str) -> bytes:
    _, _, status = status_line.partition(" ")
    kept = []
    for key, value in headers:
        lower = key.lower()
        if lower in RESPONSE_DROP or (lower == "upgrade" and connection != "Upgrade"):
            continue
        if lower == "location":
            value = public_location(value, site, scheme, host)
        elif lower == "set-cookie":
            value = public_cookie(value, site)
        kept.append((key, value))
    kept.append(("Connection", connection))
    lines = [f"HTTP/1.1 {status}", *(f"{key}: {value}" for key, value in kept)]
    return ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1")


def simple_response(status: str, body: str, extra: tuple[tuple[str, str], ...] = ()) -> bytes:
    data = body.encode()
    lines = [f"HTTP/1.1 {status}", "Content-Type: text/plain; charset=utf-8", f"Content-Length: {len(data)}",
             *(f"{key}: {value}" for key, value in extra), "Connection: close"]
    return ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1") + data


async def read_head(reader: asyncio.StreamReader) -> Optional[bytes]:
    """The next message head, or None when the connection closed before one began."""
    try:
        return await reader.readuntil(b"\r\n\r\n")
    except asyncio.IncompleteReadError as e:
        if not e.partial.strip():
            return None
        raise BadRequest("the connection closed mid-head") from None
    except asyncio.LimitOverrunError:
        raise BadRequest("a head over the limit") from None


async def relay_exact(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, size: int) -> None:
    while size > 0:
        data = await reader.read(min(size, CHUNK))
        if not data:
            raise BadRequest("the body ended early")
        writer.write(data)
        await writer.drain()
        size -= len(data)


async def relay_chunked(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """A chunked body relayed as framed, to its last chunk and trailers."""
    while True:
        line = await reader.readuntil(b"\r\n")
        writer.write(line)
        try:
            size = int(line.split(b";", 1)[0].strip(), 16)
        except ValueError:
            raise BadRequest(f"a malformed chunk size {line!r}") from None
        if size == 0:
            while (trailer := await reader.readuntil(b"\r\n")) != b"\r\n":
                writer.write(trailer)
            writer.write(trailer)
            await writer.drain()
            return
        await relay_exact(reader, writer, size + 2)


async def relay_until_eof(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    while data := await reader.read(CHUNK):
        writer.write(data)
        await writer.drain()


async def relay_body(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, headers: list[tuple[str, str]],
                     *, to_eof: bool) -> None:
    if "chunked" in tokens(headers, "transfer-encoding"):
        await relay_chunked(reader, writer)
    elif (length := header(headers, "content-length")) is not None:
        try:
            await relay_exact(reader, writer, int(length))
        except ValueError:
            raise BadRequest(f"a malformed Content-Length {length!r}") from None
    elif to_eof:
        await relay_until_eof(reader, writer)


async def splice(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """One direction of an upgraded connection, closing the other side when it ends, which ends the other direction."""
    try:
        await relay_until_eof(reader, writer)
    except (ConnectionError, OSError, ssl.SSLError):
        pass
    finally:
        writer.close()


class SiteProxy:
    """The proxy's HTTP side: one connection's requests, each forwarded on a connection of its own to its site's
    route."""

    def __init__(self, config: Config):
        self.config = config
        self.upstream_tls = ssl.create_default_context()

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, scheme: str) -> None:
        peer = (writer.get_extra_info("peername") or (LOOPBACK,))[0]
        try:
            while await self.exchange(reader, writer, scheme, peer):
                pass
        except BadRequest as e:
            writer.write(simple_response("400 Bad Request", f"{e}\n"))
        except (ConnectionError, asyncio.IncompleteReadError, ssl.SSLError, OSError):
            pass
        except Exception as e:
            print(f"site proxy: a {scheme} connection failed: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        finally:
            try:
                await writer.drain()
                writer.close()
                await writer.wait_closed()
            except (ConnectionError, OSError, ssl.SSLError):
                pass

    async def exchange(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, scheme: str,
                       peer: str) -> bool:
        """One request and its answer; whether the connection takes another."""
        head = await read_head(reader)
        if head is None:
            return False
        start, headers = parse_head(head)
        try:
            method, target, version = start.split(" ")
        except ValueError:
            raise BadRequest(f"a malformed request line {start!r}") from None
        host = header(headers, "host") or (urlsplit(target).netloc if "://" in target else "")
        site = self.config.site_for(split_host(host)) if host else None
        if site is None:
            writer.write(simple_response("421 Misdirected Request", f"{host or 'no host'} is no site of this "
                                         "computer\n", FAILED))
            return False
        if scheme == "http" and site.tls(split_host(host).lower()):
            location = f"https://{split_host(host)}{target if target.startswith('/') else urlsplit(target).path}"
            writer.write(simple_response("308 Permanent Redirect", f"{location}\n", (("Location", location),)))
            return False
        upgrade = header(headers, "upgrade") if "upgrade" in tokens(headers, "connection") else None
        if "100-continue" in tokens(headers, "expect"):
            writer.write(b"HTTP/1.1 100 Continue\r\n\r\n")
        try:
            up_reader, up_writer = await asyncio.wait_for(asyncio.open_connection(
                site.host, site.port, ssl=self.upstream_tls if site.scheme == "https" else None,
                server_hostname=site.host if site.scheme == "https" else None, limit=HEAD_LIMIT), UPSTREAM_TIMEOUT_S)
        except (OSError, asyncio.TimeoutError, ssl.SSLError) as e:
            writer.write(simple_response("502 Bad Gateway", f"the site {site.name} can't be reached at "
                                         f"{site.upstream}: {type(e).__name__}: {e}\n", FAILED))
            return False
        try:
            return await self._forward(reader, writer, up_reader, up_writer, method, target, version, headers, site,
                                       scheme=scheme, host=host, peer=peer, upgrade=upgrade)
        finally:
            up_writer.close()

    async def _forward(self, reader, writer, up_reader, up_writer, method, target, version, headers, site, *,
                       scheme, host, peer, upgrade) -> bool:
        up_writer.write(upstream_request(method, target, headers, site, scheme=scheme, host=host, peer=peer,
                                         upgrade=upgrade))
        await relay_body(reader, up_writer, headers, to_eof=False)
        await up_writer.drain()
        while True:
            answer = await read_head(up_reader)
            if answer is None:
                writer.write(simple_response("502 Bad Gateway", f"the site {site.name} closed without answering\n",
                                             FAILED))
                return False
            status_line, answer_headers = parse_head(answer)
            code = int(status_line.split(" ")[1])
            if code == 101 and upgrade:
                writer.write(response_head(status_line, answer_headers, site, scheme=scheme, host=host,
                                           connection="Upgrade"))
                await writer.drain()
                await asyncio.gather(splice(reader, up_writer), splice(up_reader, writer))
                return False
            if 100 <= code < 200:
                continue
            break
        bodiless = method == "HEAD" or code in (204, 304)
        framed = bodiless or "chunked" in tokens(answer_headers, "transfer-encoding") \
            or header(answer_headers, "content-length") is not None
        keep = framed and version == "HTTP/1.1" and "close" not in tokens(headers, "connection")
        writer.write(response_head(status_line, answer_headers, site, scheme=scheme, host=host,
                                   connection="keep-alive" if keep else "close"))
        if not bodiless:
            try:
                await relay_body(up_reader, writer, answer_headers, to_eof=True)
            except (BadRequest, asyncio.IncompleteReadError):
                return False
        await writer.drain()
        return keep


class Resolver(asyncio.DatagramProtocol):
    """DNS on loopback for a wildcard name, over UDP and over TCP, which a client retries a truncated answer on and
    ``options use-vc`` asks for: each question for a name of a site is answered with loopback, and every other is passed
    to the computer's own nameservers over the transport it came on, so the rest of the network is as it was; with none
    (a computer whose resolv.conf named no other), every other question fails at once rather than going unanswered."""

    def __init__(self, config: Config):
        self.config = config
        self.transport: Optional[asyncio.DatagramTransport] = None

    def connection_made(self, transport) -> None:
        self.transport = transport

    def datagram_received(self, data: bytes, addr) -> None:
        asyncio.ensure_future(self._answer(data, addr))

    async def _answer(self, data: bytes, addr) -> None:
        reply = await self.answer(data, stream=False)
        if reply and self.transport is not None:
            self.transport.sendto(reply, addr)

    async def handle_stream(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """A TCP connection's queries, each framed by its two-byte length, answered in turn until it closes."""
        try:
            while True:
                try:
                    (size,) = struct.unpack("!H", await reader.readexactly(2))
                except asyncio.IncompleteReadError:
                    return
                reply = await self.answer(await reader.readexactly(size), stream=True)
                if not reply:
                    return
                writer.write(struct.pack("!H", len(reply)) + reply)
                await writer.drain()
        except (ConnectionError, OSError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()

    async def answer(self, data: bytes, *, stream: bool) -> Optional[bytes]:
        question = parse_question(data)
        if question is not None and self.config.site_for(question[0]) is not None:
            return local_answer(data, question)
        if not self.config.nameservers:
            return failed_answer(data, question) if question is not None else None
        return await (self._passed_on_stream(data) if stream else self._passed_on(data))

    async def _passed_on(self, data: bytes) -> Optional[bytes]:
        loop = asyncio.get_running_loop()
        for nameserver in self.config.nameservers:
            family = socket.AF_INET6 if ":" in nameserver else socket.AF_INET
            with socket.socket(family, socket.SOCK_DGRAM) as sock:
                sock.setblocking(False)
                try:
                    await loop.sock_connect(sock, (nameserver, NAMESERVER_PORT))
                    await loop.sock_sendall(sock, data)
                    return await asyncio.wait_for(loop.sock_recv(sock, DATAGRAM_MAX), DNS_TIMEOUT_S)
                except (OSError, asyncio.TimeoutError):
                    continue
        return None

    async def _passed_on_stream(self, data: bytes) -> Optional[bytes]:
        for nameserver in self.config.nameservers:
            try:
                return await asyncio.wait_for(self._ask_stream(nameserver, data), DNS_TIMEOUT_S)
            except (OSError, asyncio.TimeoutError, asyncio.IncompleteReadError):
                continue
        return None

    @staticmethod
    async def _ask_stream(nameserver: str, data: bytes) -> bytes:
        reader, writer = await asyncio.open_connection(nameserver, NAMESERVER_PORT)
        try:
            writer.write(struct.pack("!H", len(data)) + data)
            await writer.drain()
            (size,) = struct.unpack("!H", await reader.readexactly(2))
            return await reader.readexactly(size)
        finally:
            writer.close()


def parse_question(data: bytes) -> Optional[tuple[str, int, int]]:
    """A query's one question, as (name, type, where it ends), or None for anything else."""
    if len(data) < 12:
        return None
    _, flags, questions = struct.unpack("!HHH", data[:6])
    if flags & 0x8000 or questions != 1:
        return None
    labels, at = [], 12
    while at < len(data) and (length := data[at]):
        if length & 0xC0:
            return None
        labels.append(data[at + 1:at + 1 + length].decode("ascii", "replace"))
        at += 1 + length
    if at + 5 > len(data):
        return None
    (qtype,) = struct.unpack("!H", data[at + 1:at + 3])
    return ".".join(labels).lower(), qtype, at + 5


def local_answer(query: bytes, question: tuple[str, int, int]) -> bytes:
    """The answer naming loopback for an A question about a site's name, and no records for any other type, so a
    client asking for IPv6 falls back to it."""
    _, qtype, end = question
    (ident, flags) = struct.unpack("!HH", query[:4])
    answers = 1 if qtype in (1, 255) else 0
    head = struct.pack("!HHHHHH", ident, 0x8480 | (flags & 0x0100), 1, answers, 0, 0)
    record = struct.pack("!HHHIH", 0xC00C, 1, 1, 60, 4) + socket.inet_aton(LOOPBACK) if answers else b""
    return head + query[12:end] + record


def failed_answer(query: bytes, question: tuple[str, int, int]) -> bytes:
    """The answer saying the server failed (SERVFAIL), for a question it has no nameserver to pass on to."""
    (ident, flags) = struct.unpack("!HH", query[:4])
    return struct.pack("!HHHHHH", ident, 0x8082 | (flags & 0x0100), 1, 0, 0, 0) + query[12:question[2]]


def tls_context(config: Config) -> Optional[ssl.SSLContext]:
    """The HTTPS side's context: each site's certificate, picked by the name the browser asks for."""
    contexts = {}
    for site in config.sites:
        if site.cert and site.key:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.minimum_version = ssl.TLSVersion.TLSv1_2
            context.set_alpn_protocols(["http/1.1"])
            context.load_cert_chain(site.cert, site.key)
            contexts[site.name] = context
    if not contexts:
        return None
    default = next(iter(contexts.values()))

    def pick(connection: ssl.SSLObject, name: Optional[str], _: ssl.SSLContext) -> None:
        site = config.site_for(name) if name else None
        if site is not None and site.name in contexts:
            connection.context = contexts[site.name]

    default.sni_callback = pick
    return default


async def serve(config: Config) -> None:
    """Listen, drop to the user it serves as, and serve until cancelled."""
    proxy = SiteProxy(config)
    servers = []
    if (context := tls_context(config)) is not None:
        servers.append(await asyncio.start_server(lambda r, w: proxy.handle(r, w, "https"), config.listen,
                                                  config.https_port, ssl=context, limit=HEAD_LIMIT))
    servers.append(await asyncio.start_server(lambda r, w: proxy.handle(r, w, "http"), config.listen,
                                              config.http_port, limit=HEAD_LIMIT))
    transport = None
    if config.dns_port:
        resolver = Resolver(config)
        transport, _ = await asyncio.get_running_loop().create_datagram_endpoint(
            lambda: resolver, local_addr=(config.listen, config.dns_port))
        servers.append(await asyncio.start_server(resolver.handle_stream, config.listen, config.dns_port))
    drop_privileges(config.serve_user or config.user)
    try:
        await asyncio.gather(*(server.serve_forever() for server in servers))
    finally:
        if transport is not None:
            transport.close()


def serve_user(session_user: Optional[str]) -> Optional[str]:
    """The user ``serve`` drops to: ``SERVE_USER`` where the base has it, else the session user."""
    try:
        pwd.getpwnam(SERVE_USER)
    except KeyError:
        return session_user
    return SERVE_USER


def drop_privileges(user: Optional[str]) -> None:
    if user is None or os.geteuid() != 0:
        return
    entry = pwd.getpwnam(user)
    os.setgroups([])
    os.setgid(entry.pw_gid)
    os.setuid(entry.pw_uid)


@dataclass(frozen=True)
class Paths:
    """Where ``install`` writes, under ``root``, by the conventions of Debian-family and Red Hat-family Linux."""

    root: Path = Path("/")

    @property
    def hosts(self) -> Path:
        return self.root / "etc/hosts"

    @property
    def resolv(self) -> Path:
        return self.root / "etc/resolv.conf"

    @property
    def anchors(self) -> tuple[tuple[Path, list[str]], ...]:
        """Each system store's anchor file, with the tool that rebuilds the store from its anchors."""
        return ((self.root / f"usr/local/share/ca-certificates/{CA_NAME}.crt", ["update-ca-certificates"]),
                (self.root / f"etc/pki/ca-trust/source/anchors/{CA_NAME}.pem", ["update-ca-trust", "extract"]))

    @property
    def firefox_policies(self) -> Path:
        return self.root / "etc/firefox/policies/policies.json"

    @property
    def chromium_policies(self) -> tuple[Path, ...]:
        """Where each Chromium build reads managed policies: Chrome for Testing, Chromium, a snap's Chromium and
        Chrome."""
        return tuple(self.root / folder / f"policies/managed/{CA_NAME}.json"
                     for folder in ("etc/opt/chrome_for_testing", "etc/chromium", "etc/chromium-browser",
                                    "etc/opt/chrome"))

    @property
    def profile(self) -> Path:
        return self.root / f"etc/profile.d/{CA_NAME}.sh"

    @property
    def system_bundle(self) -> Path:
        return self.root / "etc/ssl/certs/ca-certificates.crt"


def hosts_text(current: str, names: list[str]) -> str:
    """``current``, a hosts file, with its block of the sites' names replaced by one mapping ``names`` to loopback."""
    lines, inside = [], False
    for line in current.splitlines():
        if line.strip() == HOSTS_BEGIN:
            inside = True
        elif line.strip() == HOSTS_END:
            inside = False
        elif not inside:
            lines.append(line)
    if names:
        lines += [HOSTS_BEGIN, *(f"{LOOPBACK} {name}" for name in names), HOSTS_END]
    return "\n".join(lines) + "\n"


def resolv_text(current: str) -> tuple[str, list[str]]:
    """``current``, a resolv.conf, asking loopback first, and the nameservers it named, which loopback passes the rest
    to."""
    nameservers = [line.split()[1] for line in current.splitlines()
                   if line.split()[:1] == ["nameserver"] and len(line.split()) > 1]
    passed = [ns for ns in nameservers if ns != LOOPBACK]
    kept = [line for line in current.splitlines() if line.split()[:1] != ["nameserver"]]
    return "\n".join([f"nameserver {LOOPBACK}", *kept]) + "\n", passed


def firefox_policies(current: Optional[str], ca: str) -> str:
    """Firefox's policies file with ``ca`` among the certificates it installs, every other policy kept."""
    data = json.loads(current) if current and current.strip() else {}
    installs = data.setdefault("policies", {}).setdefault("Certificates", {}).setdefault("Install", [])
    if ca not in installs:
        installs.append(ca)
    return json.dumps(data, indent=2) + "\n"


def chromium_policies(ca_pem: str) -> str:
    """A Chromium managed policy trusting ``ca_pem`` (``CACertificates``, base64 DER, which is a PEM's body), which
    Chromium reads without ``certutil`` or an NSS database."""
    body = "".join(line.strip() for line in ca_pem.splitlines() if line.strip() and not line.startswith("-----"))
    return json.dumps({"CACertificates": [body]}, indent=1) + "\n"


def write_private(path: Path, text: str) -> None:
    """Write ``path`` readable by its owner alone, from its first byte."""
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as f:
        f.write(text)
    path.chmod(0o600)


def write_in_place(path: Path, text: str) -> None:
    """Rewrite ``path`` in place, which a file a container's runtime mounts (its hosts and resolv.conf) needs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def trust_system(paths: Paths, ca_pem: str) -> Optional[str]:
    """Add the CA to the system store, by the tool this Linux has; which store, or None when it has neither."""
    for anchor, tool in paths.anchors:
        if shutil.which(tool[0]):
            write_in_place(anchor, ca_pem)
            subprocess.run(tool, check=True, capture_output=True)
            return str(anchor)
    return None


def trust_nss(user: Optional[str], ca: str) -> list[str]:
    """Add the CA to the session user's NSS databases, Chromium's (and a snap Chromium's when there is one), as that
    user, for a Chromium too old to read its ``CACertificates`` policy; none without ``certutil``."""
    if user is None or not shutil.which("certutil"):
        return []
    home = Path(pwd.getpwnam(user).pw_dir)
    folders = [home / ".pki" / "nssdb"]
    if (home / "snap" / "chromium").is_dir():
        folders.append(home / "snap" / "chromium" / "current" / ".pki" / "nssdb")
    trusted = []
    for folder in folders:
        db = f"sql:{folder}"
        steps = [["mkdir", "-p", str(folder)]]
        if not (folder / "cert9.db").exists():
            steps.append(["certutil", "-N", "-d", db, "--empty-password"])
        steps.append(["certutil", "-A", "-d", db, "-n", NSS_NICKNAME, "-t", "C,,", "-i", ca])
        for step in steps:
            subprocess.run(as_user(user, step), check=True, capture_output=True)
        trusted.append(str(folder))
    return trusted


def as_user(user: Optional[str], argv: list[str]) -> list[str]:
    if user is None or os.geteuid() != 0:
        return argv
    return ["runuser", "-u", user, "--", *argv]


def install(config_path: str, paths: Paths = Paths()) -> dict:
    """As root: write the sites' certificates, names and trust, start ``serve`` on what's written, and check each name
    loads as the session user; what was trusted and each check's result. The config's folder, which arrives root's
    alone, is opened to the session user (for the CA) only once its keys and the config are private."""
    if os.geteuid() != 0 and paths.root == Path("/"):
        raise SystemExit(f"the site proxy installs as root, not uid {os.geteuid()}")
    given = json.loads(Path(config_path).read_text())
    folder = Path(config_path).parent
    user = given.get("user")
    ca = str(folder / "ca.pem")
    if given.get("ca_pem"):
        Path(ca).write_text(given["ca_pem"])
    sites = []
    for site in given["sites"]:
        served = {"name": site["name"], "upstream": site["upstream"], "domains": site["domains"]}
        if site.get("cert_pem"):
            cert, key = folder / f"{site['name']}.crt", folder / f"{site['name']}.key"
            cert.write_text(site["cert_pem"])
            write_private(key, site["key_pem"])
            served.update(cert=str(cert), key=str(key))
        sites.append(served)
    upstream_headers = {site["name"]: site["upstream_headers"] for site in given["sites"] if site.get("upstream_headers")}
    if upstream_headers:
        write_private(folder / UPSTREAM_HEADERS_FILE, json.dumps(upstream_headers))
    os.chmod(config_path, 0o600)
    folder.chmod(0o755)
    names = [name for site in given["sites"] for name in site["domains"] if not name.startswith("*.")]
    write_in_place(paths.hosts, hosts_text(paths.hosts.read_text() if paths.hosts.exists() else "", names))
    trust: dict = {}
    nameservers: list[str] = []
    wildcard = any(name.startswith("*.") for site in given["sites"] for name in site["domains"])
    if wildcard:
        text, nameservers = resolv_text(paths.resolv.read_text() if paths.resolv.exists() else "")
        write_in_place(paths.resolv, text)
    if given.get("ca_pem"):
        trust["system"] = trust_system(paths, given["ca_pem"])
        write_in_place(paths.firefox_policies, firefox_policies(
            paths.firefox_policies.read_text() if paths.firefox_policies.exists() else None, ca))
        trust["firefox"] = str(paths.firefox_policies)
        for policy in paths.chromium_policies:
            write_in_place(policy, chromium_policies(given["ca_pem"]))
        trust["chromium"] = [str(policy) for policy in paths.chromium_policies]
        trust["nss"] = trust_nss(user, ca)
        bundle = paths.system_bundle if paths.system_bundle.exists() else None
        write_in_place(paths.profile, f"export NODE_EXTRA_CA_CERTS={ca}\n"
                       + (f"export REQUESTS_CA_BUNDLE={bundle}\n" if bundle else ""))
        trust["environment"] = str(paths.profile)
    serving = {"user": user, "serve_user": serve_user(user), "sites": sites, **{key: given[key] for key in ("listen", "https_port", "http_port")
                                                 if key in given}}
    if wildcard:
        serving.update(dns_port=given.get("dns_port", 53), nameservers=nameservers)
    if upstream_headers:
        serving["upstream_headers"] = str(folder / UPSTREAM_HEADERS_FILE)
    serve_path = folder / "serve.json"
    serve_path.write_text(json.dumps(serving))
    start(str(serve_path), given.get("log"))
    return {"trust": trust, "names": names, "checks": checks_as(user, str(serve_path))}


def start(serve_path: str, log: Optional[str]) -> None:
    """Start ``keep``, which runs ``serve``, apart from this process, and wait until it listens. The log is opened
    without following a link, so a link planted at its path can't have root write elsewhere."""
    config = Config.load(serve_path)
    output = open_log(log) if log else subprocess.DEVNULL
    subprocess.Popen([sys.executable, os.path.abspath(__file__), "keep", serve_path], stdin=subprocess.DEVNULL,
                     stdout=output, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    port = config.https_port if any(site.cert for site in config.sites) else config.http_port
    deadline = time.monotonic() + LISTEN_TIMEOUT_S
    while True:
        try:
            socket.create_connection((config.listen, port), timeout=LISTEN_PROBE_S).close()
            return
        except OSError:
            if time.monotonic() > deadline:
                tail = Path(log).read_text()[-LOG_TAIL_CHARS:] if log and Path(log).exists() else ""
                raise SystemExit(f"the site proxy didn't listen on {config.listen}:{port} within "
                                 f"{LISTEN_TIMEOUT_S:.0f} s: {tail}")
            time.sleep(LISTEN_POLL_S)


def open_log(log: str) -> int:
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    return os.open(log, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)


def keep(serve_path: str) -> None:
    """As root: run ``serve`` on ``serve_path``, and start it again ``RESTART_DELAY_S`` after it ends, for as long as
    this runs."""
    while True:
        ended = subprocess.run([sys.executable, os.path.abspath(__file__), "serve", serve_path],
                               stdin=subprocess.DEVNULL).returncode
        print(f"the site proxy ended ({ended}); starting it again", flush=True)
        time.sleep(RESTART_DELAY_S)


def checks_as(user: Optional[str], serve_path: str) -> dict:
    """``check`` run as ``user``, given each site's timeout one after another and a margin for its start."""
    sites = len(Config.load(serve_path).sites)
    done = subprocess.run(as_user(user, [sys.executable, os.path.abspath(__file__), "check", serve_path]),
                          capture_output=True, text=True, timeout=CHECK_TIMEOUT_S * (sites + 2))
    if done.returncode != 0:
        return {"*": {"error": (done.stderr.strip().splitlines() or ["the check failed"])[-1]}}
    return json.loads(done.stdout)


def check(config: Config) -> dict:
    """Each site's first name loaded, over HTTPS but for a ``.localhost`` name, through the system's resolver and
    trust store: its status, or why it didn't load, which an answer the proxy made itself (it marks them ``FAILED``),
    such as its 502 for a gateway it couldn't reach, is too."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    results = {}
    for site in config.sites:
        name = next((n for n in site.domains if not n.startswith("*.")), None) \
            or "agentenv-check." + site.domains[0].removeprefix("*.")
        url = f"{'https' if site.tls(name) else 'http'}://{name}/"
        try:
            with opener.open(url, timeout=CHECK_TIMEOUT_S) as response:
                results[name] = {"url": url, "status": response.status}
        except urllib.error.HTTPError as e:
            if e.headers is not None and e.headers.get(FAILED[0][0]) == FAILED[0][1]:
                body = e.read().decode(errors="replace").strip()
                results[name] = {"url": url, "error": f"{e.code} {e.reason}: {body}"}
            else:
                results[name] = {"url": url, "status": e.code}
        except (urllib.error.URLError, OSError, ssl.SSLError) as e:
            results[name] = {"url": url, "error": str(getattr(e, "reason", e))}
    return results


def main(argv: list[str]) -> None:
    command, path = argv[0], argv[1]
    if command == "install":
        print(json.dumps(install(path)))
    elif command == "keep":
        keep(path)
    elif command == "serve":
        asyncio.run(serve(Config.load(path)))
    elif command == "check":
        print(json.dumps(check(Config.load(path))))
    else:
        raise SystemExit(f"usage: site_proxy.py install|keep|serve|check <config>, not {command!r}")


if __name__ == "__main__":
    main(sys.argv[1:])
