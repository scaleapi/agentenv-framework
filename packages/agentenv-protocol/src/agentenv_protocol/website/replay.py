"""Capture a real site and serve the capture as a website env.

``python -m agentenv_protocol.website.replay capture URL... --out capture.warc.gz`` fetches each start URL, then the
same-site pages it links to up to ``--depth`` links away and the assets each captured page loads (with what a
captured stylesheet's ``@import`` and ``url()`` name), and writes every
answer to a WARC 1.1 file, gzipped a record at a time as other WARC tools write them. Only GETs are captured, redirects
are kept as they were answered (and followed), and bodies are stored decoded: an answer's ``Content-Encoding`` and
``Transfer-Encoding`` are dropped and its ``Content-Length`` is the decoded body's. A host is the same site as a start
URL when it is that URL's host, without any ``www.``, or a subdomain of it. The request records name no header but
``Host``, so a capture holds no credential the capturing client sent.

``ReplayWebsite`` serves a capture for any host name: each request is matched by its method (HEAD as GET), its path and
its query as ``request_key`` normalises them, and a request the capture doesn't hold answers 404. A captured HTML
page's link and asset attributes and a stylesheet's ``url()`` and ``@import`` targets are served rewritten as a
``Location`` is, so they stay on the replay: a URL on a captured host is made root-relative, and a root-relative one is
led by the prefix the replay is served under; script bodies are served as captured. A captured
``Content-Security-Policy``, as a header or a page's ``<meta http-equiv>``, is not served, since the hashes and origins
it allows name the content as captured, not as rewritten. ``@api`` routes of a
subclass are layered on top and win over the capture; it keeps their state itself, loading and resetting it in
``add_site_data`` and ``reset_site_data`` and exporting it with a ``@get_data`` method of its own. ``data/add`` of a file part named ``*.warc`` or ``*.warc.gz`` (or typed ``application/warc``) serves that
capture instead of the declared one until ``data/reset``, the part's bytes, its ``file://`` file or what its https URL
(the signed URL a plugin's deployment sends) answers, up to ``MAX_CAPTURE_URL_BYTES``. Any capture read, declared or
added, is refused once it expands past ``MAX_CAPTURE_BYTES``. ``python -m agentenv_protocol.website.replay serve
capture.warc.gz --domain example.com`` serves one on ``PORT``.

The normalisation, for the capture and for each request alike: the method is upper-cased; the path is ``/`` when
empty and each of its segments is percent-decoded and encoded again the one way, so ``%7E`` and ``~`` match, while an
encoded ``/`` stays inside its segment; the query's parameters are sorted by name and then value, blank ones kept,
and those ``ignore_query`` names (the ``utm_*`` tracking parameters, ``fbclid``, ``gclid``, ``msclkid`` and the
cache-busting ``_`` by default) are dropped. The host, the fragment and every header are not part of the match. When a
capture holds one request twice, its first answer is served."""
from __future__ import annotations

import argparse
import asyncio
import base64
import dataclasses
import datetime
import gzip
import io
import re
import sys
import uuid
import zlib
from collections import deque
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, BinaryIO, ClassVar, Iterable, Iterator, Optional
from urllib.parse import parse_qsl, quote, unquote, urlencode, urljoin, urlsplit, urlunsplit

from starlette._utils import get_route_path
from starlette.applications import Starlette
from starlette.responses import Response
from starlette.routing import BaseRoute, Match, NoMatchFound, Route

from ..agent_env_environment import _maybe_await, add_data, reset_data
from ..types import FilePart
from .environment import SESSION_PATH, WebsiteEnvironment, website
from .errors import NotFound
from .routes import error_response, reserved_path

MAX_CAPTURE_URL_BYTES = 1 << 30
MAX_CAPTURE_BYTES = 1 << 30
CAPTURE_READ_CHUNK_BYTES = 1 << 20
CAPTURE_URL_TIMEOUT_SECONDS = 60
CAPTURE_MAX_REQUESTS = 500
CAPTURE_REQUEST_TIMEOUT_SECONDS = 20.0
DEFAULT_IGNORED_QUERY = ("utm_*", "fbclid", "gclid", "msclkid", "_")
WARC_VERSION = "WARC/1.1"
WARC_TYPES = ("application/warc", "application/warc+gzip")
_HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer",
               "transfer-encoding", "upgrade"}
_NOT_REPLAYED = _HOP_BY_HOP | {"content-length", "strict-transport-security", "alt-svc", "content-security-policy",
                               "content-security-policy-report-only"}
_NOT_STORED = _HOP_BY_HOP | {"content-encoding", "content-length"}
_SEGMENT_SAFE = "!$&'()*+,;=:@-._~"
_CSS_URL = re.compile(r"""url\(\s*(["']?)(?P<url>[^"')\s]*)\1\s*\)|@import\s+(["'])(?P<imported>[^"']*)\3""",
                      re.IGNORECASE)


def request_key(method: str, path: str, query: str = "", ignore_query: Iterable[str] = DEFAULT_IGNORED_QUERY,
                ) -> tuple[str, str, str]:
    """``(method, path, query)`` as a capture and a request are matched by them."""
    method = method.upper()
    if method == "HEAD":
        method = "GET"
    segments = (path or "/").split("/")
    path = "/".join(quote(unquote(segment), safe=_SEGMENT_SAFE) for segment in segments) or "/"
    if not path.startswith("/"):
        path = "/" + path
    ignored = tuple(ignore_query)
    params = sorted((k, v) for k, v in parse_qsl(query, keep_blank_values=True) if not _ignored(k, ignored))
    return method, path, urlencode(params, quote_via=quote)


def _ignored(name: str, ignored: tuple[str, ...]) -> bool:
    return any(name.startswith(rule[:-1]) if rule.endswith("*") else name == rule for rule in ignored)


@dataclasses.dataclass(frozen=True)
class Entry:
    """One captured answer: the URL it answered, its status, its headers in order, and its body."""

    method: str
    url: str
    status: int
    headers: tuple[tuple[str, str], ...]
    body: bytes


class Archive:
    """A capture's answers by ``request_key``, and the hosts they came from."""

    def __init__(self, entries: Iterable[Entry] = (), ignore_query: Iterable[str] = DEFAULT_IGNORED_QUERY) -> None:
        self.ignore_query = tuple(ignore_query)
        self.entries: dict[tuple[str, str, str], Entry] = {}
        self.hosts: set[str] = set()
        for entry in entries:
            self.add(entry)

    def __len__(self) -> int:
        return len(self.entries)

    def add(self, entry: Entry) -> bool:
        """Keep ``entry`` unless the archive holds an answer to its request already, recording its host either way, so
        links to every captured host stay on the replay; whether it was kept."""
        url = urlsplit(entry.url)
        if url.hostname:
            self.hosts.add(url.hostname.lower())
        key = request_key(entry.method, url.path, url.query, self.ignore_query)
        if key in self.entries:
            return False
        self.entries[key] = entry
        return True

    def find(self, method: str, path: str, query: str = "") -> Optional[Entry]:
        return self.entries.get(request_key(method, path, query, self.ignore_query))

    def holds(self, path: str) -> bool:
        """Whether any captured answer is for ``path``, whatever its method and query."""
        target = request_key("GET", path)[1]
        return any(key[1] == target for key in self.entries)

    @classmethod
    def read(cls, source: str | Path | bytes, ignore_query: Iterable[str] = DEFAULT_IGNORED_QUERY) -> Archive:
        """The answers of a WARC file, gzipped or not, given as its path or its bytes, refused once it expands past
        ``MAX_CAPTURE_BYTES``, or once its answers' bodies, decoded, together do."""
        if isinstance(source, bytes):
            raw = _expanded(io.BytesIO(source))
        else:
            with open(source, "rb") as file:
                raw = _expanded(file)
        return cls(_entries(io.BytesIO(raw)), ignore_query)


def _expanded(stream: BinaryIO) -> bytearray:
    """A WARC file's bytes, gunzipped when it's gzipped, read a chunk at a time and refused past
    ``MAX_CAPTURE_BYTES``, so a small file that expands a great deal never fills the memory."""
    gzipped = stream.read(2) == b"\x1f\x8b"
    stream.seek(0)
    reader: BinaryIO = gzip.GzipFile(fileobj=stream) if gzipped else stream
    data = bytearray()
    while chunk := reader.read(CAPTURE_READ_CHUNK_BYTES):
        data += chunk
        if len(data) > MAX_CAPTURE_BYTES:
            raise ValueError(f"the capture is over {MAX_CAPTURE_BYTES} bytes expanded")
    return data


def _records(stream: BinaryIO) -> Iterator[tuple[dict[str, str], bytes]]:
    """Each WARC record's headers, by lower-cased name, and its content block."""
    while True:
        line = stream.readline()
        if not line:
            return
        if not line.strip():
            continue
        if not line.startswith(b"WARC/"):
            raise ValueError(f"not a WARC record: {line[:40]!r}")
        headers: dict[str, str] = {}
        while (line := stream.readline()).strip():
            name, _, value = line.decode("utf-8").partition(":")
            headers[name.strip().lower()] = value.strip()
        block = stream.read(int(headers.get("content-length", "0")))
        yield headers, block


def _entries(stream: BinaryIO) -> Iterator[Entry]:
    """Each answer the WARC holds, its body decoded within what is left of ``MAX_CAPTURE_BYTES`` for all the bodies
    together, so many small compressed answers can't fill the memory between them."""
    methods: dict[str, str] = {}
    responses = []
    for headers, block in _records(stream):
        kind = headers.get("warc-type")
        if kind == "request":
            method = block.split(b" ", 1)[0].decode("latin-1").upper()
            if headers.get("warc-concurrent-to"):
                methods[headers["warc-concurrent-to"]] = method
        elif kind == "response" and "application/http" in headers.get("content-type", ""):
            responses.append((headers, block))
    left = MAX_CAPTURE_BYTES
    for headers, block in responses:
        status, entry_headers, body = _http_response(block, left)
        left -= len(body)
        if left < 0:
            raise ValueError(f"the capture's answers are over {MAX_CAPTURE_BYTES} bytes decoded")
        yield Entry(methods.get(headers.get("warc-record-id", ""), "GET"), headers["warc-target-uri"], status,
                    entry_headers, body)


def _http_response(block: bytes, limit: int) -> tuple[int, tuple[tuple[str, str], ...], bytes]:
    head, _, body = block.partition(b"\r\n\r\n")
    lines = head.decode("latin-1").split("\r\n")
    status = int(lines[0].split(" ", 2)[1])
    headers = tuple((name.strip(), value.strip()) for name, _, value in (line.partition(":") for line in lines[1:])
                    if name.strip())
    if any(n.lower() == "transfer-encoding" and "chunked" in v.lower() for n, v in headers):
        body = _dechunked(body)
        headers = tuple((n, v) for n, v in headers if n.lower() != "transfer-encoding")
    return status, *_decoded(headers, body, limit)


_DECODERS = {"gzip": zlib.MAX_WBITS | 16, "x-gzip": zlib.MAX_WBITS | 16, "deflate": zlib.MAX_WBITS}


def _decoded(headers: tuple[tuple[str, str], ...], body: bytes, limit: int) -> tuple[tuple[tuple[str, str], ...], bytes]:
    """``body`` with its ``Content-Encoding`` undone and that header dropped, as a capture of this module stores it, so
    a page imported from another WARC tool is rewritten too; a coding it can't undo (``br``, say) leaves both as they
    are, and a body that expands past ``limit`` bytes is refused, decoding no further than that."""
    codings = [c.strip().lower() for n, v in headers if n.lower() == "content-encoding" for c in v.split(",")]
    codings = [c for c in codings if c and c != "identity"]
    if not codings or any(c not in _DECODERS for c in codings):
        return headers, body
    for coding in reversed(codings):
        inflater = zlib.decompressobj(_DECODERS[coding])
        try:
            expanded = inflater.decompress(body, limit + 1)
        except zlib.error:
            if coding != "deflate":
                return headers, body
            inflater = zlib.decompressobj(-zlib.MAX_WBITS)
            expanded = inflater.decompress(body, limit + 1)
        if len(expanded) > limit or inflater.unconsumed_tail:
            raise ValueError(f"the capture's answers are over {MAX_CAPTURE_BYTES} bytes decoded")
        body = expanded
    return tuple((n, v) for n, v in headers if n.lower() not in ("content-encoding", "content-length")), body


def _dechunked(body: bytes) -> bytes:
    out, stream = bytearray(), io.BytesIO(body)
    while (size_line := stream.readline()) and (size := int(size_line.split(b";")[0].strip() or b"0", 16)):
        out += stream.read(size)
        stream.readline()
    return bytes(out)


class WarcWriter:
    """Writes WARC 1.1 records to a binary file, each its own gzip member when ``gzipped``."""

    def __init__(self, out: BinaryIO, gzipped: bool = True) -> None:
        self.out, self.gzipped = out, gzipped

    def warcinfo(self, fields: dict[str, str]) -> None:
        body = "".join(f"{k}: {v}\r\n" for k, v in fields.items()).encode()
        self._record("warcinfo", None, "application/warc-fields", body)

    def exchange(self, url: str, method: str, status: int, reason: str, headers: Iterable[tuple[str, str]],
                 body: bytes) -> None:
        """A response record of the answer, and a request record naming only its method, target and host."""
        response_id = f"<urn:uuid:{uuid.uuid4()}>"
        lines = [f"HTTP/1.1 {status} {reason or 'Unknown'}"]
        lines += [f"{name}: {value}" for name, value in headers if name.lower() not in _NOT_STORED]
        lines.append(f"Content-Length: {len(body)}")
        payload = ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1", "replace") + body
        self._record("response", url, "application/http; msgtype=response", payload, record_id=response_id)
        split = urlsplit(url)
        target = urlunsplit(("", "", split.path or "/", split.query, ""))
        request = f"{method} {target} HTTP/1.1\r\nHost: {split.netloc}\r\n\r\n".encode("latin-1", "replace")
        self._record("request", url, "application/http; msgtype=request", request,
                     extra={"WARC-Concurrent-To": response_id})

    def _record(self, kind: str, url: Optional[str], content_type: str, block: bytes, *,
                record_id: Optional[str] = None, extra: Optional[dict] = None) -> None:
        headers = {"WARC-Type": kind, "WARC-Record-ID": record_id or f"<urn:uuid:{uuid.uuid4()}>",
                   "WARC-Date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
        if url is not None:
            headers["WARC-Target-URI"] = url
        headers.update(extra or {})
        headers.update({"Content-Type": content_type, "Content-Length": str(len(block))})
        record = (WARC_VERSION + "\r\n" + "".join(f"{k}: {v}\r\n" for k, v in headers.items()) + "\r\n").encode()
        record += block + b"\r\n\r\n"
        self.out.write(gzip.compress(record) if self.gzipped else record)


class _Links(HTMLParser):
    """The links of an HTML page: ``pages`` its anchors and frames lead to, ``assets`` what it loads to show, a
    video's ``poster`` and the CSS URLs of its ``style`` values and ``<style>`` elements among them."""

    _ASSET_RELS = {"stylesheet", "icon", "shortcut", "preload", "modulepreload", "manifest", "apple-touch-icon"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.pages: list[str] = []
        self.assets: list[str] = []
        self.base: Optional[str] = None
        self.in_style = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        a = {k: v for k, v in attrs if v}
        self.in_style = tag == "style"
        if "style" in a:
            self._css(a["style"])
        if tag == "video" and "poster" in a:
            self.assets.append(a["poster"])
        if tag == "base" and "href" in a and self.base is None:
            self.base = a["href"]
        elif tag in ("a", "area") and "href" in a:
            self.pages.append(a["href"])
        elif tag in ("iframe", "frame") and "src" in a:
            self.pages.append(a["src"])
        elif tag == "link" and "href" in a:
            rels = set((a.get("rel") or "").lower().split())
            (self.assets if rels & self._ASSET_RELS else self.pages).append(a["href"])
        elif tag in ("script", "img", "source", "video", "audio", "track", "embed", "input") and "src" in a:
            self.assets.append(a["src"])
        if "srcset" in a:
            self.assets += [item.strip().split(" ")[0] for item in a["srcset"].split(",") if item.strip()]

    def handle_endtag(self, tag: str) -> None:
        self.in_style = False

    def handle_data(self, data: str) -> None:
        if self.in_style:
            self._css(data)

    def _css(self, css: str) -> None:
        self.assets += [href for match in _CSS_URL.finditer(css) if (href := match["url"] or match["imported"])]


def site_of(url: str) -> str:
    host = (urlsplit(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def same_site(url: str, sites: Iterable[str]) -> bool:
    host = (urlsplit(url).hostname or "").lower()
    return any(host == site or host.endswith("." + site) for site in sites)


@dataclasses.dataclass
class CaptureResult:
    captured: int = 0
    failed: list[tuple[str, str]] = dataclasses.field(default_factory=list)
    skipped: int = 0


def capture(urls: list[str], out: BinaryIO, *, depth: int = 1, max_requests: int = CAPTURE_MAX_REQUESTS,
            assets: bool = True, timeout: float = CAPTURE_REQUEST_TIMEOUT_SECONDS, client: Any = None,
            user_agent: str = "agentenv-replay-capture/1", gzipped: bool = True) -> CaptureResult:
    """Capture ``urls`` and what they lead to into ``out``, a WARC file; ``client`` is an ``httpx.Client`` to fetch
    with instead of one of its own, which must not follow redirects."""
    import httpx

    for url in urls:
        if urlsplit(url).scheme not in ("http", "https"):
            raise ValueError(f"a start URL is http or https, not {url!r}")
    sites = {site_of(url) for url in urls}
    writer = WarcWriter(out, gzipped)
    writer.warcinfo({"software": "agentenv_protocol.website.replay", "format": "WARC File Format 1.1",
                     "start-urls": " ".join(urls), "depth": str(depth)})
    own = client is None
    client = client or httpx.Client(follow_redirects=False, timeout=timeout, headers={"User-Agent": user_agent})
    queue: deque[tuple[str, int, bool]] = deque((url, 0, False) for url in urls)
    seen: set = set()
    result = CaptureResult()
    try:
        while queue and result.captured + len(result.failed) < max_requests:
            url, level, is_asset = queue.popleft()
            if (key := _fetched_as(url)) in seen:
                continue
            seen.add(key)
            try:
                response = client.get(url)
            except httpx.HTTPError as e:
                result.failed.append((url, f"{type(e).__name__}: {e}"))
                continue
            writer.exchange(url, "GET", response.status_code, response.reason_phrase, response.headers.multi_items(),
                            response.content)
            result.captured += 1
            for link, next_level, link_is_asset in _next(url, response, level, is_asset, assets):
                if next_level > depth or not same_site(link, sites):
                    result.skipped += next_level > depth and _fetched_as(link) not in seen
                    continue
                queue.append((link, next_level, link_is_asset))
    finally:
        if own:
            client.close()
    return result


def _fetched_as(url: str) -> tuple:
    split = urlsplit(url)
    return split.scheme, (split.hostname or "").lower(), split.port, *request_key("GET", split.path, split.query, ())


def _next(url: str, response: Any, level: int, is_asset: bool, assets: bool) -> Iterator[tuple[str, int, bool]]:
    """What a captured answer leads to: a redirect's target at its level, a stylesheet's ``@import`` and ``url()``
    targets as assets at its level, and an HTML page's links, its pages a level further and its assets at its level."""
    if 300 <= response.status_code < 400 and response.headers.get("location"):
        yield _absolute(url, response.headers["location"]), level, is_asset
        return
    if "css" in response.headers.get("content-type", ""):
        for match in _CSS_URL.finditer(response.text) if assets else ():
            href = match["url"] or match["imported"]
            if href and (target := _absolute(url, href)) is not None:
                yield target, level, True
        return
    if is_asset or "html" not in response.headers.get("content-type", ""):
        return
    links = _Links()
    try:
        links.feed(response.text)
    except Exception:
        return
    base = urljoin(url, links.base) if links.base else url
    for href in links.pages:
        if (target := _absolute(base, href)) is not None:
            yield target, level + 1, False
    for href in links.assets if assets else ():
        if (target := _absolute(base, href)) is not None:
            yield target, level, True


def _absolute(base: str, href: str) -> Optional[str]:
    target = urljoin(base, href.strip())
    split = urlsplit(target)
    if split.scheme not in ("http", "https"):
        return None
    return urlunsplit((split.scheme, split.netloc, split.path or "/", split.query, ""))


class ReplayWebsite(WebsiteEnvironment):
    """A site served from a capture: set ``capture`` to its WARC file (relative to the class's file) or pass one, and
    add ``@api`` routes to answer what the capture can't, which win over it. Declare it with ``@website(domains=...)``
    as any site; a static folder or ``live`` has no place on it. Its data plane serves captures: a subclass loads and
    resets the rest of its data in ``add_site_data`` and ``reset_site_data`` rather than with ``@add_data`` and
    ``@reset_data`` methods of its own."""

    capture: ClassVar[Optional[str]] = None
    ignore_query: ClassVar[tuple[str, ...]] = DEFAULT_IGNORED_QUERY

    def __init__(self, capture: str | Path | bytes | None = None) -> None:
        super().__init__()
        given = capture if capture is not None else type(self).capture
        source = self._site_path(given) if isinstance(given, (str, Path)) else given
        self.declared_archive = Archive.read(source, self.ignore_query) if source is not None else Archive()
        self.archive = self.declared_archive

    def create_app(self) -> Starlette:
        declaration = self.declaration()
        if declaration.static is not None or declaration.live:
            raise ValueError(f"{type(self).__name__} serves its capture, so it takes no static folder and isn't live")
        asgi = super().create_app()
        built_in = {SESSION_PATH: self._session}
        paths = {r.path for r in asgi.router.routes if isinstance(r, Route) and built_in.get(r.path) == r.endpoint}
        replay = _Replay(self, type(self).app)
        asgi.router.routes.insert(0, _Captured(self, paths, replay))
        asgi.router.default = replay
        return asgi

    def add_site_data(self, parts: list) -> Any:
        """Load ``parts``, those of a ``data/add`` that aren't a capture, into the site's own data; a site with none
        refuses them."""
        if parts:
            raise ValueError(f"{type(self).__name__} keeps no data but its capture; override add_site_data to load "
                             "the rest")

    def reset_site_data(self) -> Any:
        """Empty the site's own data, as ``data/reset`` does; a site with none has nothing to empty."""

    @add_data
    async def _replay_add(self, parts: list) -> None:
        """Serve a capture among ``parts`` instead of the declared one, read in a worker thread so the site goes on
        answering; the rest go to ``add_site_data``."""
        for part in parts:
            if _is_capture(part):
                self.archive = await asyncio.to_thread(Archive.read, await _capture_source(part), self.ignore_query)
        await _maybe_await(self.add_site_data([part for part in parts if not _is_capture(part)]))

    @reset_data
    async def _replay_reset(self) -> None:
        """Serve the declared capture again, and empty the site's own data with ``reset_site_data``."""
        await _maybe_await(self.reset_site_data())
        self.archive = self.declared_archive


def _is_capture(part: Any) -> bool:
    if not isinstance(part, FilePart):
        return False
    file = part.file
    name = (getattr(file, "name", None) or getattr(file, "uri", None) or "").lower()
    return getattr(file, "mimeType", None) in WARC_TYPES or name.endswith((".warc", ".warc.gz"))


async def _capture_source(part: FilePart) -> bytes | Path:
    file = part.file
    if getattr(file, "bytes", None) is not None:
        return base64.b64decode(file.bytes)
    uri = urlsplit(file.uri)
    if uri.scheme == "file":
        return Path(unquote(uri.path))
    if uri.scheme != "https":
        raise ValueError(f"a capture is loaded by its bytes, a file:// URI or an https URL, not a {uri.scheme}: URI")
    return await _fetched(file.uri)


async def _fetched(url: str) -> bytes:
    """What ``url`` answers, refused past ``MAX_CAPTURE_URL_BYTES``; an error names its status, never the URL, which
    may be signed."""
    import httpx

    data = bytearray()
    async with httpx.AsyncClient(follow_redirects=True, timeout=CAPTURE_URL_TIMEOUT_SECONDS) as client:
        async with client.stream("GET", url) as response:
            if response.status_code != 200:
                raise ValueError(f"the capture's URL answered {response.status_code}")
            async for chunk in response.aiter_bytes():
                data += chunk
                if len(data) > MAX_CAPTURE_URL_BYTES:
                    raise ValueError(f"the capture at the part's URL is over {MAX_CAPTURE_URL_BYTES} bytes")
    return bytes(data)


class _Captured(BaseRoute):
    """The built-in routes at ``paths``, answered from the capture instead for a path the archive serving now holds,
    so a capture loaded or reset later is matched as the one declared."""

    def __init__(self, env: ReplayWebsite, paths: set[str], replay: _Replay) -> None:
        self.env, self.paths, self.replay = env, paths, replay

    def matches(self, scope: dict) -> tuple[Match, dict]:
        path = get_route_path(scope)
        if scope["type"] == "http" and path in self.paths and self.env.archive.holds(path):
            return Match.FULL, {}
        return Match.NONE, {}

    def url_path_for(self, name: str, /, **path_params: Any) -> Any:
        raise NoMatchFound(name, path_params)

    async def handle(self, scope: dict, receive: Any, send: Any) -> None:
        await self.replay(scope, receive, send)


def _site_path(scope: dict) -> str:
    """The site's own path a request names, escaped as it was sent: its ``raw_path`` without the ``root_path`` a
    forwarded prefix leads it with, or the decoded path quoted again when the raw one doesn't spell it."""
    path = get_route_path(scope) or "/"
    raw = (scope.get("raw_path") or b"").decode("latin-1").split("?", 1)[0]
    root = scope.get("root_path", "")
    if root and raw:
        cut = next((i for i, char in enumerate(raw) if char == "/" and unquote(raw[:i]) == root), None)
        raw = raw[cut:] if cut is not None else ""
    return raw if raw and unquote(raw) == path else quote(path)


class _Replay:
    """What no route answers: the capture's answer to the request, then the mounted app, then 404."""

    def __init__(self, env: ReplayWebsite, app: Any) -> None:
        self.env, self.app = env, app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            if self.app is not None:
                await self.app(scope, receive, send)
            return
        path = _site_path(scope)
        if reserved_path(unquote(path)):
            return await error_response(NotFound("not found"))(scope, receive, send)
        entry = self.env.archive.find(scope["method"], path, scope.get("query_string", b"").decode("latin-1"))
        if entry is None:
            if self.app is not None:
                return await self.app(scope, receive, send)
            return await error_response(NotFound("not in the capture"))(scope, receive, send)
        await self._answer(entry, scope["method"] == "HEAD", scope.get("root_path", ""))(scope, receive, send)

    def _answer(self, entry: Entry, head: bool, root: str = "") -> Response:
        body = _rewritten_body(entry, self.env.archive.hosts, root)
        response = Response(b"" if head else body, status_code=entry.status)
        headers = [(name.lower().encode("latin-1"),
                    self._header(name.lower(), value, root).encode("latin-1", "replace"))
                   for name, value in entry.headers if name.lower() not in _NOT_REPLAYED]
        response.raw_headers = headers + [(b"content-length", str(len(body)).encode())]
        return response

    def _header(self, name: str, value: str, root: str = "") -> str:
        """``value`` as the replay answers it: a Set-Cookie without its Domain, so it holds at the replay's host, with
        its Path led by the ``root`` path the replay is served under, so it holds there for the paths it named; and a
        redirect to a captured host or a root-relative one led by ``root`` too, so it stays on the replay behind a
        gateway's prefix."""
        if name == "set-cookie":
            return ";".join(_cookie_attribute(p, root) for p in value.split(";")
                            if not p.strip().lower().startswith("domain="))
        if name == "location":
            return _rewritten_url(value, self.env.archive.hosts, root)
        return value


def _cookie_attribute(part: str, root: str) -> str:
    """A Set-Cookie attribute as the replay answers it: a ``Path`` that names a path of the site led by ``root``, so a
    cookie scoped to ``/account`` holds at ``<root>/account``; anything else as it is."""
    name, _, path = part.partition("=")
    root = root.rstrip("/")
    if not root or name.strip().lower() != "path" or not path.strip().startswith("/"):
        return part
    return f"{name}={root}{path.strip()}"


_ABSOLUTE_URL = re.compile(r"(?P<lead>\s*)(?:https?:)?//(?P<netloc>[^/?#\s]*)(?P<rest>.*)", re.IGNORECASE | re.DOTALL)
_TAG_NAME = re.compile(r"<[a-zA-Z][^\t\n\r\f />\x00]*(?:\s|/(?!>))*")
_ATTRIBUTE = re.compile(r"""(?P<name>[^\s/>][^\s/=>]*)(?:\s*=+\s*(?P<value>'[^']*'|"[^"]*"|(?!['"])[^>\s]*))?"""
                        r"""(?:\s|/(?!>))*""")
_URL_ATTRIBUTES = {"href", "src", "action", "formaction", "poster"}


def _rewritten_url(value: str, hosts: set[str], root: str) -> str:
    """``value`` as a link the replay serves: a URL on a captured host made root-relative, and a root-relative one led
    by the ``root`` path the replay is served under; any other URL is left as it is."""
    root = root.rstrip("/")
    if match := _ABSOLUTE_URL.fullmatch(value):
        try:
            host = urlsplit("//" + match["netloc"]).hostname
        except ValueError:
            return value
        if not host or host.lower() not in hosts:
            return value
        rest = match["rest"]
        return match["lead"] + root + (rest if rest.startswith("/") else "/" + rest)
    stripped = value.lstrip()
    if stripped.startswith("/"):
        return value[:len(value) - len(stripped)] + root + stripped
    return value


def _rewritten_css(css: str, hosts: set[str], root: str) -> str:
    """``css`` with each ``url()`` and ``@import`` target rewritten as ``_rewritten_url`` does, and nothing else."""
    def rewrite(match: re.Match) -> str:
        group = "url" if match["url"] is not None else "imported"
        start, end = match.start(group) - match.start(), match.end(group) - match.start()
        text = match[0]
        return text[:start] + _rewritten_url(match[group], hosts, root) + text[end:]
    return _CSS_URL.sub(rewrite, css)


def _rewritten_srcset(srcset: str, hosts: set[str], root: str) -> str:
    return re.sub(r"(^|,)(\s*)([^\s,]+)", lambda m: m[1] + m[2] + _rewritten_url(m[3], hosts, root), srcset)


class _Rewriter(HTMLParser):
    """Finds what to rewrite in an HTML page so its links stay on the replay: the ``href``, ``src``, ``srcset``,
    ``action``, ``formaction`` and ``poster`` values of each tag, its ``style`` value's and each ``<style>`` element's
    CSS URLs, as ``(start, end, text)`` spans of the page, and each ``<link>`` tag's ``integrity`` attribute dropped,
    since the stylesheet it names may be served rewritten and would fail its check, and each ``<meta http-equiv>``
    tag naming a ``Content-Security-Policy`` dropped whole, since its hashes and origins name the page as captured.
    Script bodies and every other attribute are left alone."""

    def __init__(self, page: str, hosts: set[str], root: str) -> None:
        super().__init__(convert_charrefs=False)
        self.page, self.hosts, self.root = page, hosts, root
        self.lines = [0] + [match.end() for match in re.finditer("\n", page)]
        self.spans: list[tuple[int, int, str]] = []
        self.in_style = False

    def _offset(self) -> int:
        line, column = self.getpos()
        return self.lines[line - 1] + column

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        self.in_style = tag == "style"
        text, start = self.get_starttag_text() or "", self._offset()
        if self.page[start:start + len(text)] != text or not (name := _TAG_NAME.match(text)):
            return
        equiv = (dict(attrs).get("http-equiv") or "").strip().lower()
        if tag == "meta" and equiv.startswith("content-security-policy"):
            self.spans.append((start, start + len(text), ""))
            return
        position = name.end()
        while position < len(text) and (match := _ATTRIBUTE.match(text, position)) and match.end() > position:
            position = match.end()
            value = match["value"]
            attribute = match["name"].lower()
            if tag == "link" and attribute == "integrity":
                self.spans.append((start + match.start(), start + match.end(), ""))
                continue
            if not value:
                continue
            inner, cut = (value[1:-1], 1) if value[0] in "'\"" else (value, 0)
            if attribute in _URL_ATTRIBUTES:
                changed = _rewritten_url(inner, self.hosts, self.root)
            elif attribute == "srcset":
                changed = _rewritten_srcset(inner, self.hosts, self.root)
            elif attribute == "style":
                changed = _rewritten_css(inner, self.hosts, self.root)
            else:
                continue
            if changed != inner:
                at = start + match.start("value") + cut
                self.spans.append((at, at + len(inner), changed))

    def handle_endtag(self, tag: str) -> None:
        self.in_style = False

    def handle_data(self, data: str) -> None:
        if self.in_style and (changed := _rewritten_css(data, self.hosts, self.root)) != data:
            start = self._offset()
            if self.page[start:start + len(data)] == data:
                self.spans.append((start, start + len(data), changed))


def _rewritten_body(entry: Entry, hosts: set[str], root: str) -> bytes:
    """``entry``'s body as the replay serves it: an HTML page's or a stylesheet's links rewritten by
    ``_rewritten_url``, every other byte as captured; a page the parser can't read is served as captured."""
    kind = next((value.lower() for name, value in entry.headers if name.lower() == "content-type"), "")
    if "html" not in kind and "css" not in kind:
        return entry.body
    page = entry.body.decode("latin-1")
    if "css" in kind:
        return _rewritten_css(page, hosts, root).encode("latin-1")
    rewriter = _Rewriter(page, hosts, root)
    try:
        rewriter.feed(page)
        rewriter.close()
    except Exception:
        return entry.body
    out, last = [], 0
    for start, end, text in sorted(rewriter.spans):
        if start >= last:
            out += [page[last:start], text]
            last = end
    return "".join(out + [page[last:]]).encode("latin-1")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agentenv_protocol.website.replay",
                                     description="Capture a site to a WARC file, or serve a capture.")
    commands = parser.add_subparsers(dest="command", required=True)
    grab = commands.add_parser("capture", help="capture start URLs and the same-site links they lead to")
    grab.add_argument("urls", nargs="+", metavar="URL")
    grab.add_argument("--out", required=True, help="the WARC file to write; gzipped unless it ends in .warc")
    grab.add_argument("--depth", type=int, default=1, help="how many links away from a start URL to follow (1)")
    grab.add_argument("--max-requests", type=int, default=CAPTURE_MAX_REQUESTS,
                      help=f"stop after this many requests ({CAPTURE_MAX_REQUESTS})")
    grab.add_argument("--no-assets", action="store_true", help="capture pages only, not what they load")
    grab.add_argument("--timeout", type=float, default=CAPTURE_REQUEST_TIMEOUT_SECONDS,
                      help=f"seconds one request may take ({CAPTURE_REQUEST_TIMEOUT_SECONDS:g})")
    serve = commands.add_parser("serve", help="serve a capture on PORT as a website/v1 site")
    serve.add_argument("capture")
    serve.add_argument("--domain", action="append", default=[], help="a domain the site is declared at")
    args = parser.parse_args(argv)
    if args.command == "capture":
        with open(args.out, "wb") as out:
            result = capture(args.urls, out, depth=args.depth, max_requests=args.max_requests,
                             assets=not args.no_assets, timeout=args.timeout, gzipped=not args.out.endswith(".warc"))
        for url, why in result.failed:
            print(f"failed  {url}: {why}", file=sys.stderr)
        print(f"captured {result.captured} answers to {args.out}; {len(result.failed)} failed, "
              f"{result.skipped} links beyond the depth")
        return 1 if result.captured == 0 else 0
    site = website(domains=args.domain)(type("Replay", (ReplayWebsite,), {}))
    site(Path(args.capture).resolve()).serve()
    return 0


if __name__ == "__main__":
    sys.exit(main())
