"""A website browser's network log, as the proxy it goes out through writes it and as ``browser/v1`` and ``data/get``
read it. That proxy, the shared site proxy or the base's stand-in, appends JSON lines to the file ``NETWORK_LOG_ENV``
names, each numbered (``seq``) and timed (``at``) as it is written: an ``open`` line once a request has gone upstream
or a tunnel has opened, and an ``end`` line, naming the same ``id``, once it is answered and done, or has failed.

``har`` makes HAR 1.2 of the lines: an entry for each request and each tunnel, in the order they started, with what it
ended with, or ``_open`` while it hasn't ended. A body arrives as the proxy saw it, base64 (``Body``), and leaves
decoded, up to ``DECODED_LIMIT``: gzip and deflate always, br and zstd when their libraries are installed."""
from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import zlib
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qsl, urlsplit

try:
    import brotli
except ImportError:
    brotli = None
try:
    import zstandard
except ImportError:
    zstandard = None

#: The environment variable naming the network log of the proxy the browser goes out through.
NETWORK_LOG_ENV = "AGENTENV_NETWORK_LOG"
#: The stand-in's, which its base names when no site routing names the shared proxy's instead.
DEFAULT_NETWORK_LOG = Path("/var/log/agentenv/network.jsonl")
#: The network log as ``data/get`` returns it, beside the browser's state; ``data/add`` passes over it.
HAR_FILE = "network.har"
#: How many lines, and how many of their bytes, one read returns at most.
READ_LIMIT = 500
READ_BYTES = 32 << 20
#: How much of an answer's body, and of a request's, an entry keeps once decoded; decoding stops there.
DECODED_LIMIT, REQUEST_DECODED_LIMIT = 1 << 20, 10 << 20
#: A ``charset`` in a ``Content-Type``.
CHARSET = re.compile(r"charset=\"?([\w.:-]+)", re.I)
#: What undoing a coding raises on a body that isn't what its coding says.
DECODE_ERRORS = (zlib.error, ValueError, *((brotli.error,) if brotli else ()),
                 *((zstandard.ZstdError,) if zstandard else ()))


def log_path() -> Optional[Path]:
    """The network log of the proxy this browser goes out through, or None for a browser none logs for, as on a
    computer with a desktop, which names none."""
    return Path(named) if (named := os.environ.get(NETWORK_LOG_ENV)) else None


class NetworkLog:
    """The network log in ``path``, read in the order its lines were numbered, which is the order they were written:
    where the last read ended is kept, so a poller reads only what came after."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._mark = (0, 0)

    def read(self, after: int = 0, limit: int = READ_LIMIT) -> dict:
        """The lines numbered after ``after``, up to ``limit`` of them and ``READ_BYTES`` of their text: ``lines``,
        ``last``, the number of the last one, and ``more`` when a limit left some. A line still being written is left
        for the next read, and one that isn't JSON or isn't numbered is passed over."""
        mark_seq, mark_offset = self._mark
        lines: list[dict] = []
        size, more = 0, False
        try:
            with self.path.open("rb") as file:
                fits = mark_offset <= os.fstat(file.fileno()).st_size
                file.seek(mark_offset if after >= mark_seq and fits else 0)
                while (raw := file.readline()).endswith(b"\n"):
                    try:
                        line = json.loads(raw)
                    except ValueError:
                        continue
                    seq = line.get("seq") if isinstance(line, dict) else None
                    if not isinstance(seq, int) or isinstance(seq, bool):
                        continue
                    if seq > after:
                        if len(lines) >= limit or (lines and size + len(raw) > READ_BYTES):
                            more = True
                            break
                        lines.append(line)
                        size += len(raw)
                    self._mark = (seq, file.tell())
        except FileNotFoundError:
            pass
        return {"lines": lines, "last": lines[-1]["seq"] if lines else after, "more": more}

    async def read_async(self, after: int = 0, limit: int = READ_LIMIT) -> dict:
        return await asyncio.to_thread(self.read, after, limit)

    def every_line(self) -> list[dict]:
        lines, after = [], 0
        while (page := self.read(after, limit=1 << 30))["lines"]:
            lines += page["lines"]
            after = page["last"]
            if not page["more"]:
                break
        return lines

    async def har_text(self) -> str:
        """Every line as HAR, as ``data/get`` writes it, made off the event loop."""
        return await asyncio.to_thread(lambda: json.dumps(har(self.every_line()), indent=1))


def har(lines: list[dict]) -> dict:
    """``lines`` as HAR 1.2: each request or tunnel one entry, its ``open`` and ``end`` lines joined by ``id``, the
    entries in the order they started, and those that started in the same millisecond in the order they were first
    written. A line that isn't one, or an entry that can't be read, is left out and counted in ``_unreadable``, so one
    bad line can't cost an export the rest."""
    opened: dict[str, Optional[dict]] = {}
    ended: dict[str, dict] = {}
    unreadable = 0
    for line in lines:
        kind = "open" if isinstance(line.get("open"), dict) else "end" if isinstance(line.get("end"), dict) else None
        key = line[kind].get("id") if kind else None
        if not isinstance(key, str):
            unreadable += 1
        elif kind == "open":
            if opened.get(key) is None:
                opened[key] = line["open"]
        else:
            ended.setdefault(key, line["end"])
            opened.setdefault(key, None)
    entries = []
    for key, record in opened.items():
        try:
            entries.append(_entry(record, ended.get(key)))
        except (KeyError, TypeError, ValueError, AttributeError, IndexError, OverflowError):
            unreadable += 1
    entries.sort(key=lambda entry: entry["startedDateTime"])
    log = {"version": "1.2", "creator": {"name": "agentenv", "version": _version()}, "entries": entries}
    return {"log": {**log, **({"_unreadable": unreadable} if unreadable else {})}}


def _entry(opened: Optional[dict], ended: Optional[dict]) -> dict:
    record = {**(ended or {}), **(opened or {})}
    finished = (ended or {}).get("ended")
    started = record.get("started") or finished or ""
    if not isinstance(started, str):
        raise TypeError("started")
    answered = (ended or {}).get("answered") or finished
    entry: dict[str, Any] = {"startedDateTime": started, "time": _ms(started, finished)}
    if record.get("kind") == "tunnel":
        entry.update(_tunnel(record, ended))
    else:
        request = record.get("request") or {"method": record.get("method", ""), "url": _url(record)}
        response = (ended or {}).get("response")
        entry["request"] = _request(request)
        entry["response"] = _response(response)
        if response and response.get("status") == 101:
            entry["request"]["url"] = re.sub(r"^http", "ws", entry["request"]["url"])
            entry["_resourceType"] = "websocket"
        if (ended or {}).get("websocket"):
            entry["_websocket"] = ended["websocket"]
        if record.get("address"):
            entry["serverIPAddress"] = record["address"]
    entry["cache"] = {}
    entry["timings"] = {"send": 0, "wait": _ms(started, answered), "receive": _ms(answered, finished)}
    for key in ("failed", "refused", "unparsed"):
        if (ended or {}).get(key):
            entry[f"_{key}"] = ended[key]
    if ended is None:
        entry["_open"] = True
    return entry


def _url(record: dict) -> str:
    """The URL a request the proxy refused, by its host alone, was for: https for a ``CONNECT``."""
    url = record.get("url", "")
    if not url or "://" in url:
        return url
    return f"{'https' if record.get('method') == 'CONNECT' else 'http'}://{url}/"


def _tunnel(record: dict, ended: Optional[dict]) -> dict:
    """A connection the proxy kept only at connection level: a CONNECT tunnel, answered 200 once it opened, or a plain
    request the base's stand-in forwarded, by its method and URL, its answer unread."""
    host, port = record.get("host", ""), record.get("port", 443)
    url = record.get("url") or f"https://{host}:{port}/"
    request = {"method": record.get("method", "CONNECT"), "url": url, "httpVersion": "HTTP/1.1", "cookies": [],
               "headers": [], "queryString": [{"name": name, "value": value}
                                             for name, value in parse_qsl(urlsplit(url).query, keep_blank_values=True)],
               "headersSize": -1, "bodySize": 0}
    failed, connect = (ended or {}).get("failed"), "url" not in record
    response = {"status": 200 if connect and not failed else 0,
                "statusText": "Connection established" if connect and not failed else "",
                "httpVersion": "HTTP/1.1", "cookies": [], "headers": [], "content": {"size": 0, "mimeType": ""},
                "redirectURL": "", "headersSize": -1, "bodySize": 0}
    fields = {"request": request, "response": response,
              "_tunnel": {key: record[key] for key in ("host", "port", "address", "bytes_sent", "bytes_received")
                          if key in record}}
    if record.get("address"):
        fields["serverIPAddress"] = record["address"]
    return fields


def _request(request: dict) -> dict:
    headers = request.get("headers") or []
    url = request.get("url", "")
    body = request.get("body")
    har_request = {"method": request.get("method", ""), "url": url,
                   "httpVersion": request.get("http_version", "HTTP/1.1"), "cookies": _cookies(headers),
                   "headers": _headers(headers),
                   "queryString": [{"name": name, "value": value}
                                   for name, value in parse_qsl(urlsplit(url).query, keep_blank_values=True)],
                   "headersSize": -1, "bodySize": body["size"] if body else 0}
    if body and body.get("size"):
        text, extra, _ = _text(body, _header(headers, "content-encoding"), _header(headers, "content-type"),
                               REQUEST_DECODED_LIMIT)
        har_request["postData"] = {"mimeType": _header(headers, "content-type") or "", "text": text, **extra}
    return har_request


def _response(response: Optional[dict]) -> dict:
    if response is None:
        return {"status": 0, "statusText": "", "httpVersion": "HTTP/1.1", "cookies": [], "headers": [],
                "content": {"size": 0, "mimeType": ""}, "redirectURL": "", "headersSize": -1, "bodySize": 0}
    headers = response.get("headers") or []
    status, mime = response.get("status", 0), _header(headers, "content-type") or ""
    body = response.get("body")
    content: dict[str, Any] = {"size": body["size"] if body else 0, "mimeType": mime}
    if body and "base64" in body:
        text, extra, content["size"] = _text(body, _header(headers, "content-encoding"), mime, DECODED_LIMIT)
        content.update(text=text, **extra)
    elif body:
        content["_sha256"] = body.get("sha256")
    return {"status": status, "statusText": response.get("reason", ""),
            "httpVersion": response.get("http_version", "HTTP/1.1"), "cookies": _set_cookies(headers),
            "headers": _headers(headers), "content": content,
            "redirectURL": (_header(headers, "location") or "") if 300 <= status < 400 else "",
            "headersSize": -1, "bodySize": body["size"] if body else 0}


def _text(body: dict, coding: Optional[str], mime: Optional[str], limit: int) -> tuple[str, dict, int]:
    """A body's text, decoded from ``coding`` and its charset, up to ``limit``; what a HAR says beside it: that the
    proxy or the limit cut it, how much compression saved, or, for a coding that can't be undone here, the raw body as
    base64 with the coding named; and its size as HAR's ``content.size`` has it, decoded, which is as far as decoding
    went when a cut stopped it (with ``_truncated``), and the size sent when nothing was undone."""
    raw = base64.b64decode(body.get("base64") or "")
    size = body.get("size", len(raw))
    extra: dict[str, Any] = {}
    if body.get("truncated"):
        extra["_truncated"] = True
    if coding and coding.strip().lower() not in ("", "identity"):
        decoded = _decompressed(raw, coding, limit)
        if decoded is None:
            return base64.b64encode(raw).decode(), {**extra, "encoding": "base64", "_undecoded": coding}, size
        if len(decoded) <= limit and not body.get("truncated"):
            extra["compression"] = max(0, len(decoded) - size)
        raw, size = decoded, len(decoded)
    if len(raw) > limit:
        raw, extra["_truncated"] = raw[:limit], True
    charset = (CHARSET.search(mime or "") or [None, "utf-8"])[1]
    try:
        return raw.decode(charset, errors="replace"), extra, size
    except LookupError:
        return raw.decode("utf-8", errors="replace"), extra, size


def _decompressed(raw: bytes, coding: str, limit: int) -> Optional[bytes]:
    """``raw`` with each of ``coding``'s codings undone, last first, as far as it goes, so a body the proxy cut decodes
    up to the cut, and each stopping past ``limit``, so a small body can't expand without bound; None for a coding
    this can't undo."""
    data, most = raw, limit + 1
    for name in reversed([part.strip().lower() for part in coding.split(",") if part.strip()]):
        try:
            if name in ("gzip", "x-gzip"):
                data = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(data, most)
            elif name == "deflate":
                try:
                    data = zlib.decompressobj(zlib.MAX_WBITS).decompress(data, most)
                except zlib.error:
                    data = zlib.decompressobj(-zlib.MAX_WBITS).decompress(data, most)
            elif name == "br" and brotli is not None:
                data = brotli.Decompressor().process(data, output_buffer_limit=most)[:most]
            elif name == "zstd" and zstandard is not None:
                data = _zstd(data, most)
            elif name != "identity":
                return None
        except DECODE_ERRORS:
            return None
    return data


def _zstd(data: bytes, most: int) -> bytes:
    out = bytearray()
    with zstandard.ZstdDecompressor().stream_reader(data) as reader:
        while len(out) < most and (chunk := reader.read(most - len(out))):
            out += chunk
    return bytes(out)


def _headers(headers: list) -> list[dict]:
    return [{"name": name, "value": value} for name, value in headers]


def _header(headers: list, name: str) -> Optional[str]:
    return next((value for key, value in headers if key.lower() == name), None)


def _cookies(headers: list) -> list[dict]:
    """The cookies a request sent, from its ``Cookie`` headers."""
    cookies = []
    for key, value in headers:
        if key.lower() == "cookie":
            for pair in value.split(";"):
                name, equals, cookie = pair.strip().partition("=")
                if name and equals:
                    cookies.append({"name": name, "value": cookie})
    return cookies


def _set_cookies(headers: list) -> list[dict]:
    """The cookies an answer set, from its ``Set-Cookie`` headers, with the attributes HAR names."""
    cookies = []
    for key, value in headers:
        if key.lower() != "set-cookie":
            continue
        first, *attributes = value.split(";")
        name, _, cookie_value = first.strip().partition("=")
        cookie: dict[str, Any] = {"name": name, "value": cookie_value, "httpOnly": False, "secure": False}
        for attribute in attributes:
            attr, _, attr_value = attribute.strip().partition("=")
            lower = attr.lower()
            if lower in ("path", "domain"):
                cookie[lower] = attr_value
            elif lower == "expires" and (expires := _http_date(attr_value)):
                cookie["expires"] = expires
            elif lower == "httponly":
                cookie["httpOnly"] = True
            elif lower == "secure":
                cookie["secure"] = True
            elif lower == "samesite":
                cookie["sameSite"] = attr_value
        cookies.append(cookie)
    return cookies


def _http_date(text: str) -> Optional[str]:
    """An HTTP date, as a cookie's ``Expires`` has it, in ISO 8601 as HAR wants; None for one that doesn't parse."""
    try:
        moment = parsedate_to_datetime(text)
        moment = moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)
        return moment.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    except (TypeError, ValueError, IndexError, OverflowError):
        return None


def _ms(start: Optional[str], end: Optional[str]) -> float:
    if not start or not end:
        return 0
    try:
        return max(0.0, round((_when(end) - _when(start)).total_seconds() * 1000, 3))
    except ValueError:
        return 0


def _when(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _version() -> str:
    try:
        return version("agentenv-framework-protocol")
    except PackageNotFoundError:
        return "unknown"
