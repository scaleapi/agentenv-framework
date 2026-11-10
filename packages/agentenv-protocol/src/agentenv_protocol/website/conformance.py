"""Check a running site's server against ``urn:agentenv:website/v1``, whatever it's written in: ``python -m
agentenv_protocol.website.conformance <url>`` prints each check and exits 1 when a required one failed (or, with
``--strict``, when a recommended one warned), 2 when the server couldn't be reached. Everything goes over HTTP.

The required checks are the contract, what agentenv, the gateway and a browser rely on of a website env's server: a
card declaring the site's domains, readiness and routes; ``/`` serving HTML under a made-up host; the site ready, by
the card's ``website/v1`` ``ready``, whose answer must be a ``WebsiteReadiness`` (by the declared health path,
deprecated, for a card without it); the data plane's and card's paths not the site's pages; and the data plane the
card advertises, whose round trip at the card's ``url`` (data/get, data/reset, data/get, which no longer returns the
exported data, data/add of what the first data/get returned, data/get again) returns what it was given; and, for a
card declaring an ``mcp`` interface, a site that is an MCP server too, that endpoint answering ``initialize`` and
listing, every page of ``tools/list`` read, no tool its card doesn't (streamable HTTP, ``MCP_PROTOCOL_VERSION``); a
card's tool it doesn't list is warned, since a hidden-by-default tool is listed to no caller until one enables it.
A failed one is ``failed``. The recommended checks are what an SDK site does and the gateway uses when a server offers it:
``X-Universe-Now`` on every answer the other checks got, ``clock/v1``'s ``sync_time`` against a clock the checker serves, the forwarded headers
wherever an answer shows them (on ``/`` and on a GET route with its trailing slash toggled, which a server may
redirect), and the cookies those answers set: none tied to a domain, the session path's the one the card names, and
none whose lifetime is an absolute ``Expires`` with no ``Max-Age``. A failed one is ``warned``.

Every request goes to the server under check: a card's path or URL naming another scheme, host or port fails the check
that would have used it, with nothing sent there. The round trip leaves the server's data as it found it, adding the
export back when a step after the reset fails too. The clock check ends by syncing the server back to wall time (an
empty ``env_get_time_url``), since the checker's clock is gone once the check ends. That clock starts at
``CLOCK_START``, far enough from wall time that a synced server shows it, and a synced server's ``X-Universe-Now``
may be ``CLOCK_SLACK_S`` seconds from it."""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime
import json
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Iterator, Literal

import httpx
from pydantic import ValidationError

from ..client import find_extension, find_extension_method
from ..types import MCP_TRANSPORT, METHOD_ADD, METHOD_GET, METHOD_RESET, RPC_PATH, WELL_KNOWN_PATH
from .clock import CLOCK_EXTENSION_URI, UNIVERSE_NOW_HEADER, iso, parse_time
from .readiness import DEFAULT_HEALTH_PATH, WEBSITE_EXTENSION_URI, WEBSITE_READY_OP, WebsiteReadiness

Status = Literal["passed", "failed", "skipped", "warned"]
CLOCK_START = datetime.datetime(2031, 5, 6, 7, 8, 9, tzinfo=datetime.timezone.utc)
CLOCK_SLACK_S = 120
MCP_PROTOCOL_VERSION = "2025-06-18"
MCP_LIST_PAGES = 100


@dataclasses.dataclass(frozen=True)
class Result:
    """One check's outcome; ``required`` for a check of the contract, whose failure is ``failed``, and not for a
    recommended one, whose failure is ``warned``."""

    name: str
    status: Status
    detail: str = ""
    required: bool = True

    @property
    def passed(self) -> bool:
        return self.status != "failed"

    def fails(self, strict: bool = False) -> bool:
        """Whether the result fails the run: a failed check, or with ``strict`` a warned one too."""
        return self.status == "failed" or (strict and self.status == "warned")


def check(base_url: str, *, client: httpx.Client | None = None, clock_host: str = "127.0.0.1",
          timeout: float = 30) -> list[Result]:
    """Each check's result against the server at ``base_url``. ``clock_host`` is the address the server reaches the
    checker's clock at. Raises httpx's error when the server can't be reached at all."""
    own = client is None
    client = client or httpx.Client(timeout=timeout)
    try:
        return _Checker(base_url.rstrip("/"), client, clock_host).run()
    finally:
        if own:
            client.close()


class _Checker:
    def __init__(self, base_url: str, client: httpx.Client, clock_host: str) -> None:
        self.base_url, self.client, self.clock_host = base_url, client, clock_host
        self.rpc_url = base_url + RPC_PATH
        self.responses: list[httpx.Response] = []

    def get(self, path: str, **kwargs) -> httpx.Response:
        return self._seen(self.client.get(self._url(path), **kwargs))

    def _url(self, where: str) -> str:
        """Where ``where``, a path or a full URL a card names, is; raises ``OffServer`` for one on another server."""
        url = where if "://" in where else self.base_url + (where if where.startswith("/") else "/" + where)
        if _origin(url) != _origin(self.base_url):
            raise OffServer(f"the card points outside the server under check, at {url}")
        return url

    def _seen(self, response: httpx.Response) -> httpx.Response:
        self.responses.append(response)
        return response

    def run(self) -> list[Result]:
        response = self.get(WELL_KNOWN_PATH)
        try:
            card = response.json() if response.status_code == 200 else {}
        except ValueError:
            card = {}
        card = card if isinstance(card, dict) else {}
        site = (find_extension(card, WEBSITE_EXTENSION_URI) or {}).get("params") or {}
        health = site.get("health") if isinstance(site.get("health"), str) else DEFAULT_HEALTH_PATH
        # (name, required, check): a recommended check's failure is warned.
        steps = (("card", True, lambda: self._card(response, card, site)), ("any_host", True, self._any_host),
                 ("health", True, lambda: self._health(card, health)), ("universe_now", False, self._universe_now),
                 ("reserved_paths", True, self._reserved_paths), ("data_plane", True, lambda: self._data_plane(card)),
                 ("mcp", True, lambda: self._mcp(card)),
                 ("clock", False, lambda: self._clock(card)), ("forwarded", False, lambda: self._forwarded(site)),
                 ("cookies", False, lambda: self._cookies(site)))
        outcomes = {name: _outcome(step) for name, _required, step in steps if name != "universe_now"}
        outcomes["universe_now"] = _outcome(self._universe_now)
        results = []
        for name, required, _step in steps:
            status, detail = outcomes[name]
            results.append(Result(name, "warned" if status == "failed" and not required else status, detail, required))
        return results

    def _card(self, response: httpx.Response, card: dict, site: dict) -> tuple[Status, str]:
        if response.status_code != 200:
            return "failed", f"{WELL_KNOWN_PATH} answered {response.status_code}"
        problems = []
        if not site:
            problems.append(f"declares no {WEBSITE_EXTENSION_URI} with params")
        if not isinstance(site.get("domains"), list):
            problems.append("its params have no domains list")
        if find_extension_method(card, WEBSITE_EXTENSION_URI, WEBSITE_READY_OP) is None and not (
                isinstance(site.get("health"), str) and site["health"].startswith("/")):
            problems.append(f"it has no {WEBSITE_READY_OP}, and its params no health path")
        routes = site.get("routes")
        if not isinstance(routes, list) or not all(isinstance(r, dict) and {"method", "path", "name"} <= set(r)
                                                   for r in routes):
            problems.append("its params' routes aren't a list of {method, path, name}")
        return ("failed" if problems else "passed"), "; ".join(problems)

    def _any_host(self) -> tuple[Status, str]:
        host = f"conformance-{uuid.uuid4().hex[:8]}.invalid"
        response = self.get("/", headers={"Host": host})
        kind = response.headers.get("content-type", "")
        if response.status_code == 200 and kind.startswith("text/html"):
            return "passed", f"/ served HTML for Host {host}"
        return "failed", f"/ for Host {host} answered {response.status_code} {kind or 'without a content type'}"

    def _health(self, card: dict, health: str) -> tuple[Status, str]:
        ready = find_extension_method(card, WEBSITE_EXTENSION_URI, WEBSITE_READY_OP)
        if ready is None:
            response = self.get(health)
            return (("passed" if response.status_code == 200 else "failed"),
                    f"{health} answered {response.status_code}; the card has no website/v1 ready, so the health path "
                    "was read, which is deprecated")
        response = self._seen(self.client.request(ready.get("method", "GET"), self._url(ready["endpoint"])))
        try:
            answer = response.json()
        except ValueError:
            return "failed", f"website/v1 ready answered {response.status_code} with no JSON: {response.text[:200]!r}"
        try:
            readiness = WebsiteReadiness.model_validate(answer)
        except ValidationError as e:
            return "failed", (f"website/v1 ready answered {answer!r}, not a WebsiteReadiness: "
                              f"{'; '.join(_field_error(error) for error in e.errors())}")
        if not response.is_success:
            return "failed", (f"website/v1 ready answered {response.status_code}: "
                              f"{', '.join(readiness.problems) or 'no problem named'}")
        if readiness.ready:
            return "passed", "website/v1 ready says it is ready"
        return "failed", f"website/v1 ready says it is not ready: {', '.join(readiness.problems) or 'no problem named'}"

    def _universe_now(self) -> tuple[Status, str]:
        missing = [str(r.url.path) for r in self.responses if UNIVERSE_NOW_HEADER not in r.headers]
        if missing:
            return "failed", f"no {UNIVERSE_NOW_HEADER} on {', '.join(missing)}"
        for response in self.responses:
            parse_time(response.headers[UNIVERSE_NOW_HEADER])
        return "passed", f"on all {len(self.responses)} answers"

    def _reserved_paths(self) -> tuple[Status, str]:
        paths = [RPC_PATH, WELL_KNOWN_PATH, f"{RPC_PATH}/conformance-{uuid.uuid4().hex[:8]}", f"{RPC_PATH}/",
                 f"{WELL_KNOWN_PATH}/"]
        for path in paths:
            response = self.get(path)
            if response.is_success and response.headers.get("content-type", "").startswith("text/html"):
                return "failed", f"{path} answered a page, but the gateway keeps {RPC_PATH} and {WELL_KNOWN_PATH}"
        return "passed", f"no page at {', '.join(paths)}"

    def _rpc(self, method: str, params: dict | None = None) -> dict:
        response = self._seen(self.client.post(self.rpc_url, json={"jsonrpc": "2.0", "id": method,
                                                                               "method": method, "params": params or {}}))
        body = response.json()
        if isinstance(body, dict) and "error" in body:
            raise RuntimeError(f"{method} answered {body['error']}")
        if not response.is_success or not isinstance(body, dict) or "result" not in body:
            raise RuntimeError(f"{method} answered {response.status_code} with no JSON-RPC result: {response.text[:200]}")
        return body["result"] or {}

    def _data_plane(self, card: dict) -> tuple[Status, str]:
        operations = (card.get("capabilities") or {}).get("operations")
        operations = operations if isinstance(operations, list) else []
        missing = [m for m in (METHOD_GET, METHOD_RESET, METHOD_ADD) if m not in operations]
        if missing:
            return "failed", f"the card's operations lack {', '.join(missing)}"
        url = card.get("url")
        self.rpc_url = self._url(url if isinstance(url, str) and url else RPC_PATH)
        exported = self._rpc(METHOD_GET).get("parts") or []
        self._rpc(METHOD_RESET)
        added_back = False
        try:
            cleared = self._rpc(METHOD_GET).get("parts") or []
            if _holds_data(exported) and cleared == exported:
                added_back = True
                return "failed", "data/get after data/reset returned the data it returned before"
            if not exported:
                added_back = True
                return "passed", "data/get, data/reset and data/get answered; data/get returned nothing to add back"
            added_back = True
            self._rpc(METHOD_ADD, {"parts": exported})
            again = self._rpc(METHOD_GET).get("parts") or []
        finally:
            if not added_back:
                with contextlib.suppress(Exception):
                    self._rpc(METHOD_ADD, {"parts": exported})
        if again != exported:
            return "failed", "data/get after data/reset and data/add of its own export returned other data"
        return "passed", f"{len(exported)} part(s) round-tripped through reset and add"

    def _mcp(self, card: dict) -> tuple[Status, str]:
        """The MCP endpoint the card declares answers ``initialize`` and lists no tool the card doesn't, reading up to
        ``MCP_LIST_PAGES`` pages of ``tools/list``; its answers needn't carry ``X-Universe-Now``, which the MCP
        transport doesn't promise."""
        interfaces = [i for i in card.get("additionalInterfaces") or [] if isinstance(i, dict)
                      and i.get("transport") == MCP_TRANSPORT]
        if not interfaces:
            return "skipped", "the card declares no mcp interface"
        path = interfaces[0].get("url")
        if not isinstance(path, str) or not path:
            return "failed", f"the card's mcp interface has no url: {path!r}"
        url = self._url(path)
        headers = {"Accept": "application/json, text/event-stream"}
        first = self.client.post(url, headers=headers, json={
            "jsonrpc": "2.0", "id": "initialize", "method": "initialize",
            "params": {"protocolVersion": MCP_PROTOCOL_VERSION, "capabilities": {},
                       "clientInfo": {"name": "agentenv-website-conformance", "version": "1"}}})
        server = _mcp_result(first, "initialize", "initialize").get("serverInfo") or {}
        if session := first.headers.get("mcp-session-id"):
            headers["Mcp-Session-Id"] = session
        try:
            self.client.post(url, headers=headers, json={"jsonrpc": "2.0", "method": "notifications/initialized"})
            served: list = []
            cursor = None
            for _page in range(MCP_LIST_PAGES):
                listed = _mcp_result(self.client.post(url, headers=headers, json={
                    "jsonrpc": "2.0", "id": "tools/list", "method": "tools/list",
                    **({"params": {"cursor": cursor}} if cursor else {})}), "tools/list", "tools/list")
                if not isinstance(listed, dict) or not isinstance(listed.get("tools"), list):
                    return "failed", f"tools/list at {path} answered no tools list: {str(listed)[:200]}"
                served += [t.get("name") for t in listed["tools"] if isinstance(t, dict)]
                cursor = listed.get("nextCursor")
                if not cursor:
                    break
            else:
                return "failed", f"tools/list at {path} still had a next page after {MCP_LIST_PAGES}"
        finally:
            if session:
                with contextlib.suppress(httpx.HTTPError):
                    self.client.delete(url, headers=headers)
        advertised = {t.get("name") for t in (card.get("capabilities") or {}).get("tools") or [] if isinstance(t, dict)}
        if unadvertised := sorted(set(served) - advertised):
            return "failed", f"tools/list at {path} named {unadvertised}, which the card doesn't list"
        name = server.get("name") or "the server"
        if unlisted := sorted(advertised - set(served)):
            return "warned", (f"{name} at {path} lists {len(served)} of the card's tools, not {unlisted}: hidden by "
                              "default, or missing")
        return "passed", f"{name} at {path} lists the card's {len(served)} tool(s)"

    def _clock(self, card: dict) -> tuple[Status, str]:
        if find_extension(card, CLOCK_EXTENSION_URI) is None:
            return "failed", f"the card declares no {CLOCK_EXTENSION_URI}"
        method = find_extension_method(card, CLOCK_EXTENSION_URI, "sync_time")
        if method is None or not method.get("endpoint"):
            return "failed", f"{CLOCK_EXTENSION_URI} advertises no sync_time endpoint"
        endpoint = self._url(method["endpoint"])
        with _clock_server(self.clock_host) as (url, virtual_now):
            response = self._seen(self.client.request(method["method"], endpoint, json={"env_get_time_url": url}))
            if response.status_code >= 400:
                return "failed", f"sync_time answered {response.status_code}: {response.text[:200]}"
            try:
                seen = self.get("/").headers.get(UNIVERSE_NOW_HEADER)
                expected = virtual_now()
            finally:
                with contextlib.suppress(Exception):
                    self.client.request(method["method"], endpoint, json={"env_get_time_url": ""})
        if seen is None:
            return "failed", f"no {UNIVERSE_NOW_HEADER} after sync_time"
        off = abs((parse_time(seen) - expected).total_seconds())
        if off > CLOCK_SLACK_S:
            return "failed", f"after sync_time {UNIVERSE_NOW_HEADER} is {seen}, {off:.0f}s from the clock's {iso(expected)}"
        return "passed", f"{UNIVERSE_NOW_HEADER} followed the clock to {seen}"

    def _forwarded(self, site: dict) -> tuple[Status, str]:
        internal = f"internal-{uuid.uuid4().hex[:8]}.invalid"
        forwarded = f"site-{uuid.uuid4().hex[:8]}.example"
        headers = {"Host": internal, "X-Forwarded-Host": forwarded, "X-Forwarded-Proto": "https"}
        paths = ["/", *_slash_toggled(site.get("routes"))]
        for path in paths:
            response = self.get(path, headers=headers, follow_redirects=False)
            location = response.headers.get("location", "")
            shown = [where for where, text in (("body", response.text), ("Location", location)) if internal in text]
            if shown:
                return "failed", f"the {' and '.join(shown)} of {path} named the internal Host {internal}, not {forwarded}"
            if location.startswith(f"http://{forwarded}"):
                return "failed", f"the Location of {path} is {location}, not https as X-Forwarded-Proto said"
        return "passed", f"no answer to {', '.join(paths)} named the internal Host"

    def _cookies(self, site: dict) -> tuple[Status, str]:
        session = []
        if isinstance(site.get("session_path"), str):
            session = self.get(site["session_path"]).headers.get_list("set-cookie")
        cookies = [c for r in self.responses for c in r.headers.get_list("set-cookie")]
        if not cookies:
            return "skipped", "no answer set a cookie"
        scoped = sorted({c.split("=", 1)[0] for c in cookies if "domain" in _attributes(c)})
        if scoped:
            return "failed", f"cookies tied to a domain: {', '.join(scoped)}"
        named = site.get("session_cookie")
        if isinstance(named, str) and session and named not in {c.split("=", 1)[0].strip() for c in session}:
            return "failed", (f"{site['session_path']} set {', '.join(c.split('=', 1)[0] for c in session)}, not the "
                              f"card's {named}")
        absolute = sorted({c.split("=", 1)[0] for c in cookies if "expires" in _attributes(c) and "max-age" not in _attributes(c)})
        if absolute:
            return "warned", (f"cookies with an absolute Expires and no Max-Age, which a browser on another clock "
                              f"misjudges: {', '.join(absolute)}")
        return "passed", f"{len(cookies)} cookie(s), none tied to a domain"


def _attributes(set_cookie: str) -> set[str]:
    """The attribute names of a Set-Cookie header, lowercased."""
    return {a.split("=", 1)[0].strip().lower() for a in set_cookie.split(";")[1:]}


class OffServer(ValueError):
    """A card named a path or URL on a server other than the one under check."""


def _origin(url: str) -> tuple[str, str, int | None]:
    """A URL's scheme, host and port, the port its scheme's default when it names none."""
    parsed = httpx.URL(url)
    return parsed.scheme, parsed.host, parsed.port or {"http": 80, "https": 443}.get(parsed.scheme)


def _outcome(step: Callable[[], tuple[Status, str]]) -> tuple[Status, str]:
    try:
        return step()
    except Exception as e:
        return "failed", f"{type(e).__name__}: {e}"


def _mcp_result(response: httpx.Response, method: str, request_id: str) -> dict:
    """The result of the JSON-RPC reply to request ``request_id``: the answer's JSON, or the reply among an event
    stream's messages, each event's data lines joined, as ``ToolSession`` reads them."""
    if response.headers.get("content-type", "").startswith("text/event-stream"):
        messages = []
        for data in _sse_data(response.text):
            try:
                messages.append(json.loads(data))
            except ValueError:
                raise RuntimeError(f"{method} answered an event that isn't JSON: {data[:200]!r}") from None
    else:
        try:
            messages = [json.loads(response.text)]
        except ValueError:
            raise RuntimeError(f"{method} answered {response.status_code} with no JSON-RPC answer: "
                               f"{response.text[:200]!r}") from None
    body = next((m for m in messages if isinstance(m, dict) and m.get("id") == request_id
                 and ("result" in m or "error" in m)), None)
    if body is None or "result" not in body:
        raise RuntimeError(f"{method} answered {response.status_code}: {str(body or messages)[:200]}")
    return body["result"] or {}


def _sse_data(text: str) -> Iterator[str]:
    """The data of each ``message`` event in a server-sent event stream, its data lines joined; an event the stream
    ends inside is dropped, as are events with no data."""
    event, data = "message", []
    for line in text.splitlines():
        if not line:
            payload = "\n".join(data)
            if payload.strip() and event == "message":
                yield payload
            event, data = "message", []
            continue
        field, _, value = line.partition(":")
        value = value[1:] if value.startswith(" ") else value
        if field == "data":
            data.append(value)
        elif field == "event":
            event = value


def _field_error(error: dict) -> str:
    """One of a ``ValidationError``'s errors as a person reads it: where, then what."""
    where = ".".join(map(str, error.get("loc") or ())) or "the answer"
    return f"{where}: {error.get('msg')}"


def _holds_data(parts: list) -> bool:
    """Whether exported parts carry task data: a part that isn't a data part, or a data part with a value in it."""
    return any(not (isinstance(p, dict) and p.get("kind") == "data") or _filled(p.get("data")) for p in parts)


def _filled(value: object) -> bool:
    if isinstance(value, (dict, list)):
        return any(_filled(v) for v in (value.values() if isinstance(value, dict) else value))
    return value not in (None, "")


def _slash_toggled(routes: object) -> list[str]:
    """The first GET route without path parameters, with its trailing slash added or taken off: a path a server
    may redirect to the route's own, which shows the address the redirect names."""
    for route in routes if isinstance(routes, list) else []:
        path = route.get("path") if isinstance(route, dict) and route.get("method") == "GET" else None
        if isinstance(path, str) and path.startswith("/") and path != "/" and "{" not in path:
            return [path[:-1] if path.endswith("/") else path + "/"]
    return []


@contextlib.contextmanager
def _clock_server(host: str) -> Iterator[tuple[str, Callable[[], datetime.datetime]]]:
    """A gateway-shaped ``/clock/time`` running from ``CLOCK_START`` at real speed, and its time now."""
    started = time.monotonic()

    def virtual_now() -> datetime.datetime:
        return CLOCK_START + datetime.timedelta(seconds=time.monotonic() - started)

    class Clock(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            body = json.dumps({"virtual_time": iso(virtual_now())}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args) -> None:
            return None

    server = ThreadingHTTPServer(("127.0.0.1" if host in ("127.0.0.1", "localhost") else "0.0.0.0", 0), Clock)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        address = f"[{host}]" if ":" in host else host
        yield f"http://{address}:{server.server_port}/clock/time", virtual_now
    finally:
        server.shutdown()
        server.server_close()


_MARK = {"passed": "PASS", "failed": "FAIL", "skipped": "----", "warned": "WARN"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agentenv_protocol.website.conformance",
                                     description="Check a running site's server against urn:agentenv:website/v1.")
    parser.add_argument("url", help="where the server answers, such as http://localhost:8000")
    parser.add_argument("--clock-host", default="127.0.0.1", help="the address the server reaches this checker at")
    parser.add_argument("--timeout", type=float, default=30, help="seconds for each request")
    parser.add_argument("--strict", action="store_true", help="fail on a recommended check's warning too")
    args = parser.parse_args(argv)
    try:
        results = check(args.url, clock_host=args.clock_host, timeout=args.timeout)
    except httpx.TransportError as e:
        print(f"couldn't reach {args.url}: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    for result in results:
        kind = "required" if result.required else "recommended"
        print(f"{_MARK[result.status]} {result.name} ({kind}){': ' + result.detail if result.detail else ''}")
    counts = {status: sum(r.status == status for r in results) for status in _MARK}
    print(f"{args.url}: {counts['passed']} passed, {counts['failed']} failed, {counts['warned']} warned, "
          f"{counts['skipped']} skipped{' (strict)' if args.strict else ''}")
    return 1 if any(r.fails(args.strict) for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
