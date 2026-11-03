"""Check a running site's server against ``urn:agentenv:website/v1``, whatever it's written in: ``python -m
agentenv_protocol.website.conformance <url>`` prints each check and exits 1 when a required one failed (or, with
``--strict``, when a recommended one warned), 2 when the server couldn't be reached. Everything goes over HTTP.

The required checks are the contract, what the gateway and a browser rely on of any server: ``/`` serves HTML under a
made-up host, the site is ready, by the card's ``website/v1`` ``ready``, whose answer must be a
``WebsiteReadiness``, or, for a card without it, by the health path (the card's, else ``/api/health``) answering 200, which is deprecated, the data plane and card's paths are not
the site's pages, and, when the card advertises the data plane, a round trip at the card's ``url`` (data/get,
data/reset, data/get, which no longer returns the exported data, data/add of what the first data/get returned, data/get
again) returns what it was given. A failed one is ``failed``. The recommended checks are what an SDK site does and the
gateway uses when a server offers it: a card declaring the site's domains, health path and routes, ``X-Universe-Now``
on every answer the other checks got, ``clock/v1``'s ``sync_time`` against a clock the checker serves, the forwarded
headers wherever an answer shows them (on ``/`` and on a GET route with its trailing slash toggled, which a server may
redirect), and cookies tied to no domain, the session path's the one the card names. A failed one is a ``warning``.

With ``--spec``, a website spec, the server is checked against it too, and those checks are required: its card's name,
domains, health path, routes and session against the spec's, and every route the spec declares served (an ``OPTIONS``
that finds something at its path, and a GET route without path parameters answering neither 404 nor 405), each in the
server's OpenAPI document when its card offers one; for a card without one, a route only the ``OPTIONS`` can check is
unverified, a ``warning``, unless that ``OPTIONS`` answers with an ``Allow`` listing its method. The health check
then reads the spec's health path for a card without ``website/v1`` ``ready``.

The round trip leaves the server's data as it found it, adding the export back when a step after the reset fails too.
The clock check leaves the server following a clock that is gone once the check ends, so a server it ran against
keeps extrapolating from that clock's time until it's synced again. The checker's clock starts at ``CLOCK_START``, far
enough from wall time that a synced server shows it, and a synced server's ``X-Universe-Now`` may be ``CLOCK_SLACK_S``
seconds from it."""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime
import json
import re
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Iterator, Literal

import httpx
from pydantic import ValidationError

from ..client import find_extension, find_extension_method
from ..types import METHOD_ADD, METHOD_GET, METHOD_RESET, RPC_PATH, WELL_KNOWN_PATH
from .clock import CLOCK_EXTENSION_URI, UNIVERSE_NOW_HEADER, iso, parse_time
from .environment import DEFAULT_HEALTH_PATH
from .readiness import WEBSITE_EXTENSION_URI, WEBSITE_READY_OP, WebsiteReadiness
from .spec import SpecError, WebsiteSpec, load

Status = Literal["passed", "failed", "warning", "skipped"]
CLOCK_START = datetime.datetime(2031, 5, 6, 7, 8, 9, tzinfo=datetime.timezone.utc)
CLOCK_SLACK_S = 120
REQUIRED = ("any_host", "health", "reserved_paths", "data_plane", "spec", "spec_routes")


@dataclasses.dataclass(frozen=True)
class Result:
    """One check's outcome; ``required`` for a check of the contract, whose failure is ``failed``, and not for a
    recommended one, whose failure is a ``warning``."""

    name: str
    status: Status
    detail: str = ""
    required: bool = True

    @property
    def passed(self) -> bool:
        return self.status != "failed"

    def fails(self, strict: bool = False) -> bool:
        """Whether the result fails the run: a failed check, or with ``strict`` a warning too."""
        return self.status == "failed" or (strict and self.status == "warning")


def check(base_url: str, *, client: httpx.Client | None = None, clock_host: str = "127.0.0.1",
          timeout: float = 30, spec: WebsiteSpec | None = None) -> list[Result]:
    """Each check's result against the server at ``base_url``. ``clock_host`` is the address the server reaches the
    checker's clock at; with ``spec``, the server is checked against it too. Raises httpx's error when the server
    can't be reached at all."""
    own = client is None
    client = client or httpx.Client(timeout=timeout)
    try:
        return _Checker(base_url.rstrip("/"), client, clock_host, spec).run()
    finally:
        if own:
            client.close()


class _Checker:
    def __init__(self, base_url: str, client: httpx.Client, clock_host: str, spec: WebsiteSpec | None = None) -> None:
        self.base_url, self.client, self.clock_host, self.spec = base_url, client, clock_host, spec
        self.rpc_url = base_url + RPC_PATH
        self.responses: list[httpx.Response] = []

    def get(self, path: str, **kwargs) -> httpx.Response:
        return self._seen(self.client.get(self.base_url + path, **kwargs))

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
        self.rpc_url = _rpc_url(self.base_url, card)
        site = (find_extension(card, WEBSITE_EXTENSION_URI) or {}).get("params") or {}
        health = site.get("health") if isinstance(site.get("health"), str) else DEFAULT_HEALTH_PATH
        if self.spec is not None:
            health = self.spec.health
        steps = [("card", lambda: self._card(response, site)), ("any_host", self._any_host),
                 ("health", lambda: self._health(card, health)), ("universe_now", self._universe_now),
                 ("reserved_paths", self._reserved_paths), ("data_plane", lambda: self._data_plane(card)),
                 ("clock", lambda: self._clock(card, health)), ("forwarded", lambda: self._forwarded(site)),
                 ("cookies", lambda: self._cookies(site))]
        if self.spec is not None:
            steps += [("spec", lambda: self._spec_card(card, site)), ("spec_routes", lambda: self._spec_routes(card, site))]
        outcomes = {name: _outcome(step) for name, step in steps if name != "universe_now"}
        outcomes["universe_now"] = _outcome(self._universe_now)
        results = []
        for name, _step in steps:
            status, detail = outcomes[name]
            required = name in REQUIRED
            results.append(Result(name, "warning" if status == "failed" and not required else status, detail, required))
        return results

    def _card(self, response: httpx.Response, site: dict) -> tuple[Status, str]:
        if response.status_code != 200:
            return "failed", f"{WELL_KNOWN_PATH} answered {response.status_code}"
        problems = []
        if not site:
            problems.append(f"declares no {WEBSITE_EXTENSION_URI} with params")
        if not isinstance(site.get("domains"), list):
            problems.append("its params have no domains list")
        if not (isinstance(site.get("health"), str) and site["health"].startswith("/")):
            problems.append("its params have no health path")
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
        response = self._seen(self.client.request(ready.get("method", "GET"), self.base_url + ready["endpoint"]))
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
        paths = [RPC_PATH, WELL_KNOWN_PATH, f"{RPC_PATH}/conformance-{uuid.uuid4().hex[:8]}", f"{RPC_PATH}/", f"{WELL_KNOWN_PATH}/"]
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
        if len(missing) == 3:
            return "skipped", "the card advertises no data plane"
        if missing:
            return "failed", f"the card's operations lack {', '.join(missing)}"
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

    def _clock(self, card: dict, health: str) -> tuple[Status, str]:
        if find_extension(card, CLOCK_EXTENSION_URI) is None:
            return "failed", f"the card declares no {CLOCK_EXTENSION_URI}"
        method = find_extension_method(card, CLOCK_EXTENSION_URI, "sync_time")
        if method is None or not method.get("endpoint"):
            return "failed", f"{CLOCK_EXTENSION_URI} advertises no sync_time endpoint"
        with _clock_server(self.clock_host) as (url, virtual_now):
            response = self._seen(self.client.request(method["method"], self.base_url + method["endpoint"],
                                                      json={"env_get_time_url": url}))
            if response.status_code >= 400:
                return "failed", f"sync_time answered {response.status_code}: {response.text[:200]}"
            seen = self.get(health).headers.get(UNIVERSE_NOW_HEADER)
            expected = virtual_now()
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
        scoped = [c.split("=", 1)[0] for c in cookies if any(a.strip().lower().startswith("domain=") for a in c.split(";"))]
        if scoped:
            return "failed", f"cookies tied to a domain: {', '.join(sorted(set(scoped)))}"
        named = site.get("session_cookie")
        if isinstance(named, str) and session and named not in {c.split("=", 1)[0].strip() for c in session}:
            return "failed", f"{site['session_path']} set {', '.join(c.split('=', 1)[0] for c in session)}, not the card's {named}"
        return "passed", f"{len(cookies)} cookie(s), none tied to a domain"


    def _spec_card(self, card: dict, site: dict) -> tuple[Status, str]:
        """The card's name, domains, health path, routes and session cookie and path against the spec's, and its
        state and clock when the spec declares them."""
        spec, declared = self.spec, self.spec.to_card_extension()
        problems = []
        if spec.site.name is not None and card.get("name") != spec.site.name:
            problems.append(f"the card is named {card.get('name')!r}, the spec's site {spec.site.name!r}")
        if set(site.get("domains") or []) != set(declared["domains"]):
            problems.append(f"the card's domains are {site.get('domains')}, the spec's {declared['domains']}")
        if site.get("health") != declared["health"]:
            problems.append(f"the card's health is {site.get('health')!r}, the spec's {declared['health']!r}")
        theirs = {r.get("name"): _route_key(r) for r in site.get("routes") or [] if isinstance(r, dict)}
        for route in declared["routes"]:
            if route["name"] not in theirs:
                problems.append(f"the card lacks route {route['name']}")
            elif theirs[route["name"]] != _route_key(route):
                problems.append(f"the card's {route['name']} is {_shown(theirs[route['name']])}, the spec's "
                                f"{_shown(_route_key(route))}")
        if extra := sorted(set(theirs) - {r["name"] for r in declared["routes"]}):
            problems.append(f"the card has routes the spec doesn't declare: {', '.join(map(str, extra))}")
        for key in ("session_cookie", "session_path", "state", "clock"):
            if key in declared and site.get(key) != declared[key]:
                problems.append(f"the card's {key} is {site.get(key)!r}, the spec's {declared[key]!r}")
        if problems:
            return "failed", "; ".join(problems)
        return "passed", f"name, {len(declared['domains'])} domain(s), health and {len(declared['routes'])} route(s) match"

    def _spec_routes(self, card: dict, site: dict) -> tuple[Status, str]:
        """Every route the spec declares served: an OPTIONS finding something at its path, a GET route without path
        parameters answering neither 404 nor 405, and an operation of the server's OpenAPI document. A path
        parameter is probed with a value its converter on the card's route of that name takes (``_PROBES``), ``0``
        when the card gives it none. When the card offers no document, a route the OPTIONS alone can check is
        served only when the OPTIONS answers below 500 with an ``Allow`` listing its method; any other answer, such
        as a catch-all's 200 without ``Allow``, a 500, or an ``Allow`` naming only the methods of another route at
        that path, leaves it unverified, which, when nothing failed, is a ``warning``."""
        routes = self.spec.route_table()
        served = {r.get("name"): r.get("path") for r in site.get("routes") or []
                  if isinstance(r, dict) and isinstance(r.get("path"), str)}
        if not routes:
            return "skipped", "the spec declares no routes"
        problems, unverified = [], []
        offered = find_extension_method(card, WEBSITE_EXTENSION_URI, "openapi") is not None
        try:
            document = self._openapi(card)
        except RuntimeError as e:
            problems.append(str(e))
            document = None
        for name, route in routes.items():
            card_path = served.get(name)
            template = card_path if card_path and _CONVERTER.sub(r"{\1}", card_path) == route.path else route.path
            path = _PARAM.sub(_probe, template)
            options = self.client.options(self.base_url + path)
            if options.status_code == 404:
                problems.append(f"{name}: nothing is at {route.method} {route.path}")
                continue
            if route.method == "GET" and path == route.path:
                status = self.get(path).status_code
                if status in (404, 405) or status >= 500:
                    problems.append(f"{name}: GET {route.path} answered {status}")
            elif not offered:
                allowed = _allowed(options)
                if allowed is None or options.status_code >= 500 or route.method not in allowed:
                    shown = " without an Allow" if allowed is None else f" allowing {', '.join(sorted(allowed))}"
                    unverified.append(f"{name}: OPTIONS {path} answered {options.status_code}{shown}, which doesn't "
                                      f"show {route.method} {route.path} is served")
            if document is not None:
                operation = ((document.get("paths") or {}).get(route.path) or {}).get(route.method.lower())
                if not isinstance(operation, dict) or operation.get("operationId") != name:
                    problems.append(f"{name}: the OpenAPI document has no operation {name} at {route.method} {route.path}")
        if problems:
            return "failed", "; ".join(problems + unverified)
        if unverified:
            return "warning", "unverified: " + "; ".join(unverified)
        return "passed", f"all {len(routes)} route(s) served" + (", each in the OpenAPI document" if document else "")

    def _openapi(self, card: dict) -> dict | None:
        """The OpenAPI document the card's ``website/v1`` ``openapi`` serves, None when it offers none; raises
        RuntimeError when one it offers can't be read."""
        method = find_extension_method(card, WEBSITE_EXTENSION_URI, "openapi")
        if method is None or not method.get("endpoint"):
            return None
        response = self.client.request(method["method"], self.base_url + method["endpoint"],
                                       **({"json": {}} if method["method"] != "GET" else {}))
        try:
            document = response.json() if response.status_code == 200 else None
        except ValueError:
            document = None
        if not isinstance(document, dict):
            raise RuntimeError(f"the card's openapi, {method['method']} {method['endpoint']}, answered "
                               f"{response.status_code} and no OpenAPI document")
        return document


def _allowed(response: httpx.Response) -> set[str] | None:
    """The methods ``response``'s ``Allow`` lists, None when it has none."""
    allow = response.headers.get("allow")
    return None if allow is None else {m.strip().upper() for m in allow.split(",") if m.strip()}


_PARAM = re.compile(r"{[^}]+}")
_PROBES = {"uuid": "00000000-0000-0000-0000-000000000000"}


def _probe(param: re.Match) -> str:
    """A value the path parameter ``param`` matches: its converter's from ``_PROBES``, else ``0``, which an int,
    float, str or path parameter all take."""
    _, _, converter = param.group(0)[1:-1].partition(":")
    return _PROBES.get(converter, "0")


_CONVERTER = re.compile(r"{(\w+):[^}]+}")


def _route_key(route: dict) -> tuple:
    """A route as the card and the spec can both say it: its method, its path with any converter dropped
    (``{rest:path}`` is ``{rest}``), and its roles."""
    return route.get("method"), _CONVERTER.sub(r"{\1}", route.get("path") or ""), tuple(sorted(route.get("roles") or []))


def _shown(key: tuple) -> str:
    method, path, roles = key
    return f"{method} {path}" + (f" for {list(roles)}" if roles else "")


def _rpc_url(base_url: str, card: dict) -> str:
    """Where the card says its data plane is: its ``url``, a path joined onto ``base_url`` or a full URL, else
    ``RPC_PATH``."""
    url = card.get("url") if isinstance(card.get("url"), str) and card.get("url") else RPC_PATH
    return url if "://" in url else base_url + url


def _outcome(step: Callable[[], tuple[Status, str]]) -> tuple[Status, str]:
    try:
        return step()
    except Exception as e:
        return "failed", f"{type(e).__name__}: {e}"


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
            body = json.dumps({"virtual_time": iso(virtual_now()), "virtual_seconds_per_real_second": 1.0}).encode()
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


_MARK = {"passed": "PASS", "failed": "FAIL", "warning": "WARN", "skipped": "----"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agentenv_protocol.website.conformance",
                                     description="Check a running site's server against urn:agentenv:website/v1.")
    parser.add_argument("url", help="where the server answers, such as http://localhost:8000")
    parser.add_argument("--clock-host", default="127.0.0.1", help="the address the server reaches this checker at")
    parser.add_argument("--timeout", type=float, default=30, help="seconds for each request")
    parser.add_argument("--strict", action="store_true", help="fail on a recommended check's warning too")
    parser.add_argument("--spec", help="a website spec file to check the server against too")
    args = parser.parse_args(argv)
    try:
        spec = load(args.spec) if args.spec else None
    except SpecError as e:
        print("\n".join(e.problems), file=sys.stderr)
        return 2
    try:
        results = check(args.url, clock_host=args.clock_host, timeout=args.timeout, spec=spec)
    except httpx.TransportError as e:
        print(f"couldn't reach {args.url}: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    for result in results:
        kind = "required" if result.required else "recommended"
        print(f"{_MARK[result.status]} {result.name} ({kind}){': ' + result.detail if result.detail else ''}")
    counts = {status: sum(r.status == status for r in results) for status in _MARK}
    print(f"{args.url}: {counts['passed']} passed, {counts['failed']} failed, {counts['warning']} warning(s), "
          f"{counts['skipped']} skipped{' (strict)' if args.strict else ''}")
    return 1 if any(r.fails(args.strict) for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
