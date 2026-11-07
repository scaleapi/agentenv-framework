"""The pieces of ``agent-env env website dev``, which runs a site for authoring without Docker: the site's ``web/``
built when it's out of date, ``site.py`` served by uvicorn with reload, a local stand-in for the gateway's clock, and a
supervisor that syncs each freshly started server to that clock (``clock/v1``'s ``sync_time``) and loads its seed
through the data plane (``data/reset`` then ``data/add``), as a deploy would. A server started by a reload is told
from the one before by ``BOOT_HEADER``, which ``create_app`` stamps on every answer."""
from __future__ import annotations

import datetime
import importlib.util
import inspect
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import tomllib
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

import httpx
from agentenv_protocol.client import find_extension_method, serves
from agentenv_protocol.types import METHOD_ADD, METHOD_RESET, RPC_PATH, WELL_KNOWN_PATH
from agentenv_protocol.website import CLOCK_EXTENSION_URI, WebsiteEnvironment
from agentenv_protocol.website.clock import iso, parse_time

from agent_env.env.envs.website import domains_problem

SITE_ENV = "AGENT_ENV_WEBSITE_DEV_SITE"
BOOT_HEADER = "X-Site-Boot"
POLL_QUERY = "agent-env-dev-poll"
RELOAD_INCLUDES = ["*.py", "*.json", "*.toml"]
WATCHER_STOP_TIMEOUT_S = 5.0
_WEB_SKIPPED = {"node_modules", "dist"}


class DevError(Exception):
    """Why the site can't be run as asked."""


def env_toml(site_dir: Path) -> dict | None:
    """What the folder's ``env.toml`` declares of the website env, as a bundle reads it, or None for a folder without
    one: refused when it declares another type of env, or names an ``environment_name`` or ``domains`` a website env
    can't have."""
    path = site_dir / "env.toml"
    if not path.is_file():
        return None
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as e:
        raise DevError(f"can't read {path}: {e}") from e
    if data.get("type") != "website":
        raise DevError(f"{path} declares a {data.get('type')!r} env, not a website")
    if not isinstance(data.get("environment_name", ""), str):
        raise DevError(f"{path}: environment_name is a string")
    domains = data.get("domains", [])
    if not isinstance(domains, list):
        raise DevError(f"{path}: domains is a list of host names")
    if problem := domains_problem(domains):
        raise DevError(f"{path}: {problem}")
    return data


def site_class(path: Path) -> type[WebsiteEnvironment]:
    """The WebsiteEnvironment ``path`` defines, imported under a name of its own (``site`` is the standard library's):
    the one class, or the one of several that is declared with ``@website``. Its folder goes on ``sys.path`` first,
    so it can import the files beside it."""
    folder = str(path.resolve().parent)
    if folder not in sys.path:
        sys.path.insert(0, folder)
    spec = importlib.util.spec_from_file_location(f"agent_env_dev_site_{uuid.uuid4().hex[:8]}", path)
    if spec is None or spec.loader is None:
        raise DevError(f"can't import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    found = [v for v in vars(module).values() if inspect.isclass(v) and issubclass(v, WebsiteEnvironment)
             and v.__module__ == spec.name]
    declared = [cls for cls in found if "_aee_website" in vars(cls)]
    chosen = declared if len(declared) == 1 else found
    if len(chosen) != 1:
        names = ", ".join(cls.__name__ for cls in found) or "none"
        raise DevError(f"{path} defines {len(found)} WebsiteEnvironment classes ({names}); it needs exactly one, "
                       "or one declared with @website")
    return chosen[0]


def create_app() -> Any:
    """The app of the site ``SITE_ENV`` names, for uvicorn's ``--factory``: every answer carries this process's
    ``BOOT_HEADER``."""
    path = Path(os.environ[SITE_ENV]).resolve()
    logging.getLogger("uvicorn.access").addFilter(_unpolled)
    return Booted(site_class(path)().create_app(), uuid.uuid4().hex)


def _unpolled(record: logging.LogRecord) -> bool:
    """Whether an access log line is one of a request other than the supervisor's polls."""
    return not any(isinstance(arg, str) and POLL_QUERY in arg for arg in (record.args or ()))


class Booted:
    """``app`` with ``BOOT_HEADER: boot`` on every HTTP answer."""

    def __init__(self, app: Any, boot: str) -> None:
        self.app, self.boot = app, boot

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        async def stamped(message: dict) -> None:
            if message["type"] == "http.response.start":
                message = {**message, "headers": [*message.get("headers", []),
                                                  (BOOT_HEADER.lower().encode(), self.boot.encode())]}
            await send(message)

        await self.app(scope, receive, stamped)


class DevClock:
    """A local stand-in for the gateway's clock: ``GET /clock/time`` answers ``{"virtual_time",
    "virtual_seconds_per_real_second"}`` as the gateway's does, ``GET /clock/state`` its state, ``PUT /clock/set-time`` takes ``{"virtual_time",
    "virtual_seconds_per_real_second"}``, and ``PUT /clock/advance`` moves it forward to ``{"virtual_time"}`` or by
    ``{"seconds"}``. It runs from ``start`` (wall time by default) at real speed."""

    def __init__(self, start: datetime.datetime | None = None, *, host: str = "127.0.0.1", port: int = 0) -> None:
        self._lock = threading.Lock()
        self._virtual = start or datetime.datetime.now(datetime.timezone.utc)
        self._real = time.monotonic()
        self._rate = 1.0
        self._server = ThreadingHTTPServer((host, port), _clock_handler(self))
        self._thread: threading.Thread | None = None
        self.url = f"http://{host}:{self._server.server_port}/clock/time"

    def start(self) -> DevClock:
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def now(self) -> datetime.datetime:
        with self._lock:
            return self._virtual + datetime.timedelta(seconds=(time.monotonic() - self._real) * self._rate)

    def time(self) -> dict:
        """The clock's time and its rate, read together, as ``/clock/time`` answers them."""
        with self._lock:
            virtual, real, rate = self._virtual, self._real, self._rate
        now = virtual + datetime.timedelta(seconds=(time.monotonic() - real) * rate)
        return {"virtual_time": iso(now), "virtual_seconds_per_real_second": rate}

    def state(self) -> dict:
        return {"armed": True, **self.time(), "env_get_time_url": self.url}

    def set_time(self, virtual_time: str, rate: float | None = None) -> dict:
        moment = parse_time(virtual_time)
        if rate is not None and rate < 0:
            raise ValueError("virtual_seconds_per_real_second is not negative")
        with self._lock:
            self._virtual, self._real = moment, time.monotonic()
            if rate is not None:
                self._rate = float(rate)
        return self.state()

    def advance(self, virtual_time: str | None = None, seconds: float | None = None) -> dict:
        now = self.now()
        if (virtual_time is None) == (seconds is None):
            raise ValueError("advance takes virtual_time or seconds")
        target = parse_time(virtual_time) if virtual_time is not None else now + datetime.timedelta(seconds=seconds)
        if target < now:
            raise ValueError(f"advance moves the clock forward, and {iso(target)} is before {iso(now)}")
        with self._lock:
            self._virtual, self._real = target, time.monotonic()
        return self.state()


def _clock_handler(clock: DevClock) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/clock/time":
                self._answer(200, clock.time())
            elif self.path == "/clock/state":
                self._answer(200, clock.state())
            else:
                self._answer(404, {"ok": False, "error": "not found"})

        def do_PUT(self) -> None:
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                if not isinstance(body, dict):
                    raise ValueError("body must be a JSON object")
                if self.path == "/clock/set-time":
                    self._answer(200, clock.set_time(body.get("virtual_time"), body.get("virtual_seconds_per_real_second")))
                elif self.path == "/clock/advance":
                    self._answer(200, clock.advance(body.get("virtual_time"), body.get("seconds")))
                else:
                    self._answer(404, {"ok": False, "error": "not found"})
            except (ValueError, TypeError, AttributeError) as e:
                self._answer(400, {"ok": False, "error": str(e)})

        def _answer(self, status: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args: Any) -> None:
            return None

    return Handler


def web_stale(web: Path) -> bool:
    """Whether ``web/dist`` is missing or older than a source file of ``web/``."""
    built = web / "dist" / "index.html"
    if not built.is_file():
        return True
    newest = max((p.stat().st_mtime for p in _web_sources(web)), default=0.0)
    return newest > built.stat().st_mtime


def _web_sources(web: Path):
    for path in web.rglob("*"):
        if path.is_file() and not _WEB_SKIPPED & set(path.relative_to(web).parts):
            yield path


def web_plan(web: Path, *, watch: bool) -> tuple[list[list[str]], list[str] | None]:
    """What to run in ``web/`` before the site starts: an install when ``node_modules`` is missing (``npm ci`` with a
    lockfile), and a build when ``dist`` is out of date; and with ``watch``, the build that then runs beside the site,
    rebuilding as the sources change."""
    commands = []
    if not (web / "node_modules").is_dir():
        commands.append(["npm", "ci" if (web / "package-lock.json").is_file() else "install", "--no-audit", "--no-fund"])
    if web_stale(web):
        commands.append(["npm", "run", "build"])
    return commands, (["npm", "run", "build", "--", "--watch"] if watch else None)


def prime(client: httpx.Client, base_url: str, clock_url: str, seed: Path | None) -> str | None:
    """Sync the server at ``base_url`` to the clock at ``clock_url`` with ``clock/v1``'s ``sync_time``, and load
    ``seed`` through the data plane in place of what it holds, at the card's ``url`` and refused when its declared
    ``operations`` lack ``data/reset`` or ``data/add``; returns the server's boot."""
    card_answer = client.get(base_url + WELL_KNOWN_PATH)
    card_answer.raise_for_status()
    card = card_answer.json()
    method = find_extension_method(card, CLOCK_EXTENSION_URI, "sync_time")
    if method is None or not method.get("endpoint"):
        raise DevError(f"the site's card advertises no {CLOCK_EXTENSION_URI} sync_time")
    synced = client.request(method["method"], base_url + method["endpoint"], json={"env_get_time_url": clock_url})
    synced.raise_for_status()
    if seed is not None:
        data = json.loads(seed.read_text(encoding="utf-8"))
        unserved = [m for m in (METHOD_RESET, METHOD_ADD) if serves(card, m) is False]
        if unserved:
            raise DevError(f"the site's card doesn't serve {', '.join(unserved)}, so its seed can't be loaded")
        rpc_url = data_plane_url(base_url, card)
        _rpc(client, rpc_url, METHOD_RESET, {})
        _rpc(client, rpc_url, METHOD_ADD, {"parts": [{"kind": "data", "data": data}]})
    return card_answer.headers.get(BOOT_HEADER)


def data_plane_url(base_url: str, card: dict) -> str:
    """Where the site whose ``card`` is read from ``base_url`` answers the data plane: the card's ``url``, a path
    joined onto ``base_url`` or a full URL as it is, and ``RPC_PATH`` there for a card naming none."""
    url = card.get("url") or RPC_PATH
    return url if "://" in url else f"{base_url.rstrip('/')}{url}"


def _rpc(client: httpx.Client, rpc_url: str, method: str, params: dict) -> dict:
    answer = client.post(rpc_url, json={"jsonrpc": "2.0", "id": method, "method": method, "params": params})
    answer.raise_for_status()
    body = answer.json()
    if "error" in body:
        raise DevError(f"{method} answered {body['error']}")
    return body.get("result") or {}


class Supervisor:
    """Watches the server at ``base_url`` and primes each server a reload starts; after the first, runs the
    conformance check once (and primes again, since the check leaves the server on a clock of its own). A check a
    reload cuts off is reported rather than raised, so the supervisor goes on priming later reloads. A server it
    couldn't prime, before or after the check and for a seed it couldn't read too, counts as unprimed, so the next
    poll tries again."""

    HTTP_TIMEOUT_S = 10.0

    def __init__(self, base_url: str, clock_url: str, seed: Path | None, *, check: bool = True,
                 echo: Callable[[str], None] = print, client: httpx.Client | None = None, interval: float = 1.0) -> None:
        self.base_url, self.clock_url, self.seed, self.check = base_url, clock_url, seed, check
        self.echo, self.interval = echo, interval
        self.client = client or httpx.Client(timeout=self.HTTP_TIMEOUT_S)
        self.primed: str | None = None
        self.checked = False
        self.stopped = threading.Event()

    def step(self) -> bool:
        """Prime the server if it is one not yet primed; whether it answered."""
        try:
            boot = self.client.get(f"{self.base_url}{WELL_KNOWN_PATH}?{POLL_QUERY}").headers.get(BOOT_HEADER)
        except httpx.TransportError:
            return False
        if boot is not None and boot == self.primed:
            return True
        try:
            self.primed = prime(self.client, self.base_url, self.clock_url, self.seed)
        except (httpx.HTTPError, DevError, ValueError, OSError) as e:
            self.echo(f"[dev] couldn't sync the site and load its seed: {e}")
            return True
        self.echo(f"[dev] synced the site to {self.clock_url}" + (f" and loaded {self.seed}" if self.seed else ""))
        if self.check and not self.checked:
            self.checked = True
            self._conformance()
        return True

    def _conformance(self) -> None:
        from agentenv_protocol.website import conformance

        try:
            results = conformance.check(self.base_url, client=self.client)
            for result in results:
                mark = {"passed": "PASS", "failed": "FAIL", "warning": "WARN", "skipped": "----"}[result.status]
                self.echo(f"[dev] {mark} {result.name}{': ' + result.detail if result.detail else ''}")
            failed = sum(not r.passed for r in results)
            self.echo(f"[dev] conformance: {len(results) - failed} passed or skipped, {failed} failed")
        except httpx.TransportError as e:
            self.echo(f"[dev] the conformance check lost the site, likely to a reload: {e}")
        finally:
            try:
                self.primed = prime(self.client, self.base_url, self.clock_url, self.seed)
            except (httpx.HTTPError, DevError, ValueError, OSError) as e:
                self.primed = None
                self.echo(f"[dev] couldn't sync the site again after the check: {e}")

    def run(self) -> None:
        while not self.stopped.is_set():
            self.step()
            self.stopped.wait(self.interval)

    def start(self) -> Supervisor:
        threading.Thread(target=self.run, daemon=True).start()
        return self

    def stop(self) -> None:
        self.stopped.set()


def npm_available() -> bool:
    return shutil.which("npm") is not None


def run_npm(command: list[str], web: Path, *, background: bool = False) -> subprocess.Popen | None:
    """Run ``command`` in ``web``; in the background it is returned running, else waited for (raising on failure)."""
    if background:
        return subprocess.Popen(command, cwd=web)
    subprocess.run(command, cwd=web, check=True)
    return None
