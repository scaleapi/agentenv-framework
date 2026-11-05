"""The universe's clock as a site reads it, the way MCP servers read it: ``clock/v1``'s ``sync_time`` hands the server
the gateway's ``/clock/time`` URL, and each request then reads that clock's ``{"virtual_time": <RFC3339>}`` once,
the one instant everything the request does reads (``now()``) and its answer carries as ``X-Universe-Now``, RFC3339 in
UTC. Requests that arrive while a read is in flight share it, over one pooled connection, so a burst costs the gateway
one read. An unsynced site reads wall time, and so does a synced one whose gateway clock isn't armed (404); one whose clock
can't be read raises ``ClockUnavailable``. ``sync_time`` with an empty URL goes back to wall time. ``sync_time`` may
also name the universe's time zone, an IANA name, which the clock keeps for the site to show local times in; the time
itself stays UTC."""
from __future__ import annotations

import asyncio
import contextvars
import datetime
import json
import re

import httpx

CLOCK_EXTENSION_URI = "urn:agentenv:clock/v1"
UNIVERSE_NOW_HEADER = "X-Universe-Now"
READ_TIMEOUT = httpx.Timeout(connect=0.5, read=1.0, write=1.0, pool=0.5)
READ_ATTEMPTS = 2

_FRACTION = re.compile(r"(\.\d+)(?=[+-]\d\d:?\d\d$|Z$|$)")
_request_now: contextvars.ContextVar[datetime.datetime | None] = contextvars.ContextVar("universe_now", default=None)


class ClockUnavailable(Exception):
    """The synced clock couldn't be read: unreachable, a 5xx after a retry, or an answer that isn't a time."""


def iso(moment: datetime.datetime) -> str:
    return moment.astimezone(datetime.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_time(value: str) -> datetime.datetime:
    """An RFC3339 time, its fraction of a second of any length (Python 3.10 parses only 3 or 6 digits)."""
    value = _FRACTION.sub(lambda m: m.group(1)[:7].ljust(7, "0"), value.replace("Z", "+00:00"))
    moment = datetime.datetime.fromisoformat(value)
    return moment if moment.tzinfo else moment.replace(tzinfo=datetime.timezone.utc)


def _wall() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


class UniverseClock:
    """The time a site answers with: wall time until ``sync`` names a gateway clock, that clock's, read per request,
    after."""

    def __init__(self) -> None:
        self.url: str | None = None
        self.timezone: str | None = None
        self._last: datetime.datetime | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._client: httpx.AsyncClient | None = None
        self._reading: tuple[str, asyncio.Future] | None = None

    @property
    def synced(self) -> bool:
        return self.url is not None

    def now(self) -> datetime.datetime:
        """The instant the request being answered read, else the last one read, else wall time."""
        return _request_now.get() or self._last or _wall()

    async def read(self) -> datetime.datetime:
        """The universe's time now: the synced clock's, wall time when none is synced or it isn't armed; raises
        ``ClockUnavailable`` when it can't be read."""
        if self.url is None:
            return _wall()
        self._last = await self._shared_read(self.url) or _wall()
        return self._last

    def answering_at(self, moment: datetime.datetime) -> contextvars.Token:
        """Make ``moment`` what ``now()`` reads while the current request is answered."""
        return _request_now.set(moment)

    def answered(self, token: contextvars.Token) -> None:
        _request_now.reset(token)

    async def sync(self, url: str, timezone: str | None = None) -> datetime.datetime:
        """Follow the clock at ``url`` from now on, in ``timezone`` (none when the sync names none), or wall time again
        for an empty ``url``; raises ``ClockUnavailable`` when the clock can't be read, keeping the clock and zone it
        followed."""
        if not url:
            self.url, self.timezone, self._last, self._reading = None, None, None, None
            return _wall()
        virtual = await _read(self._pooled(), url) or _wall()
        self.url, self.timezone, self._last, self._reading = url, timezone, virtual, None
        return virtual

    async def _shared_read(self, url: str) -> datetime.datetime | None:
        """The clock at ``url``'s time, from the read already in flight for it when there is one, else a new one."""
        client = self._pooled()
        if self._reading is None or self._reading[0] != url or self._reading[1].done():
            reading = asyncio.ensure_future(_read(client, url))
            reading.add_done_callback(lambda done: done.cancelled() or done.exception())
            self._reading = (url, reading)
        return await asyncio.shield(self._reading[1])

    def _pooled(self) -> httpx.AsyncClient:
        """The client this event loop reads the clock with, kept so its connections are reused."""
        loop = asyncio.get_running_loop()
        if self._loop is not loop or self._client is None:
            self._loop, self._client, self._reading = loop, httpx.AsyncClient(timeout=READ_TIMEOUT), None
        return self._client


async def _read(client: httpx.AsyncClient, url: str) -> datetime.datetime | None:
    """The clock at ``url``'s time, None when it isn't armed (404); a transport error or a 5xx is tried again once."""
    failure: Exception | None = None
    for _attempt in range(READ_ATTEMPTS):
        try:
            response = await client.get(url)
        except httpx.TransportError as e:
            failure = e
            continue
        if response.status_code == 404:
            return None
        if response.status_code >= 500:
            failure = httpx.HTTPStatusError(f"{response.status_code}", request=response.request, response=response)
            continue
        try:
            response.raise_for_status()
            return parse_time(response.json()["virtual_time"])
        except (httpx.HTTPStatusError, ValueError, KeyError, TypeError) as e:
            raise ClockUnavailable(f"the universe's clock at {url} answered {response.status_code}: {e}") from e
    raise ClockUnavailable(f"the universe's clock at {url} can't be read: {type(failure).__name__}: {failure}")


def clock_patch_script(now: datetime.datetime) -> bytes:
    """An inline script that shifts the page's ``Date`` to the universe's time ``now``, and follows the
    ``X-Universe-Now`` header of every ``fetch`` the page makes after. The ``Date`` it puts in place answers called,
    with the time's string, as constructed, and keeps ``Date.parse``, ``Date.UTC`` and ``instanceof Date``."""
    return ("<script>(function(){var D=Date,o=%s-D.now();function n(){return D.now()+o}"
            "function U(){if(!new.target)return new D(n()).toString();"
            "return Reflect.construct(D,arguments.length?[].slice.call(arguments):[n()],new.target)}"
            "Object.setPrototypeOf(U,D);U.prototype=D.prototype;D.prototype.constructor=U;U.now=n;U.parse=D.parse;"
            "U.UTC=D.UTC;window.Date=U;var f=window.fetch;if(f)window.fetch=function(){return f.apply(this,arguments)"
            ".then(function(r){var t=D.parse(r.headers.get(%s)||'');if(!isNaN(t))o=t-D.now();return r})}})();"
            "</script>" % (int(now.timestamp() * 1000), json.dumps(UNIVERSE_NOW_HEADER))).encode()
