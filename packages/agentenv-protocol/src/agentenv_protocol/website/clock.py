"""The universe's clock as a site reads it: ``clock/v1``'s ``sync_time`` hands the server the gateway's ``/clock/time``
URL, as it hands it to MCP servers, and the site's time then follows that clock's ``{"virtual_time": <RFC3339>}``,
read at most once a second and extrapolated between reads. The rate it extrapolates at is the one the clock answers
with, ``virtual_seconds_per_real_second``, as the gateway's does; for a clock that answers none, it is one two
successive intervals between reads agreed on, so a clock set or advanced between two reads is taken for a jump, not a
speed, and until two agree it is real time's. An unsynced site reads wall time. Every answer of a site carries the time it answered at as ``X-Universe-Now``,
RFC3339 in UTC. ``sync_time`` may also name the universe's time zone, an IANA name, which the clock keeps for the site
to show local times in; the time itself stays UTC."""
from __future__ import annotations

import asyncio
import datetime
import json
import logging
import math
import time

import httpx

logger = logging.getLogger(__name__)

CLOCK_EXTENSION_URI = "urn:agentenv:clock/v1"
UNIVERSE_NOW_HEADER = "X-Universe-Now"
REFRESH_S = 1.0
READ_TIMEOUT_S = 5.0
RATE_AGREEMENT = 0.1


def iso(moment: datetime.datetime) -> str:
    return moment.astimezone(datetime.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_time(value: str) -> datetime.datetime:
    moment = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    return moment if moment.tzinfo else moment.replace(tzinfo=datetime.timezone.utc)


class UniverseClock:
    """The time a site answers with: wall time until ``sync`` names a gateway clock, that clock's after."""

    def __init__(self) -> None:
        self.url: str | None = None
        self.timezone: str | None = None
        self._virtual: datetime.datetime | None = None
        self._read_at = 0.0
        self._tried_at = 0.0
        self._rate = 1.0
        self._measured: float | None = None
        self._lock: asyncio.Lock | None = None

    @property
    def synced(self) -> bool:
        return self._virtual is not None

    def now(self) -> datetime.datetime:
        if self._virtual is None:
            return datetime.datetime.now(datetime.timezone.utc)
        return self._virtual + datetime.timedelta(seconds=(time.monotonic() - self._read_at) * self._rate)

    async def sync(self, url: str, timezone: str | None = None) -> datetime.datetime:
        """Follow the clock at ``url`` from now on, in ``timezone`` (none when the sync names none); raises when it
        can't be read, keeping the clock and zone it followed. It waits for a refresh reading the clock it followed,
        so that read never lands on the new clock."""
        async with self._reading():
            virtual, rate = await _read(url)
            self.url, self._virtual, self._rate, self._measured = url, virtual, 1.0 if rate is None else rate, None
            self.timezone = timezone
            self._read_at = self._tried_at = time.monotonic()
            return virtual

    async def refresh(self) -> None:
        """Read the synced clock again when the last read is a second old; a failed read keeps extrapolating."""
        if self.url is None or time.monotonic() - self._tried_at < REFRESH_S:
            return
        async with self._reading():
            started = time.monotonic()
            if started - self._tried_at < REFRESH_S:
                return
            self._tried_at = started
            try:
                virtual, rate = await _read(self.url)
            except Exception as e:
                logger.debug("reading the universe's clock at %s failed: %r", self.url, e)
                return
            read_at = time.monotonic()
            if rate is not None:
                self._rate = rate
            elif self._virtual is not None and read_at > self._read_at:
                measured = max(0.0, (virtual - self._virtual).total_seconds() / (read_at - self._read_at))
                if self._measured is not None and abs(measured - self._measured) <= RATE_AGREEMENT * max(measured, self._measured, 1.0):
                    self._rate = measured
                self._measured = measured
            self._virtual, self._read_at = virtual, read_at

    def _reading(self) -> asyncio.Lock:
        """The lock a sync and a refresh each read the clock under."""
        self._lock = self._lock or asyncio.Lock()
        return self._lock


async def _read(url: str) -> tuple[datetime.datetime, float | None]:
    """The clock's time, and the rate it says it runs at: None when it says none, or none that is a finite rate."""
    async with httpx.AsyncClient() as client:
        response = await client.get(url, timeout=READ_TIMEOUT_S)
        response.raise_for_status()
        body = response.json()
        rate = body.get("virtual_seconds_per_real_second")
        ok = isinstance(rate, (int, float)) and not isinstance(rate, bool) and math.isfinite(rate) and rate >= 0
        return parse_time(body["virtual_time"]), float(rate) if ok else None


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
