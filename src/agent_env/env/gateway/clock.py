"""Wall-clock-derived virtual clock for the env gateway (urn:agentenv:clock/v1).

Reads compute t0 + real_elapsed * rate, so virtual time advances on its own (independent of
tool calls) and reads never mutate it. Off by default: an unarmed clock 404s.
"""
from __future__ import annotations

import math
import re
import secrets
import threading
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


class ClockError(ValueError):
    """Invalid clock configuration (surfaced as HTTP 400)."""


MAX_RATE = 86400.0  # 1 real second = 1 virtual day (the max virtual-time speed)

# Fast-fail arming a clock that would reach LATEST within a real hour at `rate` (frozen clocks exempt): an hour, so a
# day per second still arms at today's dates. Not the guarantee: a clock that runs to LATEST stands there.
_OVERFLOW_HORIZON_S = 3600.0
_MAX_T0 = datetime.max.replace(tzinfo=timezone.utc)
# The times the clock holds: those 64 bits of nanoseconds since 1970 hold, as Python and the computers' clock page keep
# time, so every program that follows the clock can read it (Python's time.time() raises outside them).
EARLIEST = datetime(1677, 9, 21, 0, 12, 44, tzinfo=timezone.utc)
LATEST = datetime(2262, 4, 11, 23, 47, 16, tzinfo=timezone.utc)


def _parse_rate(rate) -> float:
    if isinstance(rate, bool) or not isinstance(rate, (int, float)):
        raise ClockError(f"rate must be a number, got {type(rate).__name__}")
    # NaN passes both ordered bounds below, then poisons every later read with an uncatchable 500; only a float can be
    # NaN or infinite, and math.isfinite on an int past a float's range raises.
    if isinstance(rate, float) and not math.isfinite(rate):
        raise ClockError(f"rate must be a finite number, got {rate}")
    if rate < 0:
        raise ClockError("rate must be non-negative")
    if rate > MAX_RATE:
        raise ClockError(f"rate exceeds max {MAX_RATE:g} (1 real s = 1 virtual day)")
    return float(rate)


# fromisoformat is looser than RFC3339 (accepts basic format, week/ordinal dates, sub-minute
# offsets), so gate on the grammar first and let it do only the arithmetic. A space separator is ok.
_RFC3339_RE = re.compile(
    r"""^(\d{4,})-\d{2}-\d{2}       # a year past four digits is read, to be refused as outside the range
        [Tt ]
        \d{2}:\d{2}:\d{2}           # seconds are mandatory
        (?:\.\d+)?
        (?:[Zz]|[+-]\d{2}:\d{2})$   # offset is hours:minutes only, or Z
    """,
    re.VERBOSE,
)
_RFC3339_SHAPE = "YYYY-MM-DDThh:mm:ss[.fff](Z|+hh:mm)"


def _parse_rfc3339(text: str) -> datetime:
    if not isinstance(text, str):
        raise ClockError(f"virtual_time must be an RFC3339 string, got {type(text).__name__}")
    # Don't strip: a whitespace-wrapped value would slip through the anchored grammar (a 200 the contract calls a 400).
    raw = text
    grammar = _RFC3339_RE.match(raw)
    if not grammar:
        raise ClockError(f"invalid RFC3339 virtual_time {text!r}: expected {_RFC3339_SHAPE}")
    if len(grammar.group(1)) > 4 or grammar.group(1) == "0000":
        raise ClockError(outside_range(f"virtual_time {text!r}"))
    if raw.endswith(("Z", "z")):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError as e:
        raise ClockError(f"invalid RFC3339 virtual_time {text!r}: {e}")
    try:
        return dt.astimezone(timezone.utc)
    except OverflowError:
        raise ClockError(outside_range(f"virtual_time {text!r}"))


_WHY = "the times a 64-bit count of nanoseconds since 1970 holds, as Python and the computers that follow the clock keep time"


def outside_range(what: str) -> str:
    """The refusal of ``what``, a time the clock can't hold, naming its range and why."""
    return f"{what} is outside the env clock's range, {_iso(EARLIEST)} to {_iso(LATEST)} ({_WHY})"


def _past_end(what: str) -> str:
    return f"{what} would pass {_iso(LATEST)}, the end of the env clock's range, within a real hour ({_WHY})"


def _ends_within_an_hour(at: datetime, rate: float) -> bool:
    """Whether a clock at ``at``, running at ``rate``, reaches LATEST within _OVERFLOW_HORIZON_S real seconds."""
    return (LATEST - at).total_seconds() < _OVERFLOW_HORIZON_S * rate


def check_arm(virtual_time: str, rate=1.0, what: str | None = None) -> tuple[datetime, float]:
    """``virtual_time`` and ``rate`` as set_time arms the clock at them, or the ClockError it refuses them with, which
    names the time as ``what`` (``virtual_time '<it>'`` by default)."""
    t0 = _parse_rfc3339(virtual_time)
    r = _parse_rate(1.0 if rate is None else rate)
    what = what or f"virtual_time {virtual_time!r}"
    if not EARLIEST <= t0 <= LATEST:
        raise ClockError(outside_range(what))
    if _ends_within_an_hour(t0, r):
        raise ClockError(_past_end(f"{what} at rate {r:g}"))
    return t0, r


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _zone(name) -> ZoneInfo | None:
    """The IANA time zone `name` names, or None if it names none."""
    if not isinstance(name, str):
        return None
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        return None


# Calendar Y/M are absent on purpose: a month is not a fixed span. Empty ("P"/"PT") is caught by the
# any-group check in _parse_duration, not here.
_ISO8601_DURATION_RE = re.compile(
    r"""^P (?:(\d+)W)? (?:(\d+)D)?           # weeks / days
          (?: T (?:(\d+)H)? (?:(\d+)M)?      # hours / minutes ('M' is months before T, minutes after)
                (?:(\d+(?:\.\d+)?)S)? )? $   # seconds, optionally fractional
    """,
    re.VERBOSE,
)


def _parse_duration(text) -> timedelta:
    if not isinstance(text, str):
        raise ClockError(f"duration must be an ISO-8601 string, got {type(text).__name__}")
    m = _ISO8601_DURATION_RE.match(text)
    if m is None or not any(m.groups()):
        raise ClockError(f"invalid ISO-8601 duration {text!r}: expected e.g. 'PT24H', 'P1DT6H', 'PT30M'")
    weeks, days, hours, minutes, seconds = m.groups()
    try:
        return timedelta(weeks=int(weeks or 0), days=int(days or 0), hours=int(hours or 0),
                         minutes=int(minutes or 0), seconds=float(seconds or 0))
    except (OverflowError, ValueError):
        raise ClockError(f"invalid ISO-8601 duration {text!r}: out of representable range")


def _advance(t0: datetime, seconds: float) -> datetime:
    """t0 + seconds, saturating at datetime.min/max instead of raising, as a trigger's mark far past the clock's range
    may need; the clock's own time stops at LATEST (Clock._upto_end)."""
    try:
        return t0 + timedelta(seconds=seconds)
    except (OverflowError, ValueError, OSError):
        return _MAX_T0 if seconds > 0 else datetime.min.replace(tzinfo=timezone.utc)


class Clock:
    """Gateway-owned virtual clock — wall-clock-driven, monotonic, threadsafe; `now_fn` injectable for tests."""

    BOOT_ID_BYTES = 6  # a version's per-boot id, random, so a restarted gateway's versions never repeat its last boot's

    def __init__(self, now_fn=time.monotonic):
        self._now = now_fn
        self._lock = threading.Lock()
        self._armed = False
        self._t0: datetime | None = None
        self._time_zone: str | None = None  # the universe's zone, its users', as the arm named it
        self._anchor = now_fn()  # real time the current arm (or unarmed stretch) began
        self._rate = 1.0
        self._generation = 0
        self._lived = 0.0  # virtual seconds run before the current arm/clear
        self._ahead = 0.0  # virtual seconds the current arm was advanced by
        self._boot = secrets.token_hex(self.BOOT_ID_BYTES)
        self._count = 0

    @property
    def armed(self) -> bool:
        return self._armed

    @property
    def version(self) -> str:
        """Opaque, changed by every write: this clock's own id and a count, so it never repeats across restarts."""
        return self.version_of(self._count)

    def version_of(self, count: int) -> str:
        """The version this clock had at its ``count``th write."""
        return f"{self._boot}.{count}"

    def count_of(self, version) -> int | None:
        """The count of ``version`` when this clock has had it; None for another clock's, a later one, or junk."""
        boot, _, count = version.partition(".") if isinstance(version, str) else ("", "", "")
        if boot != self._boot or not (count.isascii() and count.isdigit()):
            return None
        # Counts carry no leading zeros, so one with more digits than the current count is past it, and is refused before
        # int() reads it: int() of thousands of digits raises.
        if len(count) > len(str(self._count)) or int(count) > self._count:
            return None
        return int(count)

    @property
    def rate(self) -> float:
        """Virtual seconds per real second; 1.0 while unarmed."""
        return self._rate

    @property
    def generation(self) -> int:
        """Bumped on every arm/clear — lets consumers detect a re-anchor and drop stale derived state."""
        return self._generation

    def t0(self) -> datetime | None:
        """The armed anchor virtual time (None when unarmed) — for time-trigger mark resolution."""
        with self._lock:
            return self._t0

    def set_time(self, virtual_time: str, rate=1.0, time_zone: str | None = None) -> dict:
        """Arm/re-arm: anchor t0=virtual_time to now, advancing at `rate` virtual s per real s, in the universe's
        `time_zone` (its users' IANA zone), or none."""
        t0, r = check_arm(virtual_time, rate)
        if time_zone is not None and _zone(time_zone) is None:
            raise ClockError(f"unknown time zone {time_zone!r}: pass an IANA time zone name such as 'America/New_York'")
        with self._lock:
            self._bank(t0)
            self._t0 = t0
            self._rate = r
            self._ahead = 0.0
            self._armed = True
            self._time_zone = time_zone
            self._generation += 1
            self._count += 1
        return self.state()

    def set_rate(self, rate) -> dict:
        """Change an armed clock's rate from its time at the call, keeping t0 and generation as an advance does, so time
        triggers keep their marks and a rate change can't step the clock back by a time read a moment earlier; 0 freezes
        the clock. No rate is refused for nearing the end of the range, which the clock stands at once it gets there, so
        a harness that froze it near the end can always start it again."""
        r = _parse_rate(rate)
        with self._lock:
            if not self._armed or self._t0 is None:
                raise ClockError("clock is not armed")
            now = self._now()
            ran = (now - self._anchor) * self._rate
            self._lived += ran
            self._ahead += ran
            self._anchor = now
            self._rate = r
            self._count += 1
        return self.state()

    def advance(self, virtual_time: str) -> dict:
        """Jump forward to virtual_time on the current arm, keeping t0, rate and generation, so marks stay put; a time
        past the end of the range is refused, one in its last real hour isn't, and the clock stands at the end once
        there."""
        target = _parse_rfc3339(virtual_time)
        with self._lock:
            if not self._armed or self._t0 is None:
                raise ClockError("clock is not armed")
            if target > LATEST:
                raise ClockError(outside_range(f"virtual_time {virtual_time!r}"))
            current = self._upto_end((self._now() - self._anchor) * self._rate + self._ahead)
            jump = (target - current).total_seconds()
            if jump < 0:
                raise ClockError(f"advance only moves forward: {virtual_time!r} is before the current {_iso(current)}")
            self._ahead += jump
            self._lived += jump
            self._count += 1
        return self.state()

    def clear(self) -> dict:
        """Disarm (idempotent). Afterwards reads 404."""
        with self._lock:
            self._bank()
            self._armed = False
            self._t0 = None
            self._time_zone = None
            self._rate = 1.0
            self._ahead = 0.0
            self._generation += 1
            self._count += 1
            return {"ok": True, "armed": False, "version": self.version}

    def elapsed(self) -> float:
        """Virtual seconds run so far: real time at each arm's rate (1 unarmed), plus each forward re-arm or advance."""
        with self._lock:
            return self._lived + (self._now() - self._anchor) * self._rate

    def now(self) -> datetime | None:
        with self._lock:
            if not self._armed or self._t0 is None:
                return None
            elapsed = self._now() - self._anchor
            return self._upto_end(elapsed * self._rate + self._ahead)

    @property
    def time_zone(self) -> str | None:
        """The universe's zone, its users', as the current arm named it; None unarmed or when it named none."""
        with self._lock:
            return self._time_zone if self._armed else None

    def at_elapsed(self, elapsed: float) -> datetime | None:
        """The virtual time at which elapsed() reaches `elapsed` on the current arm; None while unarmed."""
        with self._lock:
            if not self._armed or self._t0 is None:
                return None
            return _advance(self._t0, elapsed - self._lived + self._ahead)

    def read(self) -> dict:
        """The frozen consumer contract: {"virtual_time": "<RFC3339>"} — pure read, never advances."""
        current = self.now()
        if current is None:
            raise ClockError("clock is not armed")
        return {"virtual_time": _iso(current)}

    def state(self) -> dict:
        """The clock as its followers take it up; one at the end of its range stands there, at rate 0."""
        with self._lock:
            if not self._armed or self._t0 is None:
                return {"armed": False, "version": self.version}
            current = self._upto_end((self._now() - self._anchor) * self._rate + self._ahead)
            return {"armed": True, "t0": _iso(self._t0),
                    "virtual_seconds_per_real_second": 0.0 if current == LATEST else self._rate,
                    "virtual_time": _iso(current),
                    **({"time_zone": self._time_zone} if self._time_zone else {}), "version": self.version}

    def _upto_end(self, seconds: float) -> datetime:
        """The current arm's time ``seconds`` after t0, standing at LATEST: no check at a write can bound a gateway that
        outlives it, so the clock stops at the end of its range rather than leave it. Lock held."""
        return min(_advance(self._t0, seconds), LATEST)

    def _bank(self, t0: datetime | None = None) -> None:
        """Fold the stretch since the last arm/clear into `_lived` at its rate, and start the next one now."""
        now = self._now()
        ran = (now - self._anchor) * self._rate
        self._lived += ran
        if t0 is not None and self._armed:  # a re-arm forward is time the world lived through; one back adds nothing
            self._lived += max(0.0, (t0 - _advance(self._t0, ran + self._ahead)).total_seconds())
        self._anchor = now
