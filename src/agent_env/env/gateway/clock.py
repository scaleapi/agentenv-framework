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
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


class ClockError(ValueError):
    """Invalid clock configuration (surfaced as HTTP 400)."""


class OutsideRange(ClockError):
    """A time outside the range of times a clock follower holds, or one the clock would run past its end from."""


MAX_RATE = 86400.0  # 1 real second = 1 virtual day (the max virtual-time speed)

# Fast-fail a t0 that would overflow within a real day of advancement at `rate` (frozen clocks
# exempt). Not the overflow guarantee — reads still saturate (_advance) if the gateway outlives it.
_OVERFLOW_HORIZON_S = 86400.0
_MAX_T0 = datetime.max.replace(tzinfo=timezone.utc)
# Fast-fail arming a clock that a follower would see run past the latest time it holds within a real hour at `rate`: an
# hour, so a day per second still arms at today's dates on a computer, whose range ends in 2262.
_FOLLOWER_HORIZON_S = 3600.0


@dataclass(frozen=True)
class FollowerRange:
    """The times a clock follower holds, as its card declares them (clock/v1's ``range``): a computer's and a
    clock-following agent's are what 64 bits of nanoseconds since 1970 hold, as their clock page and Python keep time."""

    earliest: datetime
    latest: datetime

    def as_dict(self) -> dict[str, str]:
        return {"earliest": _iso(self.earliest), "latest": _iso(self.latest)}


def parse_follower_range(value) -> FollowerRange:
    """A follower's declared range, ``{"earliest": <RFC3339>, "latest": <RFC3339>}``."""
    if not isinstance(value, dict):
        raise ClockError(f"a follower's range takes earliest and latest RFC3339 times, got {value!r}")
    held = FollowerRange(_parse_rfc3339(value.get("earliest"), "range earliest"),
                         _parse_rfc3339(value.get("latest"), "range latest"))
    if held.earliest > held.latest:
        raise ClockError(f"a follower's range ends before it starts: {value!r}")
    return held


def _check_range(what: str, name: str, held: FollowerRange, t: datetime, rate: float | None) -> None:
    """Refuse ``t`` outside the range follower ``name`` holds, or, run at ``rate``, one it would pass the end of that
    range from within a real hour."""
    where = f"clock follower {name!r} holds, {_iso(held.earliest)} to {_iso(held.latest)}"
    if not held.earliest <= t <= held.latest:
        raise OutsideRange(f"{what} is outside the times {where}")
    if rate and (held.latest - t).total_seconds() < _FOLLOWER_HORIZON_S * rate:
        raise OutsideRange(f"{what} at rate {rate:g} would pass the end of the times {where}, within a real hour")


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
    r"""^\d{4}-\d{2}-\d{2}
        [Tt ]
        \d{2}:\d{2}:\d{2}           # seconds are mandatory
        (?:\.\d+)?
        (?:[Zz]|[+-]\d{2}:\d{2})$   # offset is hours:minutes only, or Z
    """,
    re.VERBOSE,
)
_RFC3339_SHAPE = "YYYY-MM-DDThh:mm:ss[.fff](Z|+hh:mm)"


def _parse_rfc3339(text: str, what: str = "virtual_time") -> datetime:
    if not isinstance(text, str):
        raise ClockError(f"{what} must be an RFC3339 string, got {type(text).__name__}")
    # Don't strip: a whitespace-wrapped value would slip through the anchored grammar (a 200 the contract calls a 400).
    raw = text
    if not _RFC3339_RE.match(raw):
        raise ClockError(f"invalid RFC3339 {what} {text!r}: expected {_RFC3339_SHAPE}")
    if raw.endswith(("Z", "z")):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError as e:
        raise ClockError(f"invalid RFC3339 {what} {text!r}: {e}")
    try:
        return dt.astimezone(timezone.utc)
    except OverflowError:
        raise ClockError(f"invalid RFC3339 {what} {text!r}: out of representable range")


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
    """t0 + seconds, saturating at datetime.min/max instead of raising.

    set_time's horizon check can't bound a gateway that outlives _OVERFLOW_HORIZON_S, so reads
    saturate rather than 500 — an armed clock always returns a (monotonic) time.
    """
    try:
        return t0 + timedelta(seconds=seconds)
    except (OverflowError, ValueError, OSError):
        return _MAX_T0 if seconds > 0 else datetime.min.replace(tzinfo=timezone.utc)


class Clock:
    """Gateway-owned virtual clock — wall-clock-driven, monotonic, threadsafe; `now_fn` injectable for tests."""

    BOOT_ID_BYTES = 6  # a version's per-boot id, random, so a restarted gateway's versions never repeat its last boot's

    def __init__(self, now_fn=time.monotonic,
                 follower_ranges: Callable[[], Mapping[str, FollowerRange]] = dict):
        self._now = now_fn
        self._follower_ranges = follower_ranges
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
        t0 = _parse_rfc3339(virtual_time)
        r = _parse_rate(1.0 if rate is None else rate)
        if time_zone is not None and _zone(time_zone) is None:
            raise ClockError(f"unknown time zone {time_zone!r}: pass an IANA time zone name such as 'America/New_York'")
        if t0 > _MAX_T0 - timedelta(seconds=_OVERFLOW_HORIZON_S * r):
            raise ClockError(f"virtual_time {virtual_time!r} at rate {r:g} would overflow datetime near the max")
        self._check_followers(f"virtual_time {virtual_time!r}", t0, r)
        with self._lock:
            now = self._now()
            self._stop_at_end(now)
            self._bank(now, t0)
            self._t0 = t0
            self._rate = r
            self._ahead = 0.0
            self._armed = True
            self._time_zone = time_zone
            self._generation += 1
            self._count += 1
        return self.state()

    def _check_followers(self, what: str, t: datetime, rate: float | None) -> None:
        """Refuse ``t``, a time the clock would be set or advanced to, outside the range of a follower it has, or, armed
        at ``rate``, one it would run past the end of that range from within a real hour."""
        for name, held in sorted(self._follower_ranges().items()):
            _check_range(what, name, held, t, rate)

    def check_follower(self, name: str, held: FollowerRange) -> None:
        """Refuse a follower ``name`` that holds only ``held`` while the clock is outside it now, or, at its rate, would
        run past its end from within a real hour, as set_time refuses arming with one; an unarmed clock takes any."""
        with self._lock:
            if not self._armed or self._t0 is None:
                return
            real = self._now()
            self._stop_at_end(real)
            now, rate = self._current(real), self._rate
        _check_range(f"the clock, at {_iso(now)},", name, held, now, rate)

    def set_rate(self, rate) -> dict:
        """Change an armed clock's rate from its time at the call, keeping t0 and generation as an advance does, so time
        triggers keep their marks and a rate change can't step the clock back by a time read a moment earlier; 0 freezes
        the clock."""
        r = _parse_rate(rate)
        with self._lock:
            if not self._armed or self._t0 is None:
                raise ClockError("clock is not armed")
            now = self._now()
            self._stop_at_end(now)
            ran = (now - self._anchor) * self._rate
            if _advance(self._t0, ran + self._ahead) > _MAX_T0 - timedelta(seconds=_OVERFLOW_HORIZON_S * r):
                raise ClockError(f"rate {r:g} would overflow datetime near the max")
            self._lived += ran
            self._ahead += ran
            self._anchor = now
            self._rate = r
            self._count += 1
        return self.state()

    def advance(self, virtual_time: str) -> dict:
        """Jump forward to virtual_time on the current arm, keeping t0, rate and generation, so marks stay put."""
        target = _parse_rfc3339(virtual_time)
        with self._lock:
            if not self._armed or self._t0 is None:
                raise ClockError("clock is not armed")
            if target > _MAX_T0 - timedelta(seconds=_OVERFLOW_HORIZON_S * self._rate):
                raise ClockError(f"virtual_time {virtual_time!r} at rate {self._rate:g} would overflow datetime near the max")
            self._check_followers(f"virtual_time {virtual_time!r}", target, None)
            now = self._now()
            self._stop_at_end(now)
            current = self._current(now)
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
            now = self._now()
            self._stop_at_end(now)
            self._bank(now)
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
            now = self._now()
            self._stop_at_end(now)
            return self._lived + (now - self._anchor) * self._rate

    def now(self) -> datetime | None:
        with self._lock:
            if not self._armed or self._t0 is None:
                return None
            now = self._now()
            self._stop_at_end(now)
            return self._current(now)

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
            self._stop_at_end(self._now())
            return _advance(self._t0, elapsed - self._lived + self._ahead)

    def read(self) -> dict:
        """The frozen consumer contract: {"virtual_time": "<RFC3339>"} — pure read, never advances."""
        current = self.now()
        if current is None:
            raise ClockError("clock is not armed")
        return {"virtual_time": _iso(current)}

    def state(self) -> dict:
        with self._lock:
            if not self._armed or self._t0 is None:
                return {"armed": False, "version": self.version}
            now = self._now()
            self._stop_at_end(now)
            return {"armed": True, "t0": _iso(self._t0), "virtual_seconds_per_real_second": self._rate,
                    "virtual_time": _iso(self._current(now)),
                    **({"time_zone": self._time_zone} if self._time_zone else {}), "version": self.version}

    def _current(self, now: float) -> datetime:
        """The current arm's time at real time ``now``, the one time a locked operation reads, so its stop at a
        follower's end and its reading agree. Lock held."""
        return _advance(self._t0, (now - self._anchor) * self._rate + self._ahead)

    def stop_at_end(self) -> None:
        """``_stop_at_end``, taking the lock: before a follower's range stops applying, as when it retires."""
        with self._lock:
            self._stop_at_end(self._now())

    def _stop_at_end(self, now: float) -> None:
        """Once the clock has run to the end of the narrowest range a follower holds, stop it there, frozen (rate 0) from
        the moment it got there, as the follower's page stands there: its reads, delays and marks agree with what the
        follower shows, and it stays there until a write moves it. Lock held."""
        if not self._armed or self._t0 is None or self._rate <= 0:
            return
        end = min((held.latest for held in self._follower_ranges().values()), default=None)
        if end is None:
            return
        to_end = max(0.0, (end - _advance(self._t0, self._ahead)).total_seconds())
        if (now - self._anchor) * self._rate < to_end:
            return
        self._lived += to_end
        self._ahead += to_end
        self._anchor = now
        self._rate = 0.0

    def _bank(self, now: float, t0: datetime | None = None) -> None:
        """Fold the stretch since the last arm/clear, to real time ``now``, into `_lived` at its rate, and start the next
        one then."""
        ran = (now - self._anchor) * self._rate
        self._lived += ran
        if t0 is not None and self._armed:  # a re-arm forward is time the world lived through; one back adds nothing
            self._lived += max(0.0, (t0 - _advance(self._t0, ran + self._ahead)).total_seconds())
        self._anchor = now
