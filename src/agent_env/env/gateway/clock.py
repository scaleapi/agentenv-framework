"""Wall-clock-derived virtual clock for the env gateway (urn:agentenv:clock/v1).

Reads compute t0 + real_elapsed * rate, so virtual time advances on its own (independent of
tool calls) and reads never mutate it. Off by default: an unarmed clock 404s.
"""
from __future__ import annotations

import math
import re
import threading
import time
from datetime import datetime, timedelta, timezone


class ClockError(ValueError):
    """Invalid clock configuration (surfaced as HTTP 400)."""


MAX_RATE = 86400.0  # 1 real second = 1 virtual day (the max virtual-time speed)

# Fast-fail a t0 that would overflow within a real day of advancement at `rate` (frozen clocks
# exempt). Not the overflow guarantee — reads still saturate (_advance) if the gateway outlives it.
_OVERFLOW_HORIZON_S = 86400.0
_MAX_T0 = datetime.max.replace(tzinfo=timezone.utc)


def _parse_rate(rate) -> float:
    if isinstance(rate, bool) or not isinstance(rate, (int, float)):
        raise ClockError(f"rate must be a number, got {type(rate).__name__}")
    # NaN passes both ordered bounds below, then poisons every later read with an uncatchable 500.
    if not math.isfinite(rate):
        raise ClockError(f"rate must be a finite number, got {rate}")
    if rate < 0:
        raise ClockError("rate must be non-negative")
    if rate > MAX_RATE:
        raise ClockError(f"rate {rate:g} exceeds max {MAX_RATE:g} (1 real s = 1 virtual day)")
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


def _parse_rfc3339(text: str) -> datetime:
    if not isinstance(text, str):
        raise ClockError(f"virtual_time must be an RFC3339 string, got {type(text).__name__}")
    # Don't strip: a whitespace-wrapped value would slip through the anchored grammar (a 200 the contract calls a 400).
    raw = text
    if not _RFC3339_RE.match(raw):
        raise ClockError(f"invalid RFC3339 virtual_time {text!r}: expected {_RFC3339_SHAPE}")
    if raw.endswith(("Z", "z")):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError as e:
        raise ClockError(f"invalid RFC3339 virtual_time {text!r}: {e}")
    try:
        return dt.astimezone(timezone.utc)
    except OverflowError:
        raise ClockError(f"invalid RFC3339 virtual_time {text!r}: out of representable range")


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


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

    def __init__(self, now_fn=time.monotonic):
        self._now = now_fn
        self._lock = threading.Lock()
        self._armed = False
        self._t0: datetime | None = None
        self._anchor: float | None = None
        self._rate = 1.0
        self._generation = 0

    @property
    def armed(self) -> bool:
        return self._armed

    @property
    def generation(self) -> int:
        """Bumped on every arm/clear — lets consumers detect a re-anchor and drop stale derived state."""
        return self._generation

    def t0(self) -> datetime | None:
        """The armed anchor virtual time (None when unarmed) — for time-trigger mark resolution."""
        with self._lock:
            return self._t0

    def set_time(self, virtual_time: str, rate=1.0) -> dict:
        """Arm/re-arm: anchor t0=virtual_time to now, advancing at `rate` virtual s per real s."""
        t0 = _parse_rfc3339(virtual_time)
        r = _parse_rate(1.0 if rate is None else rate)
        if t0 > _MAX_T0 - timedelta(seconds=_OVERFLOW_HORIZON_S * r):
            raise ClockError(f"virtual_time {virtual_time!r} at rate {r:g} would overflow datetime near the max")
        with self._lock:
            self._t0 = t0
            self._anchor = self._now()
            self._rate = r
            self._armed = True
            self._generation += 1
        return self.state()

    def clear(self) -> dict:
        """Disarm (idempotent). Afterwards reads 404."""
        with self._lock:
            self._armed = False
            self._t0 = None
            self._anchor = None
            self._rate = 1.0
            self._generation += 1
        return {"ok": True, "armed": False}

    def now(self) -> datetime | None:
        with self._lock:
            if not self._armed or self._t0 is None:
                return None
            elapsed = self._now() - self._anchor
            return _advance(self._t0, elapsed * self._rate)

    def read(self) -> dict:
        """The frozen consumer contract: {"virtual_time": "<RFC3339>"} — pure read, never advances."""
        current = self.now()
        if current is None:
            raise ClockError("clock is not armed")
        return {"virtual_time": _iso(current)}

    def state(self) -> dict:
        with self._lock:
            if not self._armed or self._t0 is None:
                return {"armed": False}
            elapsed = self._now() - self._anchor
            return {"armed": True, "t0": _iso(self._t0), "virtual_seconds_per_real_second": self._rate,
                    "virtual_time": _iso(_advance(self._t0, elapsed * self._rate))}
