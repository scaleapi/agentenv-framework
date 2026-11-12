"""The followers of the gateway's clock (the computers on it): each long-polls ``/clock/watch`` with the version it
shows, which is its acknowledgement, and a clock write can wait until every one shows the write's version."""
from __future__ import annotations

import asyncio
import hmac
import math
import time
from collections.abc import Callable, Mapping

from .clock import Clock, ClockError
from .constants import CLOCK_MAX_WAIT_FOLLOWERS_S, CLOCK_WATCH_HOLD_S


def parse_wait(value) -> float:
    """A write's ``wait_followers_s``: how long it waits for the followers, 0 when absent."""
    if value is None:
        return 0.0
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ClockError(f"wait_followers_s must be a non-negative number, got {value!r}")
    # Only a float can be NaN or infinite; math.isfinite on an int past a float's range raises.
    if (isinstance(value, float) and not math.isfinite(value)) or value < 0:
        raise ClockError(f"wait_followers_s must be a non-negative number, got {value!r}")
    if value > CLOCK_MAX_WAIT_FOLLOWERS_S:
        raise ClockError(f"wait_followers_s exceeds max {CLOCK_MAX_WAIT_FOLLOWERS_S:g}")
    return float(value)


class ClockFollowers:
    """The declared followers, each by name with the read token its watch carries, and the version each last showed.
    Its event and condition bind to the loop that first waits on them, so one gateway's watches and waiting writes all
    run on the loop that serves it, never one asyncio.run apiece."""

    WATCH_S = CLOCK_WATCH_HOLD_S
    #: How long since a follower last watched before it has gone quiet: longer than a watch is held and a follower takes
    #: to reconnect, so its connection is gone. A write doesn't wait for a quiet follower; it names it at once.
    QUIET_S = CLOCK_WATCH_HOLD_S + 5

    def __init__(self, clock: Callable[[], Clock], tokens: Mapping[str, str]):
        self._clock = clock
        self._tokens = dict(tokens)
        self._shown: dict[str, int | None] = dict.fromkeys(self._tokens)
        self._seen: dict[str, float] = {}
        self._changed = asyncio.Event()
        self._acks = asyncio.Condition()

    def changed(self) -> None:
        """Answer every held watch: the clock has a new version."""
        self._changed.set()
        self._changed = asyncio.Event()

    def follower_for(self, name: str | None, token: str) -> str | None:
        """``name`` when it's a declared follower and ``token`` its read token, else None."""
        expected = self._tokens.get(name or "")
        return name if expected and token and hmac.compare_digest(token.encode(), expected.encode()) else None

    def _quiet_at(self, name: str) -> float | None:
        """When ``name`` goes quiet (monotonic), if it has watched."""
        seen = self._seen.get(name)
        return None if seen is None else seen + self.QUIET_S

    def _quiet(self, name: str) -> bool:
        at = self._quiet_at(name)
        return at is not None and time.monotonic() >= at

    async def watch(self, after: str, follower: str | None) -> None:
        """Take ``after`` as ``follower``'s acknowledgement (one ``follower_for`` named) when it's a version this clock
        has had, later than the one it last showed; then return at once when the clock's version isn't ``after``, else
        at its next change or after ``WATCH_S``."""
        changed = self._changed
        if follower is not None:
            count = self._clock().count_of(after)
            async with self._acks:
                shown = self._shown[follower]
                if count is not None and (shown is None or count > shown):
                    self._shown[follower] = count
                self._seen[follower] = time.monotonic()
                self._acks.notify_all()
        if after == self._clock().version:
            try:
                await asyncio.wait_for(changed.wait(), self.WATCH_S)
            except asyncio.TimeoutError:
                pass

    async def stale_after(self, version: str, wait_s: float) -> list[str]:
        """The followers that haven't shown ``version`` (or a later one) once each has, or has gone quiet (``QUIET_S``),
        or ``wait_s`` has passed: woken by each acknowledgement and when the next follower waited for goes quiet. One
        that hasn't watched yet is waited for: it may be starting."""
        target = self._clock().count_of(version)
        deadline = time.monotonic() + wait_s

        def stale() -> list[str]:
            return sorted(name for name, shown in self._shown.items()
                          if shown is None or target is None or shown < target)

        async with self._acks:
            while (waited_for := [name for name in stale() if not self._quiet(name)]) and time.monotonic() < deadline:
                wake = min([deadline, *filter(None, map(self._quiet_at, waited_for))])
                try:
                    await asyncio.wait_for(self._acks.wait(), max(0.0, wake - time.monotonic()))
                except asyncio.TimeoutError:
                    pass
            return stale()

    def state(self) -> dict:
        """Each follower's last shown version and how long ago it last watched, for ``/clock/state``."""
        now, clock = time.monotonic(), self._clock()
        return {name: {"version": None if shown is None else clock.version_of(shown),
                       "seen_s_ago": round(now - self._seen[name], 3) if name in self._seen else None}
                for name, shown in self._shown.items()}
