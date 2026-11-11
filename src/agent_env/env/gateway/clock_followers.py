"""The computers that follow the gateway's clock: each long-polls ``/clock/watch`` with the version it shows, which is
its acknowledgement, and a clock write can wait until every one shows the write's version."""
from __future__ import annotations

import asyncio
import hmac
import math
import time
from collections.abc import Callable, Mapping

from .clock import Clock, ClockError

MAX_WAIT_FOLLOWERS_S = 30.0


def parse_wait(value) -> float:
    """A write's ``wait_followers_s``: how long it waits for the followers, 0 when absent."""
    if value is None:
        return 0.0
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ClockError(f"wait_followers_s must be a non-negative number, got {value!r}")
    if value > MAX_WAIT_FOLLOWERS_S:
        raise ClockError(f"wait_followers_s {value:g} exceeds max {MAX_WAIT_FOLLOWERS_S:g}")
    return float(value)


class ClockFollowers:
    """The declared followers, each by name with the read token its watch carries, and the version each last showed.
    A watch acknowledges only with its follower's token; ``ClockFollowers`` is bound to the gateway's event loop."""

    WATCH_S = 25.0

    def __init__(self, clock: Callable[[], Clock], tokens: Mapping[str, str]):
        self._clock = clock
        self._tokens = dict(tokens)
        self._shown: dict[str, tuple[int, str] | None] = dict.fromkeys(self._tokens)
        self._seen: dict[str, float] = {}
        self._changed = asyncio.Event()
        self._acks = asyncio.Condition()

    def changed(self) -> None:
        """Answer every held watch: the clock has a new version."""
        self._changed.set()
        self._changed = asyncio.Event()

    async def watch(self, after: str, follower: str | None, token: str) -> None:
        """Take ``after`` as ``follower``'s acknowledgement when ``token`` is its read token and it's a version this clock
        has had, later than the one it last showed; then return at once when the clock's version isn't ``after``, else
        at its next change or after ``WATCH_S``."""
        changed = self._changed
        expected = self._tokens.get(follower or "")
        if expected and token and hmac.compare_digest(token.encode(), expected.encode()):
            count = self._clock().count_of(after)
            async with self._acks:
                shown = self._shown[follower]
                if count is not None and (shown is None or count > shown[0]):
                    self._shown[follower] = (count, after)
                self._seen[follower] = time.monotonic()
                self._acks.notify_all()
        if after == self._clock().version:
            try:
                await asyncio.wait_for(changed.wait(), self.WATCH_S)
            except asyncio.TimeoutError:
                pass

    async def stale_after(self, version: str, wait_s: float) -> list[str]:
        """The followers that haven't shown ``version`` (or a later one) once each has or ``wait_s`` has passed."""
        target = self._clock().count_of(version)

        def stale() -> list[str]:
            return sorted(name for name, shown in self._shown.items()
                          if shown is None or target is None or shown[0] < target)

        async with self._acks:
            try:
                await asyncio.wait_for(self._acks.wait_for(lambda: not stale()), wait_s)
            except asyncio.TimeoutError:
                pass
            return stale()

    def state(self) -> dict:
        """Each follower's last shown version and how long ago it last watched, for ``/clock/state``."""
        now = time.monotonic()
        return {name: {"version": shown[1] if shown else None,
                       "seen_s_ago": round(now - self._seen[name], 3) if name in self._seen else None}
                for name, shown in self._shown.items()}
