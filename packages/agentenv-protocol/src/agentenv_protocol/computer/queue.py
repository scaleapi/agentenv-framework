"""The input queue: input calls run one at a time in arrival order, and delivered inputs are counted per caller so
a result can say how many others landed since the frame it named."""
from __future__ import annotations

import asyncio
import contextlib
from collections import Counter
from typing import AsyncIterator

from .errors import InputBusy


class InputQueue:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._total = 0
        self._by_caller: Counter = Counter()

    @contextlib.asynccontextmanager
    async def turn(self, wait_s: float) -> AsyncIterator[None]:
        """Hold the queue for one call, or refuse with input_busy after ``wait_s``; waiting is first come, first
        served. A wait given up, by the timeout or by the caller's cancellation, never leaves the queue held."""
        waiter = asyncio.ensure_future(self._lock.acquire())
        try:
            done, _ = await asyncio.wait({waiter}, timeout=wait_s)
        except BaseException:
            self._abandon(waiter)
            raise
        if not done:
            self._abandon(waiter)
            raise InputBusy(f"waited {wait_s:g} s for earlier input to finish")
        try:
            yield
        finally:
            self._lock.release()

    def _abandon(self, waiter: asyncio.Future) -> None:
        """Give up a wait for the lock; if it was granted anyway, give the lock back."""
        waiter.cancel()
        waiter.add_done_callback(lambda w: None if w.cancelled() or w.exception() is not None else self._lock.release())

    def delivered(self, caller: str) -> None:
        self._total += 1
        self._by_caller[caller] += 1

    def snapshot(self) -> tuple:
        return self._total, dict(self._by_caller)

    def others_since(self, snapshot: tuple, caller: str) -> int:
        total, by_caller = snapshot
        return (self._total - total) - (self._by_caller[caller] - by_caller.get(caller, 0))
