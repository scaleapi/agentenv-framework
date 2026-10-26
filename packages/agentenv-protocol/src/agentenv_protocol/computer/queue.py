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
        """Hold the queue for one call, or refuse with input_busy after ``wait_s``. Waiting is first come, first
        served; a waiter woken as its time runs out still takes its turn."""
        waiter = asyncio.ensure_future(self._lock.acquire())
        done, _ = await asyncio.wait({waiter}, timeout=wait_s)
        if not done:
            waiter.cancel()
            try:
                await waiter
            except asyncio.CancelledError:
                raise InputBusy(f"waited {wait_s:g} s for earlier input to finish") from None
        try:
            yield
        finally:
            self._lock.release()

    def delivered(self, caller: str) -> None:
        self._total += 1
        self._by_caller[caller] += 1

    def snapshot(self) -> tuple:
        return self._total, dict(self._by_caller)

    def others_since(self, snapshot: tuple, caller: str) -> int:
        total, by_caller = snapshot
        return (self._total - total) - (self._by_caller[caller] - by_caller.get(caller, 0))
