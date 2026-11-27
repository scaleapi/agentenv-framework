"""How an agent's bridge feeds its atif/1 adapter the records the CLI writes to its own files, imported by every bridge
that follows them, whose image copies it beside the bridge (``shared_sources.toml``): a tail of those files, read while
the CLI runs and once more after it exits, each record fed as an aux record as it's read."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Callable, Iterable
from typing import Any

logger = logging.getLogger(__name__)

#: A tail's read: the (aux file, record) pairs it gained since the last read, everything that's left when ``final``.
Read = Callable[[bool], Iterable[tuple[str, Any]]]


async def follow(read: Read, feed: Callable[[str, Any], None], stopped: asyncio.Event, poll_s: float) -> None:
    """Feed what ``read`` gives every ``poll_s`` until ``stopped`` is set, then what it gives read final. A read in
    progress when it's set is fed first: it has already moved past what it returns, so dropping it would lose it."""
    while not stopped.is_set():
        for name, record in await asyncio.to_thread(read, False):
            feed(name, record)
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stopped.wait(), poll_s)
    for name, record in await asyncio.to_thread(read, True):
        feed(name, record)


@contextlib.asynccontextmanager
async def following(read: Read, feed: Callable[[str, Any], None], poll_s: float, what: str) -> AsyncIterator[None]:
    """Follow ``read`` (``follow``) while the block runs, the CLI's run, and have it feed what's left once the block
    ends, however it ends. A follower that failed is logged; the turn goes on without the later records of ``what``."""
    stopped = asyncio.Event()
    task = asyncio.create_task(follow(read, feed, stopped, poll_s))
    try:
        yield
    finally:
        stopped.set()
        try:
            await task
        except asyncio.CancelledError:
            task.cancel()
            raise
        except Exception:
            logger.warning("following the %s failed; the ATIF copy lacks its later records", what, exc_info=True)
