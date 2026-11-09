"""The processes a computer app's server launched, each reaped once it exits, so its exit code can be asked for; shared
by every OS's session (``session``)."""
from __future__ import annotations

import asyncio
from typing import Optional

#: The processes launched into the session, kept until they exit so each is reaped.
_running: set[asyncio.Task] = set()
#: Each launched process's reaping, by pid, until its exit is asked for: its return code, negative for a signal.
_exits: dict[int, asyncio.Future] = {}
#: How many exits no one has asked for are kept, the latest.
KEPT_EXITS = 16


def reap(process: asyncio.subprocess.Process) -> None:
    """Reap ``process`` once it exits, its exit kept until it's asked for or ``KEPT_EXITS`` later ones push it out; any
    pipe it was started with is read to its end, so it closes with the process."""
    reaped = asyncio.ensure_future(_exited(process))
    _running.add(reaped)
    reaped.add_done_callback(_running.discard)
    _exits[process.pid] = reaped
    reaped.add_done_callback(lambda _: _forget_old_exits())


async def _exited(process: asyncio.subprocess.Process) -> int:
    await process.communicate()
    return process.returncode


def forget(pid: int) -> None:
    """No one asks how ``pid`` exited: a process the server stopped."""
    _exits.pop(pid, None)


def _forget_old_exits() -> None:
    """Drop all but the latest ``KEPT_EXITS`` exits no one has asked for."""
    for pid in [pid for pid, reaped in _exits.items() if reaped.done()][:-KEPT_EXITS]:
        del _exits[pid]


async def returncode(pid: int, timeout_s: float) -> Optional[int]:
    """The return code the process ``launch`` started as ``pid`` exited with, negative for the signal that ended it,
    once it's reaped within ``timeout_s``; None for one ``launch`` didn't start or that hasn't exited."""
    reaped = _exits.get(pid)
    if reaped is None:
        return None
    try:
        code = await asyncio.wait_for(asyncio.shield(reaped), timeout_s)
    except TimeoutError:
        return None
    _exits.pop(pid, None)
    return code
