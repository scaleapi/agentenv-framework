"""``awaiting_lock``: a coroutine waits on the event loop for another holder of the lock, told as it finds the lock held
and again every ``every_s`` until it's free."""

import asyncio

import pytest

from agent_env.store.local_state import awaiting_lock


@pytest.mark.asyncio
async def test_a_waiter_is_told_once_it_finds_the_lock_held_and_again_every_interval_with_the_time_waited(local_stores):
    told, held = [], asyncio.Event()

    async def hold():
        async with awaiting_lock("compose:desktop"):
            held.set()
            await asyncio.sleep(1)

    async def waiting(seconds):
        told.append(seconds)

    holder = asyncio.create_task(hold())
    await held.wait()
    async with awaiting_lock("compose:desktop", waiting, every_s=0.3):
        pass
    await holder

    assert told[0] == 0 and len(told) >= 2 and all(later - earlier >= 0.3 for earlier, later in zip(told, told[1:]))
