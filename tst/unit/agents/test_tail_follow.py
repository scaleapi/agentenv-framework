"""The follower every agent's bridge feeds its atif/1 adapter with (``utils/tail_follow``), which each image copies
beside its bridge: a tail's records fed while the CLI runs and once more after it exits, none of them lost however the
run ends."""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import sys
import threading
from pathlib import Path

import pytest

import agent_env

_PATH = Path(agent_env.__file__).parent / "utils" / "tail_follow" / "tail_follow.py"
_spec = importlib.util.spec_from_file_location("shared_tail_follow", _PATH)
tail_follow = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = tail_follow
_spec.loader.exec_module(tail_follow)


class _Tail:
    """A tail that gives one record a read, ``final`` on the last, the second read held until ``release`` is set."""

    def __init__(self, hold_second=False):
        self.reads, self.release, self.reading = 0, threading.Event(), threading.Event()
        self.hold_second = hold_second

    def read(self, final):
        if final:
            return [("aux", "final")]
        self.reads += 1
        if self.hold_second and self.reads == 2:
            self.reading.set()
            self.release.wait(5)
            return [("aux", "in flight")]
        return [("aux", f"read {self.reads}")] if self.reads == 1 else []


def test_a_read_in_flight_when_the_run_ends_is_fed_before_the_final_one():
    tail, fed = _Tail(hold_second=True), []

    async def run():
        async def release_once_stopping():
            await asyncio.to_thread(tail.reading.wait, 5)
            await asyncio.sleep(0.05)
            tail.release.set()

        async with tail_follow.following(tail.read, lambda name, record: fed.append(record), 0.001, "test"):
            await asyncio.to_thread(tail.reading.wait, 5)
            asyncio.get_running_loop().create_task(release_once_stopping())

    asyncio.run(run())

    assert fed == ["read 1", "in flight", "final"]


def test_a_run_that_raises_still_has_what_its_files_gained_fed():
    tail, fed = _Tail(), []

    async def run():
        async with tail_follow.following(tail.read, lambda name, record: fed.append((name, record)), 0.001, "test"):
            raise RuntimeError("the CLI failed")

    with pytest.raises(RuntimeError, match="the CLI failed"):
        asyncio.run(run())

    assert fed[-1] == ("aux", "final")


def test_a_follower_that_fails_is_logged_and_the_run_goes_on(caplog):
    def read(final):
        raise OSError("the file went away")

    async def run():
        async with tail_follow.following(read, lambda name, record: None, 0.001, "transcript"):
            await asyncio.sleep(0.01)
        return "answered"

    with caplog.at_level(logging.WARNING):
        assert asyncio.run(run()) == "answered"

    assert "following the transcript failed" in caplog.text
