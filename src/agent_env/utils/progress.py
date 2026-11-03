"""The progress lines of the run a task is part of, which code running inside the task, such as a deploy that has to
compose first, tells what it's doing, as ``agent-env run`` prints them. Steps and what they call run in the task's
context, which ``asyncio.create_task`` and ``asyncio.to_thread`` carry; outside a run there's none."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Optional

_progress: ContextVar[Optional[Callable[[str], None]]] = ContextVar("agent_env_run_progress", default=None)


@contextmanager
def run_progress(say: Callable[[str], None]) -> Iterator[None]:
    """The tasks run inside it tell ``say`` their progress."""
    token = _progress.set(say)
    try:
        yield
    finally:
        _progress.reset(token)


def current_progress() -> Optional[Callable[[str], None]]:
    """What the run in progress takes its lines with, or None outside a run."""
    return _progress.get()
