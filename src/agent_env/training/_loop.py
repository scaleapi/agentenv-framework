"""The event loop this process's training sessions share, on a daemon thread of its own: a trainer calls its
environments from threads, or awaits them on an event loop of its own that a run's blocking calls mustn't stall."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Coroutine
from typing import Any

_loop: asyncio.AbstractEventLoop | None = None
_loop_lock = threading.Lock()


def on_session_loop(coroutine: Coroutine[Any, Any, Any], timeout: float | None) -> Any:
    """``coroutine``'s result, run on the session loop; waits at most ``timeout`` seconds."""
    future = asyncio.run_coroutine_threadsafe(coroutine, session_loop())
    try:
        return future.result(timeout)
    except TimeoutError:
        future.cancel()
        raise


def session_loop() -> asyncio.AbstractEventLoop:
    global _loop
    with _loop_lock:
        if _loop is None:
            _loop = asyncio.new_event_loop()
            threading.Thread(target=_loop.run_forever, name="agentenv-sessions", daemon=True).start()
        return _loop
