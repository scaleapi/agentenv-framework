"""Retry helper for transient sandbox-exec failures.

Use only for read-only sandbox calls; wrapping write paths can leak state
or duplicate sandboxes on retry.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def is_transient_exec_error(exc: BaseException) -> bool:
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionResetError, ConnectionRefusedError)):
        return True
    try:
        import ssl
        if isinstance(exc, ssl.SSLError):
            return True
    except ImportError:
        pass
    try:
        import httpx
    except ImportError:
        httpx = None  # type: ignore
    if httpx is not None:
        if isinstance(exc, (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError, httpx.PoolTimeout, httpx.ConnectTimeout, httpx.WriteTimeout)):
            return True
        if isinstance(exc, httpx.HTTPStatusError):
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status is not None and (status >= 500 or status in (408, 429)):
                return True
    try:
        from websockets.exceptions import InvalidStatus, WebSocketException
    except ImportError:
        return False
    if isinstance(exc, InvalidStatus):
        status = getattr(getattr(exc, "response", None), "status_code", None)
        return status is None or status >= 500
    return isinstance(exc, WebSocketException)


async def retry_transient(
    op: Callable[[], Awaitable[T]],
    *,
    attempts: int = 5,
    initial_backoff: float = 2.0,
    max_backoff: float = 32.0,
    desc: str = "exec",
) -> T:
    """Retry an async sandbox call on transient failures with exponential backoff.

    Defaults (5 attempts × base 2s → cap 32s) → ~30s worst-case wall-clock.
    """
    last: BaseException | None = None
    for i in range(attempts):
        try:
            return await op()
        except BaseException as e:
            if not is_transient_exec_error(e):
                raise
            last = e
            if i + 1 == attempts:
                break
            wait = min(initial_backoff * (2 ** i), max_backoff)
            logger.warning(
                f"{desc}: transient {type(e).__name__} ({e}), retrying in {wait:.1f}s "
                f"(attempt {i + 1}/{attempts})"
            )
            await asyncio.sleep(wait)
    assert last is not None
    raise last
