"""Creating Daytona sandboxes without leaking any: rate-limit retries, and cleanup when a create is
cancelled or fails after Daytona already made the sandbox.

A cancelled caller gets its ``CancelledError`` at once, while the create it started is left to finish
and its sandbox, which nothing else holds, is deleted. A failed create may still have made a sandbox
(its response lost, or start timed out), so the sandbox's unique name is looked up and deleted.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine, Mapping
from dataclasses import dataclass
from typing import Any

from daytona import (
    DaytonaAuthenticationError,
    DaytonaAuthorizationError,
    DaytonaBadRequestError,
    DaytonaConflictError,
    DaytonaForbiddenError,
    DaytonaNotFoundError,
    DaytonaRateLimitError,
    DaytonaUnprocessableEntityError,
    DaytonaValidationError,
)

logger = logging.getLogger(__name__)

_reapers: set[asyncio.Future[Any]] = set()
# The API refused the request itself, so no sandbox exists to clean up.
_REJECTED = (
    DaytonaAuthenticationError,
    DaytonaAuthorizationError,
    DaytonaForbiddenError,
    DaytonaBadRequestError,
    DaytonaUnprocessableEntityError,
    DaytonaValidationError,
)


@dataclass(frozen=True)
class Retry:
    """How rate-limited creates are retried and how hard cleanup tries."""

    attempts: int
    base_delay: float
    reclaim_attempts: int
    reclaim_delay: float
    max_delay: float  # the longest a rate-limited create waits before retrying
    cancel_wait: float  # how long a cancelled create waits for its sandbox to be deleted before it is raised


def _retry_after(headers: Mapping[str, Any] | None) -> float | None:
    """The server's ``Retry-After*`` hint in seconds, when it sends one."""
    for key, value in (headers or {}).items():
        if str(key).lower().startswith("retry-after"):
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


async def being_destroyed(raw: Any) -> bool:
    """Whether Daytona is already deleting ``raw``, or has. A delete answered 409 means "done" only then: any
    state change in progress answers 409."""
    try:
        await raw.refresh_data()
    except DaytonaNotFoundError:
        return True
    except (
        Exception
    ):  # noqa: BLE001 - unknown counts as "not yet", which only costs another attempt
        return False
    return str(getattr(raw.desired_state, "value", raw.desired_state)) == "destroyed"


async def delete_sandbox(raw: Any, name: str, retry: Retry) -> bool:
    """Delete ``raw``, retrying through brief API failures and other state changes in progress; whether it is
    gone or being deleted."""
    for attempt in range(retry.reclaim_attempts):
        try:
            await raw.delete()
            return True
        except DaytonaNotFoundError:
            return True
        except DaytonaConflictError as exc:
            if await being_destroyed(raw):
                return True
            logger.warning(
                "Deleting Daytona sandbox %s waits out another change (attempt %s): %s",
                name,
                attempt + 1,
                exc,
            )
        except Exception as exc:  # noqa: BLE001 - retried, then reported; the TTL is the backstop
            logger.warning(
                "Deleting Daytona sandbox %s failed (attempt %s): %s", name, attempt + 1, exc
            )
        if attempt + 1 < retry.reclaim_attempts:
            await asyncio.sleep(retry.reclaim_delay * 2**attempt)
    logger.error(
        "Daytona sandbox %s could not be deleted and runs until its TTL; delete it by name", name
    )
    return False


async def delete_named(client: Any, name: str, retry: Retry) -> None:
    """Delete the sandbox called ``name`` if Daytona made one; logged by name if that can't be confirmed."""
    for attempt in range(retry.reclaim_attempts):
        try:
            raw = await client.get(name)
        except DaytonaNotFoundError:
            return
        except (
            Exception
        ) as exc:  # noqa: BLE001 - retried, then reported; never masks the caller's error
            logger.warning(
                "Looking up Daytona sandbox %s failed (attempt %s): %s", name, attempt + 1, exc
            )
            if attempt + 1 < retry.reclaim_attempts:
                await asyncio.sleep(retry.reclaim_delay * 2**attempt)
            continue
        if await delete_sandbox(raw, name, retry):
            logger.info("Deleted Daytona sandbox %s, made by a create that did not complete", name)
        return
    logger.error(
        "A Daytona sandbox named %s may be left running until its TTL; delete it by name", name
    )


def _spawn(cleanup: Coroutine[Any, Any, object]) -> asyncio.Future[Any]:
    """Run ``cleanup`` as a task no caller's cancellation reaches, kept referenced until it ends."""
    reaper = asyncio.ensure_future(cleanup)
    _reapers.add(reaper)
    reaper.add_done_callback(_reapers.discard)
    return reaper


async def settle(cleanup: Coroutine[Any, Any, object], *, within: float | None = None) -> None:
    """Run ``cleanup`` to its end, also while the caller is being cancelled. A second cancellation, or ``within``
    seconds passing, leaves it running on its own."""
    try:
        await asyncio.wait_for(asyncio.shield(_spawn(cleanup)), within)
    except TimeoutError:
        logger.warning("Daytona sandbox cleanup is still running after %ss; it continues on its own", within)


async def _delete_outcome(client: Any, task: asyncio.Future[Any], name: str, retry: Retry) -> None:
    """Delete what a create its caller gave up on made, once it answers: the sandbox it returns, or the one it may
    have made though it failed or was cancelled, found by its unique name."""
    try:
        raw = await task
    except (Exception, asyncio.CancelledError):
        await delete_named(client, name, retry)
        return
    await delete_sandbox(raw, name, retry)


async def create(client: Any, params: Any, *, timeout: float, retry: Retry) -> Any:
    """``client.create(params)``, retried while rate limited, leaving no sandbox behind on failure."""
    name = params.name
    for attempt in range(retry.attempts):
        task = asyncio.ensure_future(client.create(params, timeout=timeout))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # Raised once what the create made is deleted, so an interrupted run doesn't leave it running and billed.
            await settle(_delete_outcome(client, task, name, retry), within=retry.cancel_wait)
            raise
        except DaytonaRateLimitError as exc:
            if attempt + 1 == retry.attempts:
                raise
            delay = _retry_after(exc.headers) or retry.base_delay * 2**attempt
            logger.warning("Daytona rate-limited a sandbox create; retrying in %ss", delay)
            await asyncio.sleep(min(delay, retry.max_delay))
        except _REJECTED:
            raise
        except BaseException:
            await settle(delete_named(client, name, retry))
            raise
    raise AssertionError("unreachable")
