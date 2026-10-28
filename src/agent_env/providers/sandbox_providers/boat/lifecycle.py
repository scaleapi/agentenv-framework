"""Calling boat's synchronous SDK from asyncio, and creating and deleting sandboxes without leaking any.

SDK calls run on two thread pools of the provider's own: one for commands' output streams, which hold a thread for as
long as their command runs, and one for every other call, so neither a busy stream pool nor asyncio's default executor,
which the object stores share, can keep a stop or a delete from running. A create is idempotent: a retry after a lost answer sends
the same key and gets the same sandbox, while a refusal (rate limit, no capacity) made nothing, so the next attempt
takes a key of its own. A sandbox a create made but nothing holds, because the create failed or was cancelled, is found
by its create label and deleted.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, TypeVar

import urllib3
from boat_sdk.exceptions import ApiException

from agent_env.providers.sandbox_providers.boat.config import CREATE_LABEL

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Threads shared by every boat sandbox in the process: an exec's output stream holds one of STREAM_THREADS for as long
# as its command runs; creates, deletes, stops and the other short calls run on CALL_THREADS of their own.
STREAM_THREADS = 64
CALL_THREADS = 16
THREADS = STREAM_THREADS + CALL_THREADS
_STREAMS = ThreadPoolExecutor(max_workers=STREAM_THREADS, thread_name_prefix="agent-env-boat-stream")
_CALLS = ThreadPoolExecutor(max_workers=CALL_THREADS, thread_name_prefix="agent-env-boat")
# Failures of the connection, not answers from boat.
NETWORK_ERRORS = (urllib3.exceptions.HTTPError, OSError)
# Answers that refuse a create before anything is made: a full rate or concurrency limit, or no machine free.
_REFUSED = frozenset({429, 503})
_reapers: set[asyncio.Future[Any]] = set()


@dataclass(frozen=True)
class Retry:
    """How creates are retried and how hard cleanup tries."""

    attempts: int
    base_delay: float
    max_delay: float  # the longest a create waits before retrying
    delete_attempts: int
    delete_delay: float
    cancel_wait: float  # how long a cancelled create waits for its sandbox to be deleted before it is raised


async def call(fn: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
    """``fn(*args, **kwargs)``, a short SDK call, on the calls' thread pool."""
    return await asyncio.get_running_loop().run_in_executor(_CALLS, functools.partial(fn, *args, **kwargs))


async def stream(fn: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
    """``fn(*args, **kwargs)``, which follows a command's output to its end, on the streams' thread pool."""
    return await asyncio.get_running_loop().run_in_executor(_STREAMS, functools.partial(fn, *args, **kwargs))


def _spawn(cleanup: Awaitable[object]) -> asyncio.Future[Any]:
    """Run ``cleanup`` as a task no caller's cancellation reaches, kept referenced until it ends."""
    reaper = asyncio.ensure_future(cleanup)
    _reapers.add(reaper)
    reaper.add_done_callback(_reapers.discard)
    return reaper


async def settle(cleanup: Awaitable[object], *, within: float | None = None) -> None:
    """Run ``cleanup`` to its end, also while the caller is being cancelled. A second cancellation, or ``within``
    seconds passing, leaves it running on its own."""
    try:
        await asyncio.wait_for(asyncio.shield(_spawn(cleanup)), within)
    except TimeoutError:
        logger.warning("boat sandbox cleanup is still running after %ss; it continues on its own", within)


def _body(exc: ApiException) -> dict[str, Any]:
    try:
        body = json.loads(exc.body or "{}")
    except (TypeError, ValueError):
        return {}
    return body if isinstance(body, dict) else {}


def error_code(exc: ApiException) -> str | None:
    """boat's error code for ``exc``, e.g. ``rate_limited``."""
    body = _body(exc)
    code = body.get("code") or (body.get("error") or {}).get("code")
    return str(code) if code else None


def retry_after(exc: BaseException) -> float | None:
    """How long boat asks a refused call to wait, in seconds, when it says."""
    if not isinstance(exc, ApiException):
        return None
    body = _body(exc)
    details = (body.get("error") or {}).get("details") or {}
    for value in (body.get("retryAfterSeconds"), body.get("retryAfter"), details.get("retryAfterSeconds"),
                  (exc.headers or {}).get("Retry-After")):
        try:
            if value is not None:
                return float(value)
        except (TypeError, ValueError):
            continue
    return None


def is_not_found(exc: BaseException) -> bool:
    return isinstance(exc, ApiException) and exc.status == 404


async def delete_sandbox(api: Any, sandbox_id: str, retry: Retry) -> bool:
    """Delete ``sandbox_id`` with all its data, retrying through brief failures; whether it is gone."""
    for attempt in range(retry.delete_attempts):
        try:
            await call(api.delete_sandbox, sandbox_id=sandbox_id, x_ascii_confirm_delete=sandbox_id)
            return True
        except ApiException as exc:
            if exc.status == 404:
                return True
            logger.warning("Deleting boat sandbox %s failed (attempt %s): %s %s", sandbox_id, attempt + 1, exc.status,
                           error_code(exc))
        except NETWORK_ERRORS as exc:
            logger.warning("Deleting boat sandbox %s failed (attempt %s): %s", sandbox_id, attempt + 1, exc)
        if attempt + 1 < retry.delete_attempts:
            await asyncio.sleep(retry.delete_delay * 2**attempt)
    logger.error("boat sandbox %s could not be deleted and runs until its auto-stop; delete it by id", sandbox_id)
    return False


async def delete_labelled(api: Any, token: str, retry: Retry) -> None:
    """Delete every sandbox the create ``token`` made, if it made one."""
    for attempt in range(retry.delete_attempts):
        try:
            listed = await call(api.sandboxes, label=[f"{CREATE_LABEL}={token}"])
        except (ApiException, *NETWORK_ERRORS) as exc:
            logger.warning("Looking up the boat sandbox of create %s failed (attempt %s): %s", token, attempt + 1, exc)
            if attempt + 1 < retry.delete_attempts:
                await asyncio.sleep(retry.delete_delay * 2**attempt)
            continue
        for sandbox in listed.sandboxes or []:
            if await delete_sandbox(api, sandbox.id, retry):
                logger.info("Deleted boat sandbox %s, made by a create that did not complete", sandbox.id)
        return
    logger.error("A boat sandbox labelled %s=%s may run until its auto-stop; delete it by label", CREATE_LABEL, token)


async def _delete_outcome(api: Any, task: asyncio.Future[Any], token: str, retry: Retry) -> None:
    """Delete what a create its caller gave up on made, once it answers: the sandbox it returns, or the one it may
    have made though it failed, found by its create label."""
    try:
        created = await task
    except (Exception, asyncio.CancelledError):
        await delete_labelled(api, token, retry)
        return
    await delete_sandbox(api, created.sandbox.id, retry)


async def create(api: Any, request: Any, token: str, retry: Retry) -> Any:
    """``api.create(request)``: retried while refused or unanswered, leaving no sandbox behind when it fails."""
    refusals = 0
    last: BaseException | None = None
    for attempt in range(retry.attempts):
        task = asyncio.ensure_future(call(api.create, idempotency_key=f"agentenv-{token}-{refusals}",
                                          create_sandbox_request=request))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # Raised once what the create made is deleted, so an interrupted run doesn't leave it running and billed.
            await settle(_delete_outcome(api, task, token, retry), within=retry.cancel_wait)
            raise
        except ApiException as exc:
            if exc.status in _REFUSED:
                refusals += 1
            elif exc.status is not None and exc.status < 500 and error_code(exc) != "idempotency_in_progress":
                raise
            last = exc
        except NETWORK_ERRORS as exc:
            last = exc
        if attempt + 1 < retry.attempts:
            delay = min(retry_after(last) or retry.base_delay * 2**attempt, retry.max_delay)
            logger.warning("boat refused or did not answer a sandbox create (%s); retrying in %ss", last, delay)
            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                # An attempt that went unanswered may have made the sandbox, which nothing else knows of.
                await settle(delete_labelled(api, token, retry), within=retry.cancel_wait)
                raise
    # Out of attempts: an unanswered create may still have made its sandbox.
    await settle(delete_labelled(api, token, retry))
    assert last is not None
    raise last
