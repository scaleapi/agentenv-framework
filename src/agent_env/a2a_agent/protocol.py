"""Helper Functions for interacting with A2A agents (like sending a task to an A2A Agent)."""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any, Optional

import httpx
from a2a.types import TaskState
from agentenv_protocol.a2a_agent import Usage

logger = logging.getLogger(__name__)

_TERMINAL_TASK_STATES = frozenset({
    TaskState.completed, TaskState.failed, TaskState.canceled, TaskState.rejected,
})

# How long an agent on a sandbox may go without answering a poll before ``poll_a2a_task`` gives up on it, over at
# least _UNREACHABLE_MIN_POLLS polls in a row: a sandbox that died answers none, a tunnel blip drops one or two.
UNREACHABLE_AFTER_SECONDS = 60
_UNREACHABLE_MIN_POLLS = 3
# What a proxy in front of the agent answers when it can't reach it; any other status is the agent answering.
_GATEWAY_STATUSES = frozenset({502, 503, 504})
_CANCEL_TIMEOUT_SECONDS = 10
# The A2A tasks ``poll_a2a_task`` is polling in this process, or that ``wake_poll`` woke before their poll began, each
# with the event ``wake_poll`` sets. A task its waker just ended can show as working for a moment longer, so a woken
# poll asks again every _WOKEN_POLL_SECONDS for up to _WOKEN_FOR_SECONDS before going back to its interval.
_polls: dict[str, asyncio.Event] = {}
_WOKEN_POLL_SECONDS = 0.05
_WOKEN_FOR_SECONDS = 2.0


class AgentUnreachableError(TimeoutError):
    """The agent stopped answering, as one whose sandbox died does. A ``TimeoutError``, so whatever handles an
    agent running out of time handles this the same way, only sooner."""


_ERROR_CODE = re.compile(r"[a-z][a-z0-9_]{0,63}")
_REFUSED = " (the agent refused its control token: the record it was sent from has none, or another deployment's)"


def extension_error_code(response: httpx.Response) -> str | None:
    """The SDK error code an extension's error answer names, such as ``format_not_prescribed``, when it names one in
    the SDK's form: the agent writes the body, so anything else is no code."""
    try:
        body = response.json()
    except ValueError:
        return None
    error = body.get("error") if isinstance(body, Mapping) else None
    code = error.get("code") if isinstance(error, Mapping) else None
    return code if isinstance(code, str) and _ERROR_CODE.fullmatch(code) else None


def raise_for_extension_status(
    response: httpx.Response, *, operation: str, include_body: bool = False
) -> None:
    """Raise ``httpx.HTTPStatusError`` naming the SDK error code, but no other part of a body
    that may echo a grant. ``include_body`` keeps the body, for a request that sent no grant."""
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = response.text[:500] if include_body else extension_error_code(response) or ""
        status = f"{operation} failed with HTTP {response.status_code}"
        refused = _REFUSED if response.status_code == 401 else ""
        if refused:
            logger.warning(status + refused)
        message = status + (f": {detail}" if detail else "") + refused
        raise httpx.HTTPStatusError(message, request=exc.request, response=response) from exc


async def post_agent_config(url: str, payload: dict, timeout_seconds: int = 60, *,
                            headers: Mapping[str, str] | None = None) -> None:
    """POST to an agent's /ext/agent-config endpoint with retry on transient 404/connect errors.

    `url` should be the full endpoint URL (a2a_url + endpoint path from the agent card).
    """
    max_attempts = 10
    delay = 2.0
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(url, json=payload, headers=headers, timeout=timeout_seconds)
            if resp.status_code == 404:
                raise httpx.HTTPStatusError(
                    f"agent-config endpoint not available (404) at {url} after {max_attempts} "
                    f"attempts — the agent gateway isn't serving it (either boot up/initialization "
                    f"failed, or the agent gateway was torn down by TTL reasons).",
                    request=resp.request, response=resp,
                )
            resp.raise_for_status()
            if attempt > 1:
                logger.info(f"agent-config POST succeeded on attempt {attempt}")
            return
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as e:
            last_exc = e
            logger.warning(f"agent-config POST failed (attempt {attempt}/{max_attempts}, will retry): {e}")
        except httpx.HTTPStatusError as e:
            if e.response.status_code != 404:
                raise
            last_exc = e
            logger.warning(f"agent-config POST got 404 (attempt {attempt}/{max_attempts}, will retry): {url}")
        if attempt < max_attempts:
            await asyncio.sleep(delay)
            delay = min(delay * 2, 15.0)
    assert last_exc is not None
    raise last_exc


async def send_a2a_message(
    a2a_url: str,
    parts: list[dict],
    message_id: str,
    context_id: Optional[str],
    timeout_seconds: int,
) -> tuple[str, Optional[str]]:
    """POST /a2a message/send. Returns (task_id, resolved_context_id). Retries on connect errors."""
    message: dict[str, Any] = {
        "messageId": message_id, "role": "user",
        "parts": parts,
    }
    if context_id is not None:
        message["contextId"] = context_id
    payload = {
        "jsonrpc": "2.0", "id": "1", "method": "message/send",
        "params": {"message": message, "configuration": {"blocking": False}},
    }
    max_attempts = 5
    delay = 2.0
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(f"{a2a_url}/a2a", json=payload, timeout=timeout_seconds)
            resp.raise_for_status()
            send_body = resp.json()
            if "error" in send_body:
                raise RuntimeError(f"A2A message/send failed: {send_body['error']}")
            send_result = send_body.get("result") or {}
            task_id = send_result.get("id")
            if not task_id:
                raise RuntimeError(f"A2A message/send succeeded but returned no task id: {send_result}")
            resolved_context_id = context_id or send_result.get("contextId")
            return task_id, resolved_context_id
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            last_exc = e
            logger.warning(f"message/send connect error (attempt {attempt}/{max_attempts}, will retry): {e}")
        if attempt < max_attempts:
            await asyncio.sleep(delay)
            delay = min(delay * 2, 15.0)
    assert last_exc is not None
    raise last_exc


async def poll_a2a_task(
    a2a_url: str,
    task_id: str,
    timeout_seconds: int,
    poll_interval_seconds: int = 10,
    *,
    sandbox_id: Optional[str] = None,
) -> dict:
    """POST /a2a tasks/get until the task reaches a terminal state, and return it. ``wake_poll(task_id)`` has it ask
    at once rather than at its next interval.

    Out of time, it raises ``TimeoutError``. ``sandbox_id`` names the sandbox the agent runs on: then it raises
    ``AgentUnreachableError``, naming it, once the agent has gone UNREACHABLE_AFTER_SECONDS without answering,
    rather than waiting out the timeout. Either way it first asks the agent to cancel the task, so an agent still
    working on it stops."""
    woken = _polls.setdefault(task_id, asyncio.Event())
    try:
        return await _poll(a2a_url, task_id, timeout_seconds, poll_interval_seconds, sandbox_id, woken)
    finally:
        if _polls.get(task_id) is woken:
            del _polls[task_id]


def wake_poll(task_id: str) -> None:
    """Have this process's poll of ``task_id`` ask for its state now, or as it begins: for whoever just ended the task
    here, as an episode ending a turn its proxy holds does, so the step waiting on it goes on without a poll interval."""
    _polls.setdefault(task_id, asyncio.Event()).set()


async def _poll(
    a2a_url: str, task_id: str, timeout_seconds: int, poll_interval_seconds: int, sandbox_id: Optional[str],
    woken: asyncio.Event,
) -> dict:
    deadline = time.monotonic() + timeout_seconds
    consecutive_failures = 0
    answered_at, unanswered, last_failure = time.monotonic(), 0, ""
    quick_until = 0.0
    while time.monotonic() < deadline:
        backoff = min(poll_interval_seconds + consecutive_failures * 5, 60)
        if time.monotonic() < quick_until:
            backoff = min(backoff, _WOKEN_POLL_SECONDS)
        if await _nap(woken, backoff):
            quick_until = time.monotonic() + _WOKEN_FOR_SECONDS
        woken.clear()
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(f"{a2a_url}/a2a", json={
                    "jsonrpc": "2.0", "id": "poll", "method": "tasks/get",
                    "params": {"id": task_id},
                }, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as e:
            if 400 <= e.response.status_code < 500:
                raise
            consecutive_failures += 1
            logger.warning(f"A2A poll got {e.response.status_code} (consec={consecutive_failures}, will retry)")
            if e.response.status_code in _GATEWAY_STATUSES:
                unanswered, last_failure = unanswered + 1, f"HTTP {e.response.status_code}"
            else:
                answered_at, unanswered = time.monotonic(), 0
        except httpx.HTTPError as e:
            consecutive_failures += 1
            logger.warning(f"A2A poll failed (consec={consecutive_failures}, will retry): {type(e).__name__}: {e}")
            unanswered, last_failure = unanswered + 1, f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
        else:
            answered_at, unanswered = time.monotonic(), 0
            if "error" in data:
                consecutive_failures += 1
                logger.warning(f"A2A poll returned error (consec={consecutive_failures}, will retry): {data['error']}")
                continue
            consecutive_failures = 0
            if data["result"]["status"]["state"] in _TERMINAL_TASK_STATES:
                return data["result"]
            continue
        silent = time.monotonic() - answered_at
        if sandbox_id is not None and unanswered >= _UNREACHABLE_MIN_POLLS and silent >= UNREACHABLE_AFTER_SECONDS:
            await cancel_a2a_task(a2a_url, task_id)
            raise AgentUnreachableError(
                f"The agent on sandbox {sandbox_id} stopped answering: {unanswered} polls in a row over "
                f"{silent:.0f}s got no reply (last: {last_failure})")

    await cancel_a2a_task(a2a_url, task_id)
    raise TimeoutError(f"A2A task {task_id} did not complete within {timeout_seconds}s")


async def _nap(woken: asyncio.Event, seconds: float) -> bool:
    """Whether ``woken`` was set before ``seconds`` passed."""
    with contextlib.suppress(TimeoutError):
        async with asyncio.timeout(seconds):
            await woken.wait()
            return True
    return False


async def cancel_a2a_task(a2a_url: str, task_id: str) -> None:
    """POST /a2a tasks/cancel, so the agent stops working on ``task_id``. Best effort: an agent that can't be
    reached, or won't cancel, is only logged."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{a2a_url}/a2a", json={
                "jsonrpc": "2.0", "id": "cancel", "method": "tasks/cancel",
                "params": {"id": task_id},
            }, timeout=_CANCEL_TIMEOUT_SECONDS)
        resp.raise_for_status()
        error = resp.json().get("error")
    except (httpx.HTTPError, ValueError) as e:
        logger.warning(f"Couldn't cancel A2A task {task_id}: {type(e).__name__}: {e}")
        return
    if error:
        logger.warning(f"The agent didn't cancel A2A task {task_id}: {error}")
    else:
        logger.info(f"Canceled A2A task {task_id}")


_USAGE_FIELDS = frozenset(f.name for f in fields(Usage)) - {"provider_details"}
# How much of a value ``read_usage`` couldn't read it keeps, so a broken agent can't bloat a run's stored context.
_UNPARSED_PREVIEW_CHARS = 200
# Where agents before Usage had cache fields put their cache counts.
_LEGACY_CACHE_KEYS = ("cache_read_tokens", "cache_write_tokens")


def _preview(value: Any) -> Any:
    """``value`` as it came when it's short, else the start of its JSON as a string."""
    text = json.dumps(value, default=str)
    return value if len(text) <= _UNPARSED_PREVIEW_CHARS else text[:_UNPARSED_PREVIEW_CHARS] + "..."


def read_usage(raw: Any) -> tuple[Optional[Usage], Optional[dict]]:
    """An agent's reported usage and what of it couldn't be read, never raising: each field ``Usage`` has, checked on
    its own, a value it refuses kept under its name in the second (a long one cut to its start); a key it doesn't have
    kept in ``provider_details``; the cache counts an agent from before the typed fields put in ``provider_details``
    read from there."""
    usage, unparsed = _read_usage(raw)
    return usage, ({key: _preview(value) for key, value in unparsed.items()} if unparsed else None)


def _read_usage(raw: Any) -> tuple[Optional[Usage], Optional[dict]]:
    if raw is None:
        return None, None
    if not isinstance(raw, Mapping):
        return None, {"usage": raw}
    unparsed: dict = {}
    details = raw.get("provider_details")
    if details is not None and not isinstance(details, Mapping):
        unparsed["provider_details"] = details
        details = None
    details = dict(details or {})
    given = {key: value for key, value in raw.items() if key in _USAGE_FIELDS}
    details.update({key: value for key, value in raw.items() if key not in _USAGE_FIELDS and key != "provider_details"})
    for name in _LEGACY_CACHE_KEYS:
        if given.get(name) is None and name in details:
            given[name] = details.pop(name)
    valid = {}
    for key, value in given.items():
        try:
            Usage(**{key: value})
        except (TypeError, ValueError):
            unparsed[key] = value
        else:
            valid[key] = value
    try:
        usage = Usage(**valid, provider_details=details)
    except (TypeError, ValueError):
        unparsed["provider_details"] = details
        usage = Usage(**valid)
    return usage, unparsed or None


@dataclass(frozen=True)
class TerminalResponse:
    """What's pulled from an A2A terminal status message: the text reply, the typed
    ``structured_output`` (when ``output_format`` was set), tool/error telemetry, and the turn's ``usage`` as
    ``read_usage`` reads it, with what of it couldn't be read in ``usage_unparsed``."""

    response_text: str
    tool_call_count: Optional[int] = None
    error_type: Optional[str] = None
    error_code: Optional[str] = None
    error_class: Optional[str] = None
    error_message: Optional[str] = None
    structured_output: Optional[Any] = None
    usage: Optional[Usage] = None
    usage_unparsed: Optional[dict] = None

    @classmethod
    def from_message(cls, status_message: dict) -> "TerminalResponse":
        """Parse an A2A terminal status message. Per A2A spec 0.3 the final agent message
        (TaskUpdater.complete/failed) lives in result.status.message, not result.history."""
        response_text = ""
        tool_call_count: Optional[int] = None
        error_type: Optional[str] = None
        error_code: Optional[str] = None
        error_class: Optional[str] = None
        error_message: Optional[str] = None
        structured_output: Optional[Any] = None
        usage: Optional[Usage] = None
        usage_unparsed: Optional[dict] = None
        for part in (status_message or {}).get("parts", []):
            kind = part.get("kind")
            if kind == "text":
                response_text = part["text"]
            elif kind == "data":
                data = part.get("data", {})
                if "tool_call_count" in data:
                    tool_call_count = data["tool_call_count"]
                if "usage" in data:
                    usage, usage_unparsed = read_usage(data["usage"])
                    if isinstance(data["usage"], dict) and "tool_call_count" in data["usage"]:
                        tool_call_count = data["usage"]["tool_call_count"]
                if "error_type" in data:
                    error_type = data["error_type"]
                if "error_code" in data:
                    error_code = data["error_code"]
                if "error_class" in data:
                    error_class = data["error_class"]
                if "error_message" in data:
                    error_message = data["error_message"]
                if "structured_output" in data:
                    structured_output = data["structured_output"]
        return cls(
            response_text=response_text,
            tool_call_count=tool_call_count,
            error_type=error_type,
            error_code=error_code,
            error_class=error_class,
            error_message=error_message,
            structured_output=structured_output,
            usage=usage,
            usage_unparsed=usage_unparsed,
        )


def extract_terminal_response(
    status_message: dict,
) -> tuple[str, Optional[int], Optional[str], Optional[str], Optional[str]]:
    """Back-compat shim over ``TerminalResponse.from_message``; external callers
    (the hub's agents.py) still unpack the 5-tuple. Drop once they migrate."""
    tr = TerminalResponse.from_message(status_message)
    return tr.response_text, tr.tool_call_count, tr.error_type, tr.error_class, tr.error_message
