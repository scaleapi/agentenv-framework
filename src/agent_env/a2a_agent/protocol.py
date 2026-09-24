"""Helper Functions for interacting with A2A agents (like sending a task to an A2A Agent)."""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

import httpx
from a2a.types import TaskState

logger = logging.getLogger(__name__)

_TERMINAL_TASK_STATES = frozenset({
    TaskState.completed, TaskState.failed, TaskState.canceled, TaskState.rejected,
})


async def post_agent_config(url: str, payload: dict, timeout_seconds: int = 60) -> None:
    """POST to an agent's /ext/agent-config endpoint with retry on transient 404/connect errors.

    `url` should be the full endpoint URL (a2a_url + endpoint path from the agent card).
    """
    max_attempts = 10
    delay = 2.0
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(url, json=payload, timeout=timeout_seconds)
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
) -> dict:
    """POST /a2a tasks/get until status.state is 'completed' or 'failed'. Returns the result dict."""
    deadline = time.monotonic() + timeout_seconds
    consecutive_failures = 0
    while time.monotonic() < deadline:
        backoff = min(poll_interval_seconds + consecutive_failures * 5, 60)
        await asyncio.sleep(backoff)
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(f"{a2a_url}/a2a", json={
                    "jsonrpc": "2.0", "id": "poll", "method": "tasks/get",
                    "params": {"id": task_id},
                }, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                consecutive_failures += 1
                logger.warning(f"A2A poll returned error (consec={consecutive_failures}, will retry): {data['error']}")
                continue
            result = data["result"]
        except httpx.HTTPStatusError as e:
            if 400 <= e.response.status_code < 500:
                raise
            consecutive_failures += 1
            logger.warning(f"A2A poll got {e.response.status_code} (consec={consecutive_failures}, will retry)")
            continue
        except httpx.HTTPError as e:
            consecutive_failures += 1
            logger.warning(f"A2A poll failed (consec={consecutive_failures}, will retry): {type(e).__name__}: {e}")
            continue

        consecutive_failures = 0
        state = result["status"]["state"]
        if state in _TERMINAL_TASK_STATES:
            return result

    raise TimeoutError(f"A2A task {task_id} did not complete within {timeout_seconds}s")


@dataclass(frozen=True)
class TerminalResponse:
    """What's pulled from an A2A terminal status message: the text reply, the typed
    ``structured_output`` (when ``output_format`` was set), and tool/error telemetry."""

    response_text: str
    tool_call_count: Optional[int] = None
    error_type: Optional[str] = None
    error_code: Optional[str] = None
    error_class: Optional[str] = None
    error_message: Optional[str] = None
    structured_output: Optional[Any] = None

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
        for part in (status_message or {}).get("parts", []):
            kind = part.get("kind")
            if kind == "text":
                response_text = part["text"]
            elif kind == "data":
                data = part.get("data", {})
                if "tool_call_count" in data:
                    tool_call_count = data["tool_call_count"]
                usage = data.get("usage")
                if isinstance(usage, dict) and "tool_call_count" in usage:
                    tool_call_count = usage["tool_call_count"]
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
        )


def extract_terminal_response(
    status_message: dict,
) -> tuple[str, Optional[int], Optional[str], Optional[str], Optional[str]]:
    """Back-compat shim over ``TerminalResponse.from_message``; external callers
    (the hub's agents.py) still unpack the 5-tuple. Drop once they migrate."""
    tr = TerminalResponse.from_message(status_message)
    return tr.response_text, tr.tool_call_count, tr.error_type, tr.error_class, tr.error_message
