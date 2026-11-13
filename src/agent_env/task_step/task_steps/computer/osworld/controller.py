"""The OSWorld controller beside a computer, as the ``osworld_*`` steps call it over HTTP: ``/reset``,
``/initialize_task``, ``/setup``, and ``/evaluate``, the last started with ``/evaluate_async`` and polled with short
requests, since a judge can run for minutes and a long silent connection is reaped by what's between. Every call
carries the bearer token ``osworld_deploy_controller`` started the controller with (``Endpoint``)."""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

from agent_env.config import get_config

logger = logging.getLogger(__name__)

#: How long one controller call may take: OSWorld-V2 ``setup()`` downloading assets, or a judge.
DEFAULT_TIMEOUT_S = 1800
#: The controller has no health route; ``GET /configure`` is a read that answers once its app is up.
READY_PATH = "/configure"
_POLL_INTERVAL_S = 15
_POLL_READ_TIMEOUT_S = 30
#: ``/evaluate_async`` answers with a job id at once; the grading runs after.
START_TIMEOUT_S = 60
#: One ``GET`` of ``READY_PATH``, while waiting for the controller and checking it requires its token.
READY_CALL_TIMEOUT_S = 15
#: A task ref the controller fetches is signed for this long, covering one step's download window.
TASK_REF_TTL_S = 6 * 60 * 60


@dataclass(frozen=True)
class Endpoint:
    """A controller's address and what every call to it sends: the bearer token it requires."""

    url: str
    headers: dict[str, str] = field(default_factory=dict, repr=False)

    def at(self, path: str) -> str:
        return f"{self.url.rstrip('/')}{path}"


def bearer_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


class ControllerCallError(RuntimeError):
    """A controller call that failed in transport or answered with an HTTP error; carries the status and body."""

    def __init__(self, message: str, *, status: int | None = None, body: str | None = None):
        super().__init__(message)
        self.status = status
        self.body = body


def _parsed(path: str, response: httpx.Response, *, expect_json: bool) -> dict:
    """A 2xx answer's JSON object. Raises ControllerCallError on an HTTP error, and, with ``expect_json``, on an answer
    that isn't a JSON object, since every caller reads it as one; without it such an answer is ``{}``."""
    if response.status_code >= 400:
        raise ControllerCallError(f"controller {path} returned HTTP {response.status_code}: {response.text[:500]}",
                                  status=response.status_code, body=response.text)
    try:
        answer = json.loads(response.text)
    except json.JSONDecodeError as e:
        if not expect_json:
            return {}
        raise ControllerCallError(f"controller {path} returned non-JSON: {e}: {response.text[:300]}",
                                  status=response.status_code, body=response.text) from e
    if isinstance(answer, dict):
        return answer
    if not expect_json:
        return {}
    raise ControllerCallError(f"controller {path} returned {type(answer).__name__}, not a JSON object: "
                              f"{response.text[:300]}", status=response.status_code, body=response.text)


async def post(endpoint: Endpoint, path: str, payload: dict, *, timeout: float = DEFAULT_TIMEOUT_S,
               expect_json: bool = True) -> dict:
    """POST ``payload`` to the controller's ``path``. With ``expect_json`` False a 2xx body that isn't JSON (``/reset``
    answers a bare string) returns ``{}``. Raises ControllerCallError on a transport failure or an HTTP error."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(endpoint.at(path), json=payload, headers=endpoint.headers, timeout=timeout)
    except httpx.HTTPError as e:
        raise ControllerCallError(f"controller {path} transport failure: {type(e).__name__}: {e}", body=str(e)) from e
    return _parsed(path, response, expect_json=expect_json)


async def evaluate(endpoint: Endpoint, payload: dict, *, timeout: float = DEFAULT_TIMEOUT_S) -> dict:
    """``/evaluate`` with ``payload``: started with ``/evaluate_async`` and polled at ``/evaluate_result`` until it's
    done, or the blocking ``/evaluate`` on a controller without the async routes."""
    try:
        async with httpx.AsyncClient() as client:
            start = await client.post(endpoint.at("/evaluate_async"), json=payload, headers=endpoint.headers,
                                      timeout=START_TIMEOUT_S)
    except httpx.HTTPError as e:
        raise ControllerCallError(f"controller /evaluate_async transport failure: {type(e).__name__}: {e}",
                                  body=str(e)) from e
    if start.status_code == 404:
        logger.info("controller has no /evaluate_async; evaluating with the blocking /evaluate")
        return await post(endpoint, "/evaluate", payload, timeout=timeout)
    job_id = _parsed("/evaluate_async", start, expect_json=True).get("job_id")
    if not job_id:
        raise ControllerCallError(f"controller /evaluate_async returned no job_id: {start.text[:300]}")
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        if loop.time() >= deadline:
            raise ControllerCallError(f"controller /evaluate job {job_id} still running after {timeout}s")
        await asyncio.sleep(_POLL_INTERVAL_S)
        try:
            async with httpx.AsyncClient() as client:
                poll = await client.get(endpoint.at("/evaluate_result"), params={"job_id": job_id},
                                        headers=endpoint.headers, timeout=_POLL_READ_TIMEOUT_S)
        except httpx.HTTPError as e:
            logger.warning(f"controller /evaluate_result poll failed, polling again: {type(e).__name__}: {e}")
            continue
        if poll.status_code >= 400:
            raise ControllerCallError(f"controller /evaluate_result returned HTTP {poll.status_code}: "
                                      f"{poll.text[:500]}", status=poll.status_code, body=poll.text)
        try:
            data = json.loads(poll.text) or {}
        except json.JSONDecodeError:
            logger.warning(f"controller /evaluate_result returned non-JSON, polling again: {poll.text[:200]}")
            continue
        if not isinstance(data, dict):
            raise ControllerCallError(f"controller /evaluate_result returned {type(data).__name__}, not a JSON object: "
                                      f"{poll.text[:300]}", status=poll.status_code, body=poll.text)
        status = data.get("status")
        if status == "done":
            result = data.get("result") or {}
            if not isinstance(result, dict):
                raise ControllerCallError(f"controller /evaluate job {job_id} result is {type(result).__name__}, not a "
                                          f"JSON object: {poll.text[:300]}", status=poll.status_code, body=poll.text)
            return result
        if status in ("error", "unknown"):
            raise ControllerCallError(f"controller /evaluate job {job_id} {status}: {str(data.get('message', ''))[:300]}",
                                      body=json.dumps(data)[:500])


async def wait_ready(endpoint: Endpoint, *, timeout: float = 300.0, interval: float = 5.0) -> None:
    """Return once the controller answers ``READY_PATH``; TimeoutError after ``timeout``."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    last = None
    while loop.time() < deadline:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(endpoint.at(READY_PATH), headers=endpoint.headers,
                                            timeout=READY_CALL_TIMEOUT_S)
            if response.status_code == 200:
                return
            last = f"HTTP {response.status_code}"
        except httpx.HTTPError as e:
            last = repr(e)
        await asyncio.sleep(interval)
    raise TimeoutError(f"OSWorld controller {endpoint.url} not ready within {timeout}s (last: {last})")


async def require_auth(endpoint: Endpoint) -> None:
    """Raise unless the controller refuses a call without its token, as one whose image predates
    ``CONTROLLER_AUTH_TOKEN`` doesn't: its port may be reachable from the computer, and so by the agent."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(endpoint.at(READY_PATH), timeout=READY_CALL_TIMEOUT_S)
    except httpx.HTTPError as e:
        raise ControllerCallError(f"controller {READY_PATH} transport failure: {type(e).__name__}: {e}") from e
    if response.status_code != 401:
        raise ControllerCallError(
            f"the OSWorld controller at {endpoint.url} answered a call without its token with HTTP "
            f"{response.status_code}, not 401: its image doesn't enforce CONTROLLER_AUTH_TOKEN, so the agent could "
            f"call its grader; use a controller image that does", status=response.status_code)


def resolve_task_ref(ref: str) -> str:
    """An OSWorld-V2 task ref as the controller fetches it, with no credentials: an http(s) URL as it is, an object of
    the configured object store under its artifacts as a signed URL. Anything else raises, rather than hand the
    controller a URL it can't fetch or sign an object outside the artifacts."""
    if ref.startswith(("http://", "https://")):
        return ref
    config = get_config()
    store = config.get_object_store_at(ref)
    prefix = f"{config.get_artifact_key_prefix()}artifacts/"
    if not store.owns(ref) or not store.get_object_key(ref).startswith(prefix):
        raise ValueError(f"refusing to sign task ref {ref!r}: not an object under {prefix!r} in the object store")
    if (signed := store.signed_get_url(ref, expires_in=TASK_REF_TTL_S)) is None:
        raise ValueError(f"the object store can't sign {ref!r} for the controller to fetch")
    return signed


def normalize_score(score: Any) -> float:
    """A controller score as a float: a bool as 1 or 0, a number as it is, a dict as the mean of its numbers, else 0."""
    if isinstance(score, bool):
        return 1.0 if score else 0.0
    if isinstance(score, (int, float)):
        return float(score)
    if isinstance(score, dict):
        numbers = [float(v) for v in score.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        return sum(numbers) / len(numbers) if numbers else 0.0
    return 0.0
