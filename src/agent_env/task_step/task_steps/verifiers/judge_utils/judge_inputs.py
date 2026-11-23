"""What a judge was sent, stored by its sha256 under ``judge-inputs/`` in the object store, so a retried request's input
or a file another run staged is stored once and a reader checks it against the key it's stored under. A judge run's
uploads run in the background, off its judging's path (``Uploads``); its record waits for them once, before its grade is
written (``JudgeUsage.settle``), when an upload that failed or hadn't finished is marked ``input_error``, which never
fails the grade."""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import logging
from typing import Any, Optional

from agent_env.config import get_config
from agent_env.task_step.thread_work import finish_on_thread

logger = logging.getLogger(__name__)

INPUTS_PREFIX = "judge-inputs"
#: How many of a judge run's uploads run at once, and how long its grade waits for those still running.
UPLOAD_CONCURRENCY = 16
SETTLE_SECONDS = 120.0


class Uploads:
    """A judge run's uploads: each distinct body put once, in the background, ``UPLOAD_CONCURRENCY`` at a time."""

    def __init__(self) -> None:
        self._tasks: dict[str, asyncio.Task] = {}
        self._late: set[str] = set()
        self._gate = asyncio.Semaphore(UPLOAD_CONCURRENCY)

    def put(self, data: bytes, content_type: str, digest: Optional[str] = None) -> dict[str, Any]:
        """``data`` stored under ``judge-inputs/<sha256>``, in the background and once however often the run sends it:
        its reference (``object_url``, ``bytes``, ``sha256``) at once."""
        digest = digest or hashlib.sha256(data).hexdigest()
        config = get_config()
        store, key = config.get_object_store(), f"{config.get_artifact_key_prefix()}{INPUTS_PREFIX}/{digest}"
        if digest not in self._tasks:
            task = asyncio.ensure_future(self._put(store, key, data, content_type))
            task.add_done_callback(_logged)
            self._tasks[digest] = task
        return {"object_url": store.object_url(key), "bytes": len(data), "sha256": digest}

    async def put_json(self, body: Any) -> dict[str, Any]:
        """``body`` as JSON (``put``), encoded and hashed off the event loop, since a call's input may be tens of MB."""
        data, digest = await asyncio.to_thread(_encoded, body)
        return self.put(data, "application/json", digest)

    @staticmethod
    def deadline() -> float:
        """When a run that settles now stops waiting for its uploads."""
        return asyncio.get_running_loop().time() + SETTLE_SECONDS

    async def wait(self, deadline: float) -> None:
        """Waits for the uploads still running until ``deadline``, then cancels those that haven't finished."""
        pending = {digest: task for digest, task in self._tasks.items() if not task.done()}
        if pending:
            await asyncio.wait(pending.values(), timeout=max(0.0, deadline - asyncio.get_running_loop().time()))
        for digest, task in pending.items():
            if not task.done():
                task.cancel()
                self._late.add(digest)

    def failed(self, digest: Optional[str]) -> Optional[str]:
        """Why the upload of ``digest`` didn't store it, once ``wait`` has returned; None when it did, or the run
        uploaded no such thing."""
        task = self._tasks.get(digest) if digest else None
        if task is None:
            return None
        if digest in self._late:
            return f"TimeoutError: still uploading after {SETTLE_SECONDS:.0f}s"
        if task.cancelled():
            return "CancelledError: the upload was cancelled"
        exc = task.exception()
        return f"{type(exc).__name__}: {exc}"[:300] if exc is not None else None

    async def _put(self, store: Any, key: str, data: bytes, content_type: str) -> None:
        async with self._gate:
            await finish_on_thread(functools.partial(_put_once, store, key, data, content_type),
                                   "Storing a judge's input")


class StagingRecorder:
    """A judge's sandbox, every file written into it kept in its run's manifest (``staged``): a file written as text
    stored by its sha256 too (``Uploads.put``), one copied from a stored object named by that object's URL."""

    def __init__(self, sandbox: Any, staged: list[dict[str, Any]], uploads: Uploads) -> None:
        self._sandbox, self._staged, self._uploads = sandbox, staged, uploads

    def __getattr__(self, name: str) -> Any:
        return getattr(self._sandbox, name)

    async def write_file_from_text(self, text: str, path: str, **kwargs: Any) -> Any:
        written = await self._sandbox.write_file_from_text(text, path, **kwargs)
        self._staged.append({"path": path, **self._uploads.put(text.encode(), "text/plain; charset=utf-8")})
        return written

    async def write_file_from_object(self, object_url: str, path: str, **kwargs: Any) -> Any:
        written = await self._sandbox.write_file_from_object(object_url, path, **kwargs)
        self._staged.append({"path": path, "object_url": object_url})
        return written


def _encoded(body: Any) -> tuple[bytes, str]:
    data = json.dumps(body, ensure_ascii=False, sort_keys=True).encode()
    return data, hashlib.sha256(data).hexdigest()


def _put_once(store: Any, key: str, data: bytes, content_type: str) -> None:
    if not store.exists(key):
        store.put(key, data, content_type=content_type, allow_overwrite=True)


def _logged(task: asyncio.Task) -> None:
    if not task.cancelled() and (exc := task.exception()) is not None:
        logger.warning("a judge's input wasn't stored: %s: %s", type(exc).__name__, exc)
