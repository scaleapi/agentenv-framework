"""Moving bytes onto a Daytona sandbox through its file API, which needs no egress from the sandbox.

Uploads go as parts in ``/tmp`` (the target may be root-owned), joined by root and checked against the
local sha256.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import shlex
import uuid
from collections.abc import AsyncIterator
from typing import IO, TYPE_CHECKING

if TYPE_CHECKING:
    from agent_env.providers.sandbox_providers.daytona.sandbox import DaytonaSandbox

logger = logging.getLogger(__name__)

UPLOAD_CHUNK_BYTES = 4 * 1024 * 1024
_UPLOADS_IN_FLIGHT = 8
# The join rewrites the whole file, so retrying it after a dropped transport is safe.
_JOIN_RETRIES = 2


async def read_parts(source: IO[bytes]) -> AsyncIterator[bytes]:
    """``source`` in upload-sized parts, read off the event loop."""
    while part := await asyncio.to_thread(source.read, UPLOAD_CHUNK_BYTES):
        yield part


async def upload(sandbox: DaytonaSandbox, parts: AsyncIterator[bytes], vm_path: str) -> None:
    """Upload ``parts`` several at once, join them as root at ``vm_path`` and check its sha256."""
    stem = f"/tmp/_ae_upload_{uuid.uuid4().hex[:12]}"
    digest = hashlib.sha256()
    gate = asyncio.Semaphore(_UPLOADS_IN_FLIGHT)
    fs = (await sandbox.handle()).fs
    count = 0

    async def put(path: str, part: bytes) -> None:
        try:
            await fs.upload_file(part, path)
        finally:
            gate.release()

    try:
        try:
            async with asyncio.TaskGroup() as group:
                async for part in parts:
                    await gate.acquire()
                    digest.update(part)
                    group.create_task(put(f"{stem}.{count}", part))
                    count += 1
        except ExceptionGroup as failed:
            raise failed.exceptions[0] from failed
        quoted = shlex.quote(vm_path)
        last = max(count - 1, 0)
        join = f'for i in $(seq 0 {last}); do cat "{stem}.$i" || exit 1; done > {quoted} && sha256sum {quoted}'
        if count == 0:
            join = f": > {quoted} && sha256sum {quoted}"
        written = await sandbox.exec_script(join, max_retries=_JOIN_RETRIES)
    finally:
        if count:
            try:
                await sandbox.exec_script(f'for i in $(seq 0 {count - 1}); do rm -f "{stem}.$i"; done')
            except Exception as exc:  # noqa: BLE001 - leftover parts in /tmp go with the sandbox
                logger.warning("Removing upload parts %s.* failed (ignored): %s", stem, exc)
    actual = written.split()[0] if written.split() else ""
    if actual != digest.hexdigest():
        raise RuntimeError(
            f"{vm_path} in Daytona sandbox {sandbox.sandbox_id} does not match the upload: its sha256 is "
            f"{actual or 'missing'}, the upload's {digest.hexdigest()}"
        )
