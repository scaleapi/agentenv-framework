"""The media files of an agent's atif/1 copies, spooled to its disk as the adapter names them: one file per sha256,
held by the contexts whose copies reference it, until a push uploads it for a context or that context goes. The worker
pulls them at each turn's end (``TRAJECTORY_V1.media``), so the bytes never wait in memory."""

from __future__ import annotations

import asyncio
import itertools
import logging
import os
import shutil
import tempfile
import threading
from collections.abc import Callable, Mapping
from pathlib import Path

from ..atif import media_file
from ..transfers import TransferError, Uploaded, WriteObject, sha256_file, upload
from .extensions import TrajectoryMediaResponse

logger = logging.getLogger(__name__)

#: How many of a push's files upload at once.
MAX_PARALLEL_MEDIA_PUSHES = 8


class MediaSpool:
    """Each context's media files on disk. Safe to fill from the agent's thread while the loop pushes them; the folder
    is made on the first file and removed by ``close``. Each hold has its own number, so an upload that outlives its
    context's copy (a restore, an eviction) releases only the hold it began with, never a later copy's. A file that
    can't be written, on a full disk say, is left out: a push reports it missing, and the copy goes on."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._folder: Path | None = None
        #: Each context's held paths, ``media/<sha256>.<ext>``, each with its hold's number.
        self._held: dict[str, dict[str, int]] = {}
        self._holds = itertools.count(1)

    @property
    def folder(self) -> Path | None:
        """Where the files are, once the first one is written."""
        return self._folder

    def sink(self, context_id: str) -> Callable[[str, bytes], None]:
        """Where a context's copy writes each media file once, as its builder names it."""

        def write(path: str, data: bytes) -> None:
            with self._lock:
                try:
                    file = self._file(path)
                    if not file.exists():
                        part = file.with_name(f"{file.name}.{os.getpid()}.part")
                        try:
                            part.write_bytes(data)
                            os.replace(part, file)
                        finally:
                            part.unlink(missing_ok=True)
                except OSError as exc:
                    logger.warning("the atif/1 copy of context %s keeps no file for %s, which a push reports "
                                   "missing: %s", context_id, path, exc)
                    return
                self._held.setdefault(context_id, {}).setdefault(path, next(self._holds))

        return write

    def held(self, context_id: str, path: str) -> Path | None:
        """The file a context holds at ``path``, or None when it holds none there."""
        hold = self._hold(context_id, path)
        return None if hold is None else hold[0]

    async def push(self, context_id: str, objects: Mapping[str, WriteObject]) -> TrajectoryMediaResponse:
        """Upload the files a context holds at ``objects``' paths, each under its grant, ``MAX_PARALLEL_MEDIA_PUSHES``
        at a time. Each file's sha256 is checked against its name before its bytes leave, and the upload's after; a file
        uploaded is the context's no longer. A path the context holds no file for is ``missing``, an upload that fails
        is ``failed``, and neither stops the rest."""
        uploaded: dict[str, Uploaded] = {}
        missing: list[str] = []
        failed: dict[str, str] = {}
        gate = asyncio.Semaphore(MAX_PARALLEL_MEDIA_PUSHES)

        async def push_one(path: str, target: WriteObject) -> None:
            held = self._hold(context_id, path)
            if held is None:
                missing.append(path)
                return
            file, hold = held
            async with gate:
                try:
                    digest = await asyncio.to_thread(sha256_file, file)
                except OSError as exc:
                    failed[path] = f"the spooled file can't be read: {exc.strerror}"
                    return
                if digest != media_file(path):
                    failed[path] = f"the spooled file's sha256 is {digest}"
                    return
                try:
                    done = await upload(target, file)
                except TransferError as exc:
                    failed[path] = f"{exc.code}: {exc}"
                    return
            if done.sha256 is not None and done.sha256 != digest:
                failed[path] = f"the upload's sha256 is {done.sha256}"
                return
            uploaded[path] = done
            self._release(context_id, path, hold)

        await asyncio.gather(*(push_one(path, target) for path, target in objects.items()))
        return TrajectoryMediaResponse(objects=uploaded, missing=sorted(missing), failed=failed)

    def drop(self, context_id: str) -> None:
        """The context is gone, evicted or restored over: its files go, but those another context still holds."""
        with self._lock:
            for path in self._held.pop(context_id, {}):
                self._remove_unheld(path)

    def close(self) -> None:
        with self._lock:
            if self._folder is not None:
                shutil.rmtree(self._folder, ignore_errors=True)
            self._folder = None
            self._held.clear()

    def _hold(self, context_id: str, path: str) -> tuple[Path, int] | None:
        with self._lock:
            hold = self._held.get(context_id, {}).get(path)
            return None if hold is None else (self._file(path), hold)

    def _release(self, context_id: str, path: str, hold: int) -> None:
        """Hold ``hold`` on ``path`` is the context's no longer. A later hold stays, and the file goes once no context
        holds it."""
        with self._lock:
            held = self._held.get(context_id)
            if held is None or held.get(path) != hold:
                return
            del held[path]
            if not held:
                del self._held[context_id]
            self._remove_unheld(path)

    def _file(self, path: str) -> Path:
        if self._folder is None:
            self._folder = Path(tempfile.mkdtemp(prefix="agentenv-atif-media-"))
        return self._folder / path.removeprefix("media/")

    def _remove_unheld(self, path: str) -> None:
        if self._folder is not None and not any(path in held for held in self._held.values()):
            (self._folder / path.removeprefix("media/")).unlink(missing_ok=True)
