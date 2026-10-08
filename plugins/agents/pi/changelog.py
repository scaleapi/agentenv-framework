"""Tool-call changelog capture and replay for the snapshot extension's changelog feature.

After each tool call the capture uploads one increment, ``NNNNNN.tar`` named by the tool call's zero-based
position: the files under the roots that changed since the previous increment (``files/<absolute path>``),
the paths that disappeared, and the conversation's pi session as it stood (``session.jsonl``). A failed
upload leaves its changes for the next increment, so positions may be sparse, as the protocol allows; the
last failed one is retried by ``flush`` when the task ends, since no later increment may follow it.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import tarfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from agentenv_protocol.a2a_agent import NamespaceUploader, WriteNamespaceGrant

META = "meta.json"
SESSION = "session.jsonl"
FILES = "files/"
logger = logging.getLogger(__name__)
_SKIPPED_DIRS = frozenset({"__pycache__", ".git"})

Signature = tuple[int, int, int]


def scan(roots: Iterable[str]) -> dict[str, Signature]:
    """Every regular file under ``roots`` with its (mtime, size, mode)."""
    found: dict[str, Signature] = {}
    for root in roots:
        for directory, subdirectories, files in os.walk(root):
            subdirectories[:] = [name for name in subdirectories if name not in _SKIPPED_DIRS]
            for name in files:
                path = os.path.join(directory, name)
                try:
                    stat = os.lstat(path)
                except FileNotFoundError:
                    continue
                if os.path.isfile(path) and not os.path.islink(path):
                    found[path] = (stat.st_mtime_ns, stat.st_size, stat.st_mode & 0o7777)
    return found


def increment(
    changed: Sequence[str], deleted: Sequence[str], roots: Sequence[str], session: Path | None, meta: dict
) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for path in changed:
            try:
                archive.add(path, arcname=FILES + path.lstrip("/"), recursive=False)
            except FileNotFoundError:
                continue
        if session is not None and session.is_file():
            archive.add(session, arcname=SESSION)
        body = json.dumps({**meta, "roots": list(roots), "deleted": list(deleted)}).encode()
        info = tarfile.TarInfo(META)
        info.size = len(body)
        archive.addfile(info, io.BytesIO(body))
    return buffer.getvalue()


@dataclass
class Applied:
    session: bytes | None


def apply(path: Path, roots: Sequence[str]) -> Applied:
    """Replay one increment onto the filesystem. Every path it touches must resolve, symlinks followed, under
    one of ``roots``, the replaying agent's own: the roots an increment names are not trusted."""
    allowed = [Path(os.path.realpath(root)) for root in roots]

    def within(target: Path) -> bool:
        resolved = Path(os.path.realpath(target))
        return any(resolved.is_relative_to(root) for root in allowed)

    with tarfile.open(path) as archive:
        meta = json.load(archive.extractfile(META))
        for name in meta["deleted"]:
            target = Path(os.path.normpath("/" + name.lstrip("/")))
            if not within(target.parent):
                raise ValueError(f"increment deletes outside the agent's roots: {target}")
            target.unlink(missing_ok=True)
        session = None
        for member in archive.getmembers():
            if member.name == SESSION:
                session = archive.extractfile(member).read()
            elif member.name.startswith(FILES) and member.isfile():
                target = Path(os.path.normpath("/" + member.name.removeprefix(FILES)))
                if not within(target):
                    raise ValueError(f"increment writes outside the agent's roots: {target}")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.extractfile(member).read())
                target.chmod(member.mode & 0o7777)
    return Applied(session=session)


class ChangelogCapture:
    def __init__(self, grant: WriteNamespaceGrant, roots: Sequence[str]) -> None:
        self.roots = list(roots)
        self._uploader = NamespaceUploader(grant)
        self._lock = asyncio.Lock()
        self._position = 0
        self._state: dict[str, Signature] = {}
        self._pending: tuple[int, dict] | None = None

    async def start(self) -> None:
        self._state = await asyncio.to_thread(scan, self.roots)

    async def capture(self, session: Path | None, *, context_id: str, session_id: str) -> None:
        """Upload the increment for the next tool call. A failure is logged and left pending, not raised: the
        scanned state only advances on success, so a later increment, or ``flush``, carries the changes."""
        async with self._lock:
            position = self._position
            self._position += 1
            meta = {"position": position, "context_id": context_id, "session_id": session_id}
            self._pending = (position, meta)
            await self._upload(session)

    async def flush(self, session: Path | None) -> bool:
        """Retry the pending increment, if any; whether every increment so far is uploaded."""
        async with self._lock:
            if self._pending is not None:
                await self._upload(session)
            return self._pending is None

    async def _upload(self, session: Path | None) -> None:
        position, meta = self._pending
        try:
            current = await asyncio.to_thread(scan, self.roots)
            changed = [path for path, signature in current.items() if self._state.get(path) != signature]
            deleted = [path for path in self._state if path not in current]
            body = await asyncio.to_thread(increment, changed, deleted, self.roots, session, meta)
            await self._uploader.upload(f"{position:06d}.tar", body)
        except Exception:  # noqa: BLE001 -- a lost increment must not fail the tool call; flush reports it
            logger.exception("changelog increment %06d was not uploaded", position)
            return
        self._state = current
        self._pending = None
