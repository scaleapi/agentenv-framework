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
from collections import deque
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
        self._pending: deque[tuple[str, bytes]] = deque()

    async def start(self) -> None:
        self._state = await asyncio.to_thread(scan, self.roots)

    async def capture(self, session: Path | None, *, context_id: str, session_id: str) -> None:
        """Build the increment for the next tool call and upload it with any still pending, in order. A failed
        upload is logged and kept, not raised, for the next capture or ``flush``."""
        async with self._lock:
            current = await asyncio.to_thread(scan, self.roots)
            changed = [path for path, signature in current.items() if self._state.get(path) != signature]
            deleted = [path for path in self._state if path not in current]
            meta = {"position": self._position, "context_id": context_id, "session_id": session_id}
            body = await asyncio.to_thread(increment, changed, deleted, self.roots, session, meta)
            self._pending.append((f"{self._position:06d}.tar", body))
            self._position += 1
            self._state = current
            await self._drain()

    async def flush(self) -> bool:
        """Retry the pending increments; whether every increment so far is uploaded."""
        async with self._lock:
            await self._drain()
            return not self._pending

    async def _drain(self) -> None:
        while self._pending:
            name, body = self._pending[0]
            try:
                await self._uploader.upload(name, body)
            except Exception:  # noqa: BLE001 -- a lost increment must not fail the tool call; flush reports it
                logger.exception("changelog increment %s was not uploaded", name)
                return
            self._pending.popleft()
