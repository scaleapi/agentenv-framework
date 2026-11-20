"""The media files of an agent's atif/1 copies, spooled to its disk as the adapter names them: one file per sha256,
held by the contexts whose copies reference it, until a push confirms it for a context or that context goes. The worker
pulls them at each turn's end (``TRAJECTORY_V1.media``), so the bytes never wait in memory."""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path


class MediaSpool:
    """Each context's media files on disk. Safe to fill from the agent's thread while the loop pushes them; the folder
    is made on the first file and removed by ``close``."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._folder: Path | None = None
        #: Each context's held paths, ``media/<sha256>.<ext>``.
        self._held: dict[str, set[str]] = {}

    def sink(self, context_id: str) -> Callable[[str, bytes], None]:
        """Where a context's copy writes each media file once, as its builder names it."""

        def write(path: str, data: bytes) -> None:
            with self._lock:
                file = self._file(path)
                if not file.exists():
                    part = file.with_name(f"{file.name}.{os.getpid()}.part")
                    part.write_bytes(data)
                    os.replace(part, file)
                self._held.setdefault(context_id, set()).add(path)

        return write

    def held(self, context_id: str, path: str) -> Path | None:
        """The file a context holds at ``path``, or None when it holds none there."""
        with self._lock:
            return self._file(path) if path in self._held.get(context_id, ()) else None

    def release(self, context_id: str, path: str) -> None:
        """``path`` is the context's no longer: a push confirmed it. The file goes once no context holds it."""
        with self._lock:
            held = self._held.get(context_id)
            if held is not None:
                held.discard(path)
                if not held:
                    del self._held[context_id]
            self._remove_unheld(path)

    def drop(self, context_id: str) -> None:
        """The context is gone, evicted or restored over: its files go, but those another context still holds."""
        with self._lock:
            for path in self._held.pop(context_id, set()):
                self._remove_unheld(path)

    def close(self) -> None:
        with self._lock:
            if self._folder is not None:
                shutil.rmtree(self._folder, ignore_errors=True)
            self._folder = None
            self._held.clear()

    def _file(self, path: str) -> Path:
        if self._folder is None:
            self._folder = Path(tempfile.mkdtemp(prefix="agentenv-atif-media-"))
        return self._folder / path.removeprefix("media/")

    def _remove_unheld(self, path: str) -> None:
        if self._folder is not None and not any(path in held for held in self._held.values()):
            (self._folder / path.removeprefix("media/")).unlink(missing_ok=True)
