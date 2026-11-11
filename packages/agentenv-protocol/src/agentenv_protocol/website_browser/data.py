"""What a website browser's data plane moves, as files by their path under the session user's home: its typed state in
``STATE_FILE``, its profile without caches or locks, its downloads, and files for an agent to upload."""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Optional

from ..computer.app_data import write_within
from ..computer.dataplane import part_bytes
from ..types import DataPart, FilePart, uploaded_object_path
from .control import DOWNLOADS, UPLOADS

#: The browser's typed state, as ``data/get`` returns it beside its files and ``data/add`` reads it back.
STATE_FILE = "browser-state.json"
#: The browser's log, as ``data/get`` returns it beside its state; ``data/add`` passes over it, since a log isn't
#: loaded back.
LOG_FILE = "browser-log.jsonl"
#: What a copy of the profile leaves out: caches, which the browser rebuilds, and the locks a running browser holds,
#: which would keep another from opening the copy.
CACHES = frozenset({"Cache", "Code Cache", "GPUCache", "ShaderCache", "GrShaderCache", "GraphiteDawnCache",
                    "DawnCache", "DawnGraphiteCache", "DawnWebGPUCache", "CacheStorage", "ScriptCache", "Crashpad",
                    "component_crx_cache"})
LOCKS = frozenset({"SingletonLock", "SingletonCookie", "SingletonSocket"})
#: A download the browser is still writing.
PARTIAL = ".crdownload"


@dataclass
class Load:
    """A ``data/add``'s parts sorted by what each is: the typed state, the profile's files, and files for the folders
    the browser downloads into and uploads from, each by its path under the home."""
    state: Optional[dict] = None
    profile: dict[str, bytes] = field(default_factory=dict)
    files: dict[str, bytes] = field(default_factory=dict)


async def sorted_load(parts: list, profile: str) -> Load:
    """``parts`` sorted into a ``Load``; ValueError, before anything is applied, for a part that's none of them."""
    load = Load()
    for part in parts:
        if isinstance(part, DataPart):
            if load.state is not None:
                raise ValueError(f"data/add takes the browser's state once, as a data part or as {STATE_FILE}")
            load.state = part.data
            continue
        path = uploaded_object_path(part) if isinstance(part, FilePart) else None
        if path is None:
            raise ValueError(f"data/add on a website browser takes its state as a data part, and files naming their "
                             f"path, not a {part.kind} part")
        under = PurePosixPath(path)
        if under in (PurePosixPath(profile), PurePosixPath(DOWNLOADS), PurePosixPath(UPLOADS)):
            raise ValueError(f"{path!r} is a folder of the browser's, not a file in it")
        if path == LOG_FILE:
            continue
        if path == STATE_FILE:
            if load.state is not None:
                raise ValueError(f"data/add takes the browser's state once, as a data part or as {STATE_FILE}")
            load.state = _json(await part_bytes(part))
        elif under.is_relative_to(profile):
            load.profile[path] = await part_bytes(part)
        elif under.is_relative_to(DOWNLOADS) or under.is_relative_to(UPLOADS):
            load.files[path] = await part_bytes(part)
        else:
            raise ValueError(f"{path!r} is none of the browser's: its state, {STATE_FILE}, and files under its "
                             f"profile, {profile}, ~/{DOWNLOADS} and ~/{UPLOADS}")
    return load


def _json(raw: bytes) -> Any:
    try:
        return json.loads(raw)
    except ValueError as e:
        raise ValueError(f"{STATE_FILE} isn't JSON: {e}") from None


def stage_profile(home: Path, profile: str, staging: Path) -> None:
    """A copy of the profile at its path under ``staging``, without caches or locks: each database with its journal
    or log, copied in one pass, as the browser's own recovery after a crash would find them. A file or folder the
    browser removes meanwhile, as it does a journal once its write ends, is left out."""
    source = home / profile
    for folder, folders, names in os.walk(source):
        links = [name for name in folders if Path(folder, name).is_symlink()]
        folders[:] = [name for name in folders if name not in CACHES and name not in links]
        target = staging / profile / Path(folder).relative_to(source)
        target.mkdir(parents=True, exist_ok=True)
        for name in [*names, *links]:
            if name not in LOCKS:
                _copied(Path(folder, name), target / name)


def _copied(source: Path, target: Path) -> None:
    try:
        if source.is_symlink():
            target.symlink_to(os.readlink(source))
        else:
            shutil.copy2(source, target)
    except FileNotFoundError:
        pass


def finished_downloads(home: Path) -> list[str]:
    """The path under the home of each file in the downloads folder the browser has finished writing."""
    folder = home / DOWNLOADS
    if not folder.is_dir():
        return []
    return sorted(f"{DOWNLOADS}/{path.relative_to(folder).as_posix()}" for path in folder.rglob("*")
                  if path.is_file() and not path.name.endswith(PARTIAL))


def restore_profile(home: Path, profile: str, files: dict[str, bytes]) -> None:
    """The profile made exactly ``files``, by their paths, with the browser stopped."""
    target = home / profile
    shutil.rmtree(target, ignore_errors=True)
    write_within(home, files, [profile])


def empty(home: Path, *folders: str) -> None:
    """Each folder under the home emptied."""
    for name in folders:
        folder = home / name
        if folder.is_dir():
            for child in folder.iterdir():
                if child.is_dir() and not child.is_symlink():
                    shutil.rmtree(child)
                else:
                    child.unlink()
