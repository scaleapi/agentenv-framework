"""The data of an app that declares only the paths it keeps it in, under the session user's home: a load writes the files
it's given under them, an export returns the files there, and a reset puts back the copy its server kept when it first
started on this boot, which is the app as installed, since every boot starts from the computer's image."""
from __future__ import annotations

import asyncio
import shutil
from pathlib import Path, PurePosixPath
from typing import Optional

from ..transfers import WriteNamespaceGrant
from ..types import FilePart, uploaded_object_path
from .dataplane import file_parts, part_bytes

#: Where an app's server keeps its paths as installed, in the app's folder.
INITIAL = Path(".agentenv/initial")


def keep_initial(paths: list[str]) -> None:
    """Copy what's under each of ``paths`` of the home into ``INITIAL``, unless this boot kept a copy already."""
    if INITIAL.exists():
        return
    staging = INITIAL.with_name(f"{INITIAL.name}.partial")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    for path in paths:
        _copy(Path.home() / path, staging / path)
    staging.rename(INITIAL)


async def add(parts: list, paths: list[str]) -> None:
    """Write each file part at the path it names, which must be under one of ``paths``."""
    files = {}
    for part in parts:
        path = uploaded_object_path(part) if isinstance(part, FilePart) else None
        if path is None:
            raise ValueError(f"data/add on this app takes file parts naming their path, got a {part.kind} part")
        if not any(PurePosixPath(path).is_relative_to(owned) for owned in paths):
            raise ValueError(f"{path!r} is under none of this app's paths: {', '.join(paths)}")
        files[path] = await part_bytes(part)
    await asyncio.to_thread(_write, Path.home(), files)


async def get(paths: list[str], write_namespace: Optional[WriteNamespaceGrant]) -> list:
    """A part for each file under ``paths``."""
    return await file_parts(Path.home(), paths, write_namespace)


async def reset(paths: list[str]) -> None:
    """Put each of ``paths`` back as ``keep_initial`` kept it, or remove it where it wasn't there then."""
    await asyncio.to_thread(_restore, Path.home(), paths)


def _write(home: Path, files: dict[str, bytes]) -> None:
    for path, data in files.items():
        target = home / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def _restore(home: Path, paths: list[str]) -> None:
    for path in paths:
        target = home / path
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink(missing_ok=True)
        _copy(INITIAL / path, target)


def _copy(source: Path, target: Path) -> None:
    if source.is_dir() and not source.is_symlink():
        shutil.copytree(source, target, symlinks=True)
    elif source.exists() or source.is_symlink():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target, follow_symlinks=False)
