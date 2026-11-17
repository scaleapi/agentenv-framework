"""The package locks a computer's compose resolves, and the files they name, shared between the fresh local stores the
tests compose in, so a lock resolved once in a run, or restored from an earlier run's cache, isn't resolved from the
network again.

Everything under the packages prefix is content-addressed, a lock by a digest of its inputs and a file by its sha256, so
any store may take any of it. The rest of an object store isn't: its keys follow entity ids and versions. The shared
folder is ``AGENTENV_TEST_PACKAGES``, which tst/conftest.py sets for the run when it's unset; it mirrors the store's
layout under the prefix, content types included."""

from __future__ import annotations

import errno
import os
import shutil
from pathlib import Path

from agent_env.config import get_config
from agent_env.env.envs.computer.utils.packages import PREFIX, Lock
from agent_env.store.object_store import LocalFilesystemObjectStore

ENV_VAR = "AGENTENV_TEST_PACKAGES"


def seed() -> None:
    """Give the configured local object store every shared lock whose files are all shared: its files, then the lock."""
    if (shared := _shared()) is None:
        return
    store = _store()
    for lock in sorted((shared.packages / "locks").glob("*.json")):
        files = _files(lock)
        if files is None or not all((shared.packages / "files" / sha256).is_file() for sha256 in files):
            continue
        for sha256 in files:
            _link(shared.packages / "files" / sha256, store.packages / "files" / sha256)
        _link_lock(lock.name, shared, store)


def keep() -> None:
    """Share what the configured local object store holds under the prefix: its files first, then its locks, so the
    shared folder never has a lock without its files."""
    if (shared := _shared()) is None:
        return
    store = _store()
    for path in sorted((store.packages / "files").glob("*")):
        _link(path, shared.packages / "files" / path.name)
    for lock in sorted((store.packages / "locks").glob("*.json")):
        _link_lock(lock.name, store, shared)


class _Tree:
    """Where a folder keeps the objects under the prefix, and their content types (the local store's
    ``.agentenv-meta``)."""

    def __init__(self, root: Path, prefix: str):
        self.packages, self.meta = root / prefix, root / ".agentenv-meta" / prefix


def _shared() -> _Tree | None:
    folder = os.environ.get(ENV_VAR)
    return _Tree(Path(folder), PREFIX) if folder else None


def _store() -> _Tree:
    config = get_config()
    store = config.get_object_store()
    if not isinstance(store, LocalFilesystemObjectStore):
        raise TypeError(f"packages are shared with a local object store, not {type(store).__name__}")
    return _Tree(store.root, f"{config.get_artifact_key_prefix()}{PREFIX}")


def _files(lock: Path) -> list[str] | None:
    """The sha256 of every file ``lock`` names; None for one in a shape this version doesn't read."""
    try:
        parsed = Lock.model_validate_json(lock.read_bytes())
    except ValueError:
        return None
    named = [*parsed.apt, *(f for files in parsed.wheels.values() for f in files),
             *(f for files in parsed.downloads.values() for f in files)]
    return [f.sha256 for f in named]


def _link_lock(name: str, source: _Tree, target: _Tree) -> None:
    if (meta := source.meta / "locks" / name).is_file():
        _link(meta, target.meta / "locks" / name)
    _link(source.packages / "locks" / name, target.packages / "locks" / name)


def _link(source: Path, target: Path) -> None:
    """``source`` at ``target`` unless something is there: a hard link, safe as the store never writes an object in
    place, or a copy where the two folders can't share one, keeping its modification time, which the store's record of
    an object's content type names it by."""
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except FileExistsError:
        pass
    except OSError as e:
        if e.errno not in (errno.EXDEV, errno.EPERM, errno.EMLINK, errno.ENOTSUP, errno.EOPNOTSUPP):
            raise
        staged = target.with_name(f".{target.name}.{os.getpid()}")
        shutil.copy2(source, staged)
        os.replace(staged, target)
