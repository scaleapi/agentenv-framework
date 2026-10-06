"""On-disk state directories for the local (no-infra) stores."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path

from agent_env.config.errors import ConfigError
from agent_env.config.paths import state_root

try:
    import fcntl
except ImportError:  # Windows: writers of one id aren't serialized
    fcntl = None


def ensure_state_dir(path: Path) -> None:
    """Create ``path`` if missing. A directory this creates is private to the user (``0o700``)
    and gets a ``.gitignore`` of ``*`` so generated state is never committed; a directory that
    already existed is left as is. A directory that cannot be created is a ``ConfigError``."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.mkdir(mode=0o700)
        except FileExistsError:
            if path.is_dir():
                return
            raise
        (path / ".gitignore").write_text("*\n")
    except OSError as e:
        raise ConfigError(
            f"cannot create the local store directory {path} ({e}); set XDG_STATE_HOME to a "
            "writable directory, or point the store somewhere writable in config.toml"
        ) from e


@contextmanager
def holding_locks(ids: Iterable[str], on_wait: Callable[[], None] | None = None) -> Iterator[None]:
    """Hold a lock on each of ``ids`` in the per-user state root, so another process writing any of them waits
    here, calling ``on_wait`` first. The locks are taken in one order, so two holders can't each hold one the
    other waits for."""
    if fcntl is None:
        yield
        return
    ensure_state_dir(state_root() / "locks")
    fds, waited = [], False
    try:
        for path in sorted({_lock_path(id) for id in ids}):
            fds.append(os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600))
            try:
                fcntl.flock(fds[-1], fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                if on_wait is not None and not waited:
                    on_wait()
                    waited = True
                fcntl.flock(fds[-1], fcntl.LOCK_EX)
        yield
    finally:
        for fd in fds:
            os.close(fd)


def _lock_path(id: str) -> Path:
    return state_root() / "locks" / f"entity-{hashlib.sha256(id.encode()).hexdigest()[:16]}.lock"
