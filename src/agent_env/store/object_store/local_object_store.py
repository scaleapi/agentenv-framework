"""Local filesystem implementation of the ObjectStore interface (no infra)."""

from __future__ import annotations

import contextlib
import errno
import hashlib
import ipaddress
import json
import logging
import os
import secrets
import shutil
import tempfile
import time
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO

from agentenv_protocol.transfers import HttpGetGrant, HttpPostPolicyGrant, HttpPutGrant

from agent_env.config.paths import state_root
from agent_env.store.base import ObjectAlreadyExistsError, ObjectNotFoundError

if os.name != "nt":
    import fcntl
from agent_env.store.local_state import ensure_state_dir
from agent_env.store.object_store.local_grants.server import grant_server
from agent_env.store.object_store.local_grants.tls import check_local_host
from agent_env.store.object_store.local_grants.tokens import Op
from agent_env.store.object_store.object_store import (
    DEFAULT_CONTENT_TYPE,
    DEFAULT_GRANT_LIFETIME_SECONDS,
    ObjectMetadata,
    ObjectStore,
    UploadPolicy,
    grant_lifetime,
)

logger = logging.getLogger(__name__)

_META_DIR = ".agentenv-meta"  # each object's content type, at the object's own key
_STAGING_DIR = ".agentenv-tmp"  # files being written, inside the root so a rename into place is atomic
_RESERVED = (".gitignore", _META_DIR, _STAGING_DIR)
# A filesystem without hard links: a no-overwrite write relies on the key's lock instead.
_NO_HARD_LINKS = frozenset({errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EXDEV, errno.EMLINK})
_LOCK_STRIPES = 256  # keys share this many lock files, so locking leaves no file behind per key
_STAGED_PREFIX = "staged-"
_STAGED_GRACE_SECONDS = 60  # a staged file younger than this may not be locked by its writer yet
_GRANT_MODES = ("auto", "off")
_LOCAL_SANDBOX_TYPE = "local"  # LocalSandbox.type: containers on this host's Docker, which trust the local CA

logger = logging.getLogger(__name__)

_META_DIR = ".agentenv-meta"  # each object's content type, at the object's own key
_STAGING_DIR = ".agentenv-tmp"  # files being written, inside the root so a rename into place is atomic
_RESERVED = (".gitignore", _META_DIR, _STAGING_DIR)
# A filesystem without hard links: a no-overwrite write relies on the key's lock instead.
_NO_HARD_LINKS = frozenset({errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EXDEV, errno.EMLINK})
_LOCK_STRIPES = 256  # keys share this many lock files, so locking leaves no file behind per key
_STAGED_PREFIX = "staged-"
_STAGED_GRACE_SECONDS = 60  # a staged file younger than this may not be locked by its writer yet


class LocalFilesystemObjectStore(ObjectStore):
    """ObjectStore backed by a local directory tree, ``<state root>/object_store`` unless ``root`` is given.

    A write lands whole or not at all, and keeps any content type but the default. The root-level ``.gitignore``,
    ``.agentenv-meta`` and ``.agentenv-tmp`` keys are reserved for the store's own files.

    With ``grants = "auto"`` (the default) it hands agents on the local sandbox provider HTTPS transfer grants,
    served by a server in this process, started by the first grant: it listens on ``grant_bind_host`` (by default
    loopback, or the Docker bridge gateway on Linux) and grant URLs name ``grant_advertise_host`` (by default
    ``host.docker.internal``), which must be a local name or a loopback or private address. A grant works until it
    expires or this process exits. ``grants = "off"`` hands out none, so agents get the forms that carry no grant."""

    def __init__(
        self,
        root: str | None = None,
        *,
        grants: str = "auto",
        grant_bind_host: str | None = None,
        grant_advertise_host: str | None = None,
        grant_lifetime_seconds: int = DEFAULT_GRANT_LIFETIME_SECONDS,
    ) -> None:
        if grants not in _GRANT_MODES:
            raise ValueError(f"grants must be one of {_GRANT_MODES}, got {grants!r}")
        if grant_bind_host is not None:
            try:
                ipaddress.ip_address(grant_bind_host)
            except ValueError:
                raise ValueError(f"grant_bind_host must be an IP address, got {grant_bind_host!r}") from None
        if grant_advertise_host is not None:
            check_local_host(grant_advertise_host)
        self._root = Path(root) if root is not None else state_root() / "object_store"
        self.supports_transfer_grants = grants == "auto"
        self.grant_lifetime_seconds = grant_lifetime(grant_lifetime_seconds)
        self._grant_bind_host = grant_bind_host
        self._grant_advertise_host = grant_advertise_host
        self._swept = False

    @property
    def root(self) -> Path:
        return self._root

    def put(
        self,
        key: str,
        data: bytes,
        content_type: str = DEFAULT_CONTENT_TYPE,
        allow_overwrite: bool = False,
    ) -> str:
        path = self._resolve(key)
        if path.exists() and not allow_overwrite:
            raise ObjectAlreadyExistsError(f"Object already exists at {path}.")
        with self._staged() as staged:
            staged.write_bytes(data)
            self._commit(staged, path, content_type, allow_overwrite=allow_overwrite)
        return self._to_url(path)

    def put_file(self, key: str, file_path: str, content_type: str = DEFAULT_CONTENT_TYPE) -> str:
        path = self._resolve(key)
        if path.exists():
            raise ObjectAlreadyExistsError(f"Object already exists at {path}.")
        with self._staged() as staged:
            shutil.copyfile(file_path, staged)
            self._commit(staged, path, content_type, allow_overwrite=False)
        return self._to_url(path)

    def put_file_at(self, object_url: str, file_path: str, content_type: str = DEFAULT_CONTENT_TYPE) -> str:
        return self.put_file(self.get_object_key(object_url), file_path, content_type)

    def get(self, object_url: str) -> bytes:
        with _missing_as_not_found(object_url):
            return Path(self._from_url(object_url)).read_bytes()

    def download_to_file(self, object_url: str, dest_path: str) -> None:
        source = self._from_url(object_url)
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copyfile(source, dest)
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError) as e:
            if e.filename != source:
                raise
            raise ObjectNotFoundError(f"No object at {object_url}.") from e

    def open(self, object_url: str) -> BinaryIO:
        with _missing_as_not_found(object_url):
            return Path(self._from_url(object_url)).open("rb")

    def get_object_metadata(self, key: str) -> ObjectMetadata | None:
        path = self._resolve(key)
        with self._locked(path, shared=True):  # never between another write's bytes and its type
            if not path.is_file():
                return None
            st = path.stat()
            content_type = self._read_content_type(path, st)
        return ObjectMetadata(
            content_type=content_type,
            size=st.st_size,
            last_modified=datetime.fromtimestamp(st.st_mtime, tz=timezone.utc),
        )

    def get_object_metadata_at(self, object_url: str) -> ObjectMetadata | None:
        return self.get_object_metadata(self.get_object_key(object_url))

    def list(self, prefix: str) -> list[str]:
        root = self._root.resolve()
        return [key for key in (p.relative_to(root).as_posix() for p in self._objects_below(root)) if key.startswith(prefix)]

    def list_at(self, url_prefix: str) -> list[str]:
        base = url_prefix if url_prefix.endswith("/") else url_prefix + "/"
        dir_path = self._resolve(self.get_object_key(url_prefix))
        if not dir_path.is_dir():
            return []
        return [base + p.relative_to(dir_path).as_posix() for p in sorted(self._objects_below(dir_path))]

    def object_url(self, key: str) -> str:
        return self._to_url(self._resolve(key))

    def get_object_key(self, object_url: str) -> str:
        if not object_url.startswith("file://"):
            raise ValueError(f"{object_url!r} is not a file:// url of this store.")
        root = self._root.resolve()
        try:
            path = Path(object_url[len("file://"):]).resolve()
        except (OSError, RuntimeError) as e:
            raise ValueError(f"{object_url!r} is not an object in {root}.") from e
        if path != root and root not in path.parents:
            raise ValueError(f"{object_url!r} is not an object in {root}.")
        return path.relative_to(root).as_posix()

    def grants_reach(self, sandbox_type: str | None) -> bool:
        return sandbox_type == _LOCAL_SANDBOX_TYPE

    def issue_read_grant(self, object_url: str, *, expires_in: int | None = None) -> HttpGetGrant:
        expires_in = self.grant_lifetime_seconds if expires_in is None else expires_in
        url, expires_at = self._grant("get", expires_in, key=self._grant_key(object_url))
        return HttpGetGrant(kind="http-get", url=url, expires_at=expires_at)

    def issue_write_grant(
        self, object_url: str, *, media_type: str, max_bytes: int, expires_in: int | None = None
    ) -> HttpPutGrant:
        expires_in = self.grant_lifetime_seconds if expires_in is None else expires_in
        url, expires_at = self._grant(
            "put", expires_in, key=self._grant_key(object_url), max_bytes=max_bytes, content_type=media_type
        )
        return HttpPutGrant(kind="http-put", url=url, expires_at=expires_at, headers={"Content-Type": media_type})

    def issue_upload_policy(self, prefix_url: str, *, max_object_bytes: int, expires_in: int) -> UploadPolicy:
        key = self.get_object_key(prefix_url)
        url, expires_at = self._grant(
            "post", expires_in, prefix="" if key == "." else key + "/", max_bytes=max_object_bytes
        )
        return UploadPolicy(
            write=HttpPostPolicyGrant(kind="http-post-policy", url=url, fields={}, path_field="key", file_field="file"),
            expires_at=expires_at,
        )

    def _grant(self, op: Op, expires_in: int, **scope) -> tuple[str, datetime]:
        if expires_in <= 0:
            raise ValueError(f"expires_in must be positive, got {expires_in}")
        expires = int(time.time()) + expires_in
        server = grant_server(self._grant_bind_host, self._grant_advertise_host)
        return server.issue(self, op, expires, **scope), datetime.fromtimestamp(expires, timezone.utc)

    def _grant_key(self, object_url: str) -> str:
        """The key of the one object at ``object_url``, refusing the store's own files."""
        key = self.get_object_key(object_url)
        self._resolve(key)
        return key

    def _resolve(self, key: str) -> Path:
        root = self._root.resolve()
        resolved = (self._root / key).resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f"Key {key!r} escapes the store root.")
        parts = resolved.relative_to(root).parts
        if parts and parts[0] in _RESERVED:
            raise ValueError(f"Key {key!r} is reserved: the store keeps its own {parts[0]} there.")
        return resolved

    def _objects_below(self, dir_path: Path) -> Iterator[Path]:
        """Every object file below ``dir_path``, leaving out the store's own files at the root."""
        root = self._root.resolve()
        if not dir_path.is_dir():
            return
        for entry in dir_path.iterdir():
            if dir_path == root and entry.name in _RESERVED:
                continue
            if entry.is_file():
                yield entry
            elif entry.is_dir():
                yield from (p for p in entry.rglob("*") if p.is_file())

    @contextlib.contextmanager
    def _staged(self) -> Iterator[Path]:
        """An empty file to write an object into before it is committed; removed if it never is."""
        ensure_state_dir(self._root)
        staging = self._root / _STAGING_DIR
        staging.mkdir(exist_ok=True)
        if not self._swept:
            self._swept = True
            _sweep(staging)
        fd, name = tempfile.mkstemp(dir=staging, prefix=_STAGED_PREFIX)
        staged = Path(name)
        try:
            if os.name == "nt":  # an open file cannot be renamed there, and there is no flock to hold
                os.close(fd)
                fd = None
            else:
                fcntl.flock(fd, fcntl.LOCK_EX)  # held while the write lives, so a sweep never takes its file
            staged.chmod(0o644)
            yield staged
        finally:
            if fd is not None:
                os.close(fd)
            staged.unlink(missing_ok=True)

    def _commit(self, staged: Path, path: Path, content_type: str, *, allow_overwrite: bool) -> None:
        """Put a staged file in place at ``path`` and record its content type, both under the key's lock, so no
        other write to the key lands in between and no reader sees one write's type with another's bytes. An
        overwrite drops the old type first, so one cut short leaves the type unknown, never wrong.

        The type is also recorded with the identity of the file it describes, which is what keeps them paired
        where there is no lock (Windows, whose NTFS keeps times fine enough to tell writes apart)."""
        identity = _stamp(staged)
        meta = self._meta_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._locked(path):
            if allow_overwrite:
                # The old type is set aside while the bytes land: a write cut short reads as unknown, never as the
                # old type, and one that fails before its bytes land puts the old type back.
                aside = self._set_aside(meta)
                try:
                    os.replace(staged, path)
                except BaseException:
                    if aside is not None:
                        with contextlib.suppress(FileNotFoundError):  # swept meanwhile: the type reads as unknown
                            os.replace(aside, meta)
                    raise
                if aside is not None:
                    with contextlib.suppress(OSError):  # the bytes have landed; a sweep takes what is left
                        aside.unlink(missing_ok=True)
            else:
                try:
                    os.link(staged, path)  # exclusive even against a writer that takes no lock
                except FileExistsError:
                    raise ObjectAlreadyExistsError(f"Object already exists at {path}.") from None
                except OSError as e:
                    if e.errno not in _NO_HARD_LINKS:
                        raise
                    # The lock makes the check and the rename one step; on Windows, rename refuses an existing target.
                    if path.exists():
                        raise ObjectAlreadyExistsError(f"Object already exists at {path}.") from None
                    try:
                        (os.rename if os.name == "nt" else os.replace)(staged, path)
                    except FileExistsError:
                        raise ObjectAlreadyExistsError(f"Object already exists at {path}.") from None
            # The default type is what a writer that named none gets; it reads back as unknown, so readers
            # still guess from the name as they did before types were kept.
            try:
                if content_type == DEFAULT_CONTENT_TYPE:
                    meta.unlink(missing_ok=True)
                    return
                meta.parent.mkdir(parents=True, exist_ok=True)
                with self._staged() as staged_meta:
                    staged_meta.write_text(json.dumps({"content_type": content_type, "object": identity}))
                    os.replace(staged_meta, meta)
            except OSError as e:  # the object is in place and its type reads back as unknown
                logger.warning("Could not record the content type of %s: %s", path, e)

    @contextlib.contextmanager
    def _locked(self, path: Path, *, shared: bool = False) -> Iterator[None]:
        """The key's lock, exclusive to write it and shared to read its type: a flock on one of a fixed set of
        files in the staging directory, which the kernel drops if its holder dies. None on Windows."""
        if os.name == "nt":
            yield
            return
        try:
            f = self._lock_path(path).open("r" if shared else "a")
        except OSError:
            if not shared:
                raise
            f = None  # no write has taken this lock yet, or the store is read-only: nothing to wait for
        if f is None:
            yield
            return
        with f:
            fcntl.flock(f, fcntl.LOCK_SH if shared else fcntl.LOCK_EX)
            yield

    def _set_aside(self, meta: Path) -> Path | None:
        """Move a type record into the staging directory, fresh enough that no sweep takes it; None when there is
        none."""
        aside = self._root / _STAGING_DIR / f"{_STAGED_PREFIX}aside-{secrets.token_hex(8)}"
        try:
            os.replace(meta, aside)
        except FileNotFoundError:
            return None
        with contextlib.suppress(OSError):
            os.utime(aside)
        return aside

    def _lock_path(self, path: Path) -> Path:
        stripe = int(hashlib.sha256(str(path).encode()).hexdigest()[:8], 16) % _LOCK_STRIPES
        return self._root / _STAGING_DIR / f"lock-{stripe:02x}"

    def _read_content_type(self, path: Path, st: os.stat_result | None = None) -> str | None:
        """The content type recorded for the object at ``path``, if it was recorded for this very file; None
        otherwise, as for one written before types were kept."""
        try:
            recorded = json.loads(self._meta_path(path).read_text())
            if recorded.get("object") != _identity(st or path.stat()):
                return None
            return recorded.get("content_type")
        except (OSError, ValueError, AttributeError):
            return None

    def _meta_path(self, path: Path) -> Path:
        root = self._root.resolve()
        return root / _META_DIR / path.relative_to(root)

    @staticmethod
    def _to_url(path: Path) -> str:
        return f"file://{path.resolve()}"

    @staticmethod
    def _from_url(object_url: str) -> str:
        return object_url[len("file://"):] if object_url.startswith("file://") else object_url


def _stamp(staged: Path) -> list[int]:
    """Give a staged file a modification time no other write shares (now, to the millisecond, plus a random
    remainder) and return its identity. A rename keeps it, and so does a copy that keeps modification times."""
    now_ns = time.time_ns()
    stamp = now_ns - now_ns % 1_000_000 + secrets.randbelow(1_000_000)
    os.utime(staged, ns=(stamp, stamp))
    return _identity(staged.stat())


def _sweep(staging: Path) -> None:
    """Remove files staged by writes that died: each live write holds a lock on its file, which the kernel drops
    when the writer exits, so a file no one holds will never be committed. Not on Windows, which has no flock."""
    if os.name == "nt":
        return
    cutoff = time.time() - _STAGED_GRACE_SECONDS
    for staged in staging.glob(f"{_STAGED_PREFIX}*"):
        with contextlib.suppress(OSError):  # BlockingIOError, the one that matters: a live write holds it
            if staged.stat().st_mtime > cutoff:
                continue
            with staged.open("rb") as f:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                staged.unlink()


def _identity(st: os.stat_result) -> list[int]:
    return [st.st_size, st.st_mtime_ns]



@contextlib.contextmanager
def _missing_as_not_found(object_url: str) -> Iterator[None]:
    try:
        yield
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError) as e:
        raise ObjectNotFoundError(f"No object at {object_url}.") from e
