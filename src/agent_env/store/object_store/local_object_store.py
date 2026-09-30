"""Local filesystem implementation of the ObjectStore interface (no infra)."""

from __future__ import annotations

import contextlib
import errno
import ipaddress
import json
import logging
import os
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
from agent_env.store.local_state import ensure_state_dir
from agent_env.store.object_store.local_grants.server import grant_server
from agent_env.store.object_store.local_grants.tls import check_local_host
from agent_env.store.object_store.local_grants.tokens import Op
from agent_env.store.object_store.object_store import (
    DEFAULT_CONTENT_TYPE,
    ObjectMetadata,
    ObjectStore,
    UploadPolicy,
)

logger = logging.getLogger(__name__)

_META_DIR = ".agentenv-meta"  # each object's content type, at the object's own key
_STAGING_DIR = ".agentenv-tmp"  # files being written, inside the root so a rename into place is atomic
_RESERVED = (".gitignore", _META_DIR, _STAGING_DIR)
# A filesystem without hard links: a no-overwrite write reserves its key by exclusive create instead.
_NO_HARD_LINKS = frozenset({errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EXDEV, errno.EMLINK})


class LocalFilesystemObjectStore(ObjectStore):
    """ObjectStore backed by a local directory tree, ``<state root>/object_store`` unless ``root`` is given.

    A write lands whole or not at all, and keeps any content type but the default. The root-level ``.gitignore``,
    ``.agentenv-meta`` and ``.agentenv-tmp`` keys are reserved for the store's own files.

    Its HTTPS transfer grants are served by a server in this process, started by the first grant: it listens on
    ``grant_bind_host`` (by default loopback, or the Docker bridge gateway on Linux) and grant URLs name
    ``grant_advertise_host`` (by default ``host.docker.internal``), which must be a local name or a loopback or
    private address. A grant works until it expires or this process exits."""

    def __init__(
        self, root: str | None = None, *, grant_bind_host: str | None = None, grant_advertise_host: str | None = None
    ) -> None:
        if grant_bind_host is not None:
            try:
                ipaddress.ip_address(grant_bind_host)
            except ValueError:
                raise ValueError(f"grant_bind_host must be an IP address, got {grant_bind_host!r}") from None
        if grant_advertise_host is not None:
            check_local_host(grant_advertise_host)
        self._root = Path(root) if root is not None else state_root() / "object_store"
        self._grant_bind_host = grant_bind_host
        self._grant_advertise_host = grant_advertise_host

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
        if not path.is_file():
            return None
        st = path.stat()
        return ObjectMetadata(
            content_type=self._read_content_type(path),
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

    def issue_read_grant(self, object_url: str, *, expires_in: int = 3600) -> HttpGetGrant:
        url, expires_at = self._grant("get", expires_in, key=self._grant_key(object_url))
        return HttpGetGrant(kind="http-get", url=url, expires_at=expires_at)

    def issue_write_grant(
        self, object_url: str, *, media_type: str, max_bytes: int, expires_in: int = 3600
    ) -> HttpPutGrant:
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
        fd, name = tempfile.mkstemp(dir=staging)
        os.close(fd)
        staged = Path(name)
        staged.chmod(0o644)
        try:
            yield staged
        finally:
            staged.unlink(missing_ok=True)

    def _commit(self, staged: Path, path: Path, content_type: str, *, allow_overwrite: bool) -> None:
        """Put a staged file in place at ``path`` and record its content type. An overwrite renames over
        whatever is there; otherwise a hard link claims ``path`` only if no other writer got there first."""
        path.parent.mkdir(parents=True, exist_ok=True)
        if allow_overwrite:
            os.replace(staged, path)
        else:
            try:
                os.link(staged, path)
            except FileExistsError:
                raise ObjectAlreadyExistsError(f"Object already exists at {path}.") from None
            except OSError as e:
                if e.errno not in _NO_HARD_LINKS:
                    raise
                try:
                    os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
                except FileExistsError:
                    raise ObjectAlreadyExistsError(f"Object already exists at {path}.") from None
                os.replace(staged, path)
        # The default type is what a writer that named none gets; it reads back as unknown, so readers
        # still guess from the name as they did before types were kept.
        try:
            meta = self._meta_path(path)
            if content_type == DEFAULT_CONTENT_TYPE:
                meta.unlink(missing_ok=True)
                return
            meta.parent.mkdir(parents=True, exist_ok=True)
            with self._staged() as staged_meta:
                staged_meta.write_text(json.dumps({"content_type": content_type}))
                os.replace(staged_meta, meta)
        except OSError as e:  # the object is in place; only its type goes unrecorded
            logger.warning("Could not record the content type of %s: %s", path, e)

    def _read_content_type(self, path: Path) -> str | None:
        """The content type recorded for the object at ``path``; None for one written before types were kept."""
        try:
            return json.loads(self._meta_path(path).read_text()).get("content_type")
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


@contextlib.contextmanager
def _missing_as_not_found(object_url: str) -> Iterator[None]:
    try:
        yield
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError) as e:
        raise ObjectNotFoundError(f"No object at {object_url}.") from e
