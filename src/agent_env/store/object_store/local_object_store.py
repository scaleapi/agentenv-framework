"""Local filesystem implementation of the ObjectStore interface (no infra)."""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

from agent_env.store.base import ObjectAlreadyExistsError
from agent_env.store.local_state import ensure_state_dir
from agent_env.store.object_store.object_store import DEFAULT_CONTENT_TYPE, ObjectMetadata, ObjectStore


class LocalFilesystemObjectStore(ObjectStore):
    """ObjectStore backed by a local directory tree; content types are not persisted.

    The root-level ``.gitignore`` key is reserved for the ignore rule the store writes."""

    def __init__(self, root: str) -> None:
        self._root = Path(root)

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
        ensure_state_dir(self._root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return self._to_url(path)

    def put_file(self, key: str, file_path: str, content_type: str = DEFAULT_CONTENT_TYPE) -> str:
        path = self._resolve(key)
        if path.exists():
            raise ObjectAlreadyExistsError(f"Object already exists at {path}.")
        ensure_state_dir(self._root)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file_path, path)
        return self._to_url(path)

    def put_file_at(self, object_url: str, file_path: str, content_type: str = DEFAULT_CONTENT_TYPE) -> str:
        return self.put_file(self.get_object_key(object_url), file_path, content_type)

    def get(self, object_url: str) -> bytes:
        return Path(self._from_url(object_url)).read_bytes()

    def download_to_file(self, object_url: str, dest_path: str) -> None:
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self._from_url(object_url), dest)

    def get_object_metadata(self, key: str) -> ObjectMetadata | None:
        path = self._resolve(key)
        if not path.exists():
            return None
        st = path.stat()
        return ObjectMetadata(content_type=None, size=st.st_size, last_modified=datetime.fromtimestamp(st.st_mtime, tz=timezone.utc))

    def get_object_metadata_at(self, object_url: str) -> ObjectMetadata | None:
        return self.get_object_metadata(self.get_object_key(object_url))

    def list(self, prefix: str) -> list[str]:
        keys = []
        for path in self._root.rglob("*"):
            if not path.is_file() or path == self._root / ".gitignore":
                continue
            key = path.relative_to(self._root).as_posix()
            if key.startswith(prefix):
                keys.append(key)
        return keys

    def list_at(self, url_prefix: str) -> list[str]:
        base = url_prefix if url_prefix.endswith("/") else url_prefix + "/"
        dir_path = self._resolve(self.get_object_key(url_prefix))
        if not dir_path.is_dir():
            return []
        return [
            base + p.relative_to(dir_path).as_posix()
            for p in sorted(dir_path.rglob("*"))
            if p.is_file() and p != self._root.resolve() / ".gitignore"
        ]

    def object_url(self, key: str) -> str:
        return self._to_url(self._resolve(key))

    def get_object_key(self, object_url: str) -> str:
        path = Path(self._from_url(object_url)).resolve()
        root = self._root.resolve()
        if path != root and root not in path.parents:
            raise ValueError(f"{object_url!r} is not an object in {root}.")
        return path.relative_to(root).as_posix()

    def _resolve(self, key: str) -> Path:
        resolved = (self._root / key).resolve()
        if not resolved.is_relative_to(self._root.resolve()):
            raise ValueError(f"Key {key!r} escapes the store root.")
        if resolved == self._root.resolve() / ".gitignore":
            raise ValueError(f"Key {key!r} is reserved: the store keeps its .gitignore there.")
        return resolved

    @staticmethod
    def _to_url(path: Path) -> str:
        return f"file://{path.resolve()}"

    @staticmethod
    def _from_url(object_url: str) -> str:
        return object_url[len("file://"):] if object_url.startswith("file://") else object_url
