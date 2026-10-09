"""Which services of a universe keep bytes outside the database.

A service bundle is a zip of ``data.json`` plus, for a service that serves file bytes
(Drive files, mail attachments, repository trees), a ``root/`` tree the service stages
into its own container when it loads the bundle. A servicedb snapshot carries the
database alone, so a restore has to re-ingest exactly those services; this module says
which ones by reading each bundle's central directory in place, without downloading it.
"""

from __future__ import annotations

import io
import logging
import urllib.request
import zipfile
from collections.abc import Callable

from agent_env.config import get_config

logger = logging.getLogger(__name__)

#: Bundle members under this prefix are file bytes the service stages outside its database.
FILE_TREE_PREFIX = "root/"
_SIGNED_URL_TTL_S = 600
_RANGE_REQUEST_TIMEOUT_S = 60


def environments_with_file_trees(environment_universe_artifact) -> list[str]:
    """Names of the universe's environments whose bundle ships a ``root/`` tree, in universe order."""
    names: list[str] = []
    for environment_artifact in environment_universe_artifact.get_environment_artifacts():
        file_artifact = environment_artifact.get_file_artifact()
        if bundle_has_file_tree(file_artifact.object_url):
            names.append(environment_artifact.environment_name)
    return names


def bundle_has_file_tree(object_url: str) -> bool:
    """Whether the bundle at ``object_url`` is a zip with at least one file under ``root/``.

    A bundle that is not a zip (a bare ``data.json``, gzipped or not) has no file tree.
    Only the zip's end record and central directory are read, so a multi-GB bundle costs a
    few ranged requests.
    """
    with _open_object(object_url) as fp:
        if not zipfile.is_zipfile(fp):
            return False
        fp.seek(0)
        with zipfile.ZipFile(fp) as zf:
            return any(_is_file_tree_member(name) for name in zf.namelist())


def _is_file_tree_member(name: str) -> bool:
    return name.startswith(FILE_TREE_PREFIX) and not name.endswith("/")


def _open_object(object_url: str) -> io.IOBase:
    """A seekable reader over the object: ranged requests against a signed URL when the
    store can sign one, otherwise the whole object in memory."""
    store = get_config().get_object_store_at(object_url)
    signed_url = store.signed_get_url(object_url, expires_in=_SIGNED_URL_TTL_S)
    metadata = store.get_object_metadata_at(object_url) if signed_url else None
    if signed_url and metadata is not None and metadata.size is not None:
        return RangedReader(metadata.size, _http_range_fetcher(signed_url))
    return io.BytesIO(store.get(object_url))


def _http_range_fetcher(url: str) -> Callable[[int, int], bytes]:
    def fetch(start: int, end: int) -> bytes:
        request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(request, timeout=_RANGE_REQUEST_TIMEOUT_S) as response:
            data = response.read()
            if response.status != 206:
                # The server ignored the range and sent the whole object.
                data = data[start:end + 1]
            return data
    return fetch


class RangedReader(io.RawIOBase):
    """A read-only, seekable file over ``fetch(start, end)`` (inclusive byte range) for an
    object of known ``size``. Each read is one fetch of exactly the bytes asked for, which is
    what ``zipfile`` needs: the end record, then the central directory, nothing else."""

    def __init__(self, size: int, fetch: Callable[[int, int], bytes]):
        super().__init__()
        self._size = size
        self._fetch = fetch
        self._pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._pos, io.SEEK_END: self._size}[whence]
        self._pos = max(0, base + offset)
        return self._pos

    def read(self, size: int = -1) -> bytes:
        end = self._size if size is None or size < 0 else min(self._size, self._pos + size)
        if end <= self._pos:
            return b""
        data = self._fetch(self._pos, end - 1)
        self._pos += len(data)
        return data

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)
