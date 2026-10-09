"""Which services of a universe keep bytes outside the database.

A service bundle is ``data.json`` plus, for a service that serves file bytes (Drive
files, mail attachments, repository trees), a ``root/`` tree the service stages into its
own container when it loads the bundle. A bundle can also carry small files inline, as
``files[]`` entries with ``content`` in ``data.json``, which the service writes to disk
the same way. A servicedb snapshot holds the database alone, so a restore has to
re-ingest exactly those services; this module says which ones by reading each bundle's
zip directory in place, and ``data.json`` only when the zip has no ``root/`` tree.
"""

from __future__ import annotations

import gzip
import io
import json
import logging
import urllib.request
import zipfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from agent_env.config import get_config

logger = logging.getLogger(__name__)

#: Bundle members under this prefix are file bytes the service stages outside its database.
FILE_TREE_PREFIX = "root/"
DATA_JSON_MEMBER = "data.json"
_SIGNED_URL_TTL_S = 600
_RANGE_REQUEST_TIMEOUT_S = 60
#: Sequential reads (a zip member, a bare JSON document) are served from one fetch of at
#: least this size instead of one request per 4 KiB chunk.
_READ_AHEAD_BYTES = 1024 * 1024
#: A store that cannot sign URLs is read whole; past this the bundle is refused rather than
#: held in memory, and the caller re-ingests every service.
_WHOLE_OBJECT_MAX_BYTES = 256 * 1024 * 1024
#: Inline ``files[]`` carry their bytes in the JSON text, so they are small by construction;
#: a data.json past this is a database dump and is not parsed for them.
_INLINE_FILES_SCAN_MAX_BYTES = 64 * 1024 * 1024
_GZIP_MAGIC = b"\x1f\x8b"
#: Bundles are inspected concurrently; each costs a few requests, so this bounds open connections.
_INSPECT_CONCURRENCY = 8


def environments_with_file_trees(environment_universe_artifact) -> list[str]:
    """Names of the universe's environments whose bundle ships file bytes, in universe order.

    Bundles are inspected concurrently; any one failing fails the whole answer, since a
    partial list would restore a service blind."""
    environment_artifacts = environment_universe_artifact.get_environment_artifacts()
    if not environment_artifacts:
        return []
    with ThreadPoolExecutor(max_workers=min(_INSPECT_CONCURRENCY, len(environment_artifacts))) as pool:
        verdicts = list(pool.map(
            lambda artifact: bundle_has_file_tree(artifact.get_file_artifact().object_url),
            environment_artifacts,
        ))
    return [a.environment_name for a, ships_files in zip(environment_artifacts, verdicts) if ships_files]


def bundle_has_file_tree(object_url: str) -> bool:
    """Whether the bundle at ``object_url`` ships file bytes the service writes to disk:
    zip members under ``root/``, or ``files[]`` entries with ``content`` in its data.json.

    Only the zip's end record and central directory are read for the first; data.json is
    read only when the zip has no ``root/`` tree, and only up to a size bound.
    """
    with _open_object(object_url) as fp:
        if zipfile.is_zipfile(fp):
            fp.seek(0)
            with zipfile.ZipFile(fp) as zf:
                if any(_is_file_tree_member(name) for name in zf.namelist()):
                    return True
                try:
                    info = zf.getinfo(DATA_JSON_MEMBER)
                except KeyError:
                    return False
                if info.file_size > _INLINE_FILES_SCAN_MAX_BYTES:
                    logger.info("%s: data.json is %d bytes; not scanned for inline files", object_url, info.file_size)
                    return False
                with zf.open(info) as member:
                    return _declares_inline_files(_load_json(member, object_url))
        size = fp.seek(0, io.SEEK_END)
        if size > _INLINE_FILES_SCAN_MAX_BYTES:
            logger.info("%s: bundle is %d bytes; not scanned for inline files", object_url, size)
            return False
        fp.seek(0)
        if fp.read(len(_GZIP_MAGIC)) == _GZIP_MAGIC:
            fp.seek(0)
            with gzip.GzipFile(fileobj=fp) as plain:
                return _declares_inline_files(_load_json(plain, object_url))
        fp.seek(0)
        return _declares_inline_files(_load_json(fp, object_url))


def _is_file_tree_member(name: str) -> bool:
    return name.startswith(FILE_TREE_PREFIX) and not name.endswith("/")


def _load_json(stream, object_url: str):
    try:
        return json.load(stream)
    except (ValueError, UnicodeDecodeError) as e:
        logger.warning("%s: data.json is not JSON (%s); treating it as data-only", object_url, e)
        return None


def _declares_inline_files(data) -> bool:
    """A top-level ``files`` list with at least one entry carrying ``content``: the filesystem
    server's inline file definitions. File *records* without content (a messaging server's
    uploads table) are rows like any other."""
    if not isinstance(data, dict) or not isinstance(data.get("files"), list):
        return False
    return any(isinstance(entry, dict) and "content" in entry for entry in data["files"])


def _open_object(object_url: str) -> io.IOBase:
    """A seekable reader over the object: ranged requests against a signed URL when the
    store can sign one, otherwise the whole object in memory, within a size bound."""
    store = get_config().get_object_store_at(object_url)
    signed_url = store.signed_get_url(object_url, expires_in=_SIGNED_URL_TTL_S)
    metadata = store.get_object_metadata_at(object_url)
    size = metadata.size if metadata is not None else None
    if signed_url and size is not None:
        return RangedReader(size, _http_range_fetcher(signed_url))
    if size is not None and size > _WHOLE_OBJECT_MAX_BYTES:
        raise RuntimeError(
            f"{object_url} is {size} bytes and its store cannot sign ranged reads; refusing to read it whole"
        )
    return io.BytesIO(store.get(object_url))


def _http_range_fetcher(url: str) -> Callable[[int, int], bytes]:
    def fetch(start: int, end: int) -> bytes:
        request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(request, timeout=_RANGE_REQUEST_TIMEOUT_S) as response:
            if response.status != 206:
                # Reading the body would be the whole object; the caller falls back
                # to re-ingesting every service instead.
                raise RuntimeError(f"ranged read refused: HTTP {response.status} for bytes={start}-{end}")
            return response.read()
    return fetch


class RangedReader(io.RawIOBase):
    """A read-only, seekable file over ``fetch(start, end)`` (inclusive byte range) for an
    object of known ``size``. A read fetches at least ``_READ_AHEAD_BYTES`` from its position
    and serves later reads inside that window from memory, so ``zipfile``'s end record,
    central directory and a member body each cost about one request."""

    def __init__(self, size: int, fetch: Callable[[int, int], bytes], read_ahead: int = _READ_AHEAD_BYTES):
        super().__init__()
        self._size = size
        self._fetch = fetch
        self._read_ahead = read_ahead
        self._pos = 0
        self._window_start = 0
        self._window = b""

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
        window_end = self._window_start + len(self._window)
        if not (self._window_start <= self._pos and end <= window_end):
            fetch_end = min(self._size, max(end, self._pos + self._read_ahead))
            self._window = self._fetch(self._pos, fetch_end - 1)
            self._window_start = self._pos
        offset = self._pos - self._window_start
        data = self._window[offset: offset + (end - self._pos)]
        self._pos += len(data)
        return data

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)
