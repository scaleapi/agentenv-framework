"""The staging extension: a small object store on the agent's own server.

When an object store's grants cannot reach an agent's sandbox, as with a local store and a remote
sandbox, agent-env moves the objects through the agent instead. Before a call it pushes what the agent
will read into staging; after the call it pulls what the agent wrote. The grants it sends are ordinary
HTTPS grants naming staged paths on the agent's own URL, so the agent's transfer helpers don't change.

Routes, below the extension's endpoint:

* ``PUT {path}`` stores the request body;
* ``GET {path}`` returns it, with an ``ETag``;
* ``POST {prefix}`` stores a multipart upload's file at ``{prefix}/{key}``, the way an upload-policy
  grant is used: form fields first, ``key`` among them, then ``file``;
* ``GET {prefix}/`` lists what is stored below a prefix;
* ``DELETE {path}`` removes an object, and only while it still has the ``If-Match`` tag when one is
  given; ``DELETE {prefix}/`` removes everything below a prefix.

A path's first segment is an id agent-env generates and tells no one else, so it must be long enough
not to be guessed, and the directories holding staged objects are the server user's alone. Everything
staged counts against ``AGENTENV_STAGING_MAX_BYTES`` (a limit of 0 turns staging off), and lives until
agent-env removes it or the server process ends: in a directory of the process's own, or under
``AGENTENV_STAGING_DIR`` when that is set. One server process owns a staging directory, since the byte
count and the ``If-Match`` check are its own.
"""

from __future__ import annotations

import asyncio
import atexit
import hashlib
import os
import shutil
import tempfile
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from stat import S_ISREG
from typing import BinaryIO

from python_multipart.exceptions import FormParserError
from python_multipart.multipart import MultipartParser, parse_options_header
from starlette.requests import ClientDisconnect, Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

STAGING_V1_URI = "urn:agentenv:staging/v1"
STAGING_ENDPOINT = "/ext/staging"
STAGING_DIR_ENV = "AGENTENV_STAGING_DIR"
STAGING_MAX_BYTES_ENV = "AGENTENV_STAGING_MAX_BYTES"
DEFAULT_STAGING_MAX_BYTES = 16 * 1024**3
MIN_ID_LENGTH = 22  # about 128 bits of a URL-safe base64 id

_CHUNK_BYTES = 1024 * 1024
_MAX_PATH_LENGTH = 1024
_KEY_FIELD = "key"
_FILE_FIELD = "file"
_MAX_PART_HEADER_BYTES = 16 * 1024
_MAX_FIELD_BYTES = 64 * 1024
_MAX_FIELDS = 32


class StagingError(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class Staged:
    """One stored object, as a write or a listing reports it."""

    path: str
    size_bytes: int
    etag: str
    sha256: str | None = None

    def to_json(self) -> dict[str, object]:
        body: dict[str, object] = {"path": self.path, "size_bytes": self.size_bytes, "etag": self.etag}
        if self.sha256 is not None:
            body["sha256"] = self.sha256
        return body


def staging_max_bytes() -> int:
    """The configured staging limit, ``AGENTENV_STAGING_MAX_BYTES`` or 16 GiB."""
    value = os.environ.get(STAGING_MAX_BYTES_ENV, "").strip()
    if not value:
        return DEFAULT_STAGING_MAX_BYTES
    limit = int(value)
    if limit < 0:
        raise ValueError(f"{STAGING_MAX_BYTES_ENV} must not be negative")
    return limit


class StagingStore:
    """The objects staged on one agent, below ``root``, within ``max_bytes`` in all.

    Writes land in a staging file and are renamed into place, so a reader sees a whole object or none.
    Each write stamps a unique modification time, which with the size makes the object's tag.
    """

    def __init__(self, root: str | Path | None = None, *, max_bytes: int | None = None) -> None:
        configured = root or os.environ.get(STAGING_DIR_ENV)
        self._root = Path(configured) if configured else None
        self.max_bytes = staging_max_bytes() if max_bytes is None else max_bytes
        self._lock = asyncio.Lock()
        self._used: int | None = None  # stored plus reserved bytes, counted on first use
        self._last_stamp = 0

    @property
    def root(self) -> Path:
        """Where staged objects live; unless one was given, a new directory, made on first use and removed
        when the process exits, so a restarted server never counts what an earlier one left."""
        if self._root is None:
            self._root = Path(tempfile.mkdtemp(prefix="agentenv-staging-"))
            atexit.register(shutil.rmtree, self._root, ignore_errors=True)
        return self._root

    @property
    def _objects(self) -> Path:
        return self.root / "objects"

    @property
    def _incoming(self) -> Path:
        return self.root / "incoming"

    def card_extension(self, endpoint: str = STAGING_ENDPOINT) -> dict[str, object]:
        return {
            "uri": STAGING_V1_URI,
            "description": "Stage objects on the agent for agent-env to push and pull.",
            "params": {"endpoint": endpoint, "max_bytes": self.max_bytes},
        }

    async def put(self, path: str, chunks: AsyncIterator[bytes], *, size_bytes: int | None = None) -> Staged:
        """Store ``chunks`` at ``path``, replacing what is there. ``size_bytes``, when known, is
        reserved before the first byte arrives and must be what arrives."""
        target = self._file(path)
        async with _Incoming(self, size_bytes) as incoming:
            async for chunk in chunks:
                await incoming.write(chunk)
            if size_bytes is not None and incoming.size_bytes != size_bytes:
                raise StagingError(400, "The body is shorter than its Content-Length.")
            return await self._commit(incoming, target, path)

    def open(self, path: str) -> tuple[BinaryIO, Staged]:
        """The object at ``path``, opened, and its description."""
        try:
            file = self._file(path).open("rb")
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            raise StagingError(404, "Nothing is staged at this path.") from None
        stat = os.fstat(file.fileno())
        return file, Staged(path, stat.st_size, _etag(stat))

    def list_below(self, prefix: str) -> list[Staged]:
        """Everything below ``prefix``, by path relative to it."""
        top = self._file(prefix)
        found = []
        for directory, _, files in os.walk(top):
            for name in files:
                file = Path(directory) / name
                try:
                    stat = file.stat()
                except FileNotFoundError:
                    continue
                found.append(Staged(file.relative_to(top).as_posix(), stat.st_size, _etag(stat)))
        return sorted(found, key=lambda staged: staged.path)

    async def delete(self, path: str, *, if_match: str | None = None) -> None:
        target = self._file(path)
        async with self._lock:
            used = await self._usage()
            try:
                found = target.stat()
            except (FileNotFoundError, NotADirectoryError):
                raise StagingError(404, "Nothing is staged at this path.") from None
            if not S_ISREG(found.st_mode):
                raise StagingError(404, "Nothing is staged at this path.")
            if if_match is not None and if_match != _etag(found):
                raise StagingError(412, "The staged object has changed.")
            target.unlink()
            self._used = used - found.st_size
            self._prune(target.parent)

    async def delete_below(self, prefix: str) -> None:
        top = self._file(prefix)
        async with self._lock:
            used = await self._usage()
            freed = sum(staged.size_bytes for staged in await asyncio.to_thread(self.list_below, prefix))
            await asyncio.to_thread(shutil.rmtree, top, ignore_errors=True)
            self._used = used - freed
            self._prune(top.parent)

    async def post(self, prefix: str, request: Request) -> Staged:
        """Store a multipart upload's file at ``{prefix}/{key}``."""
        prefix_path = _check_path(prefix)
        media_type, params = parse_options_header(request.headers.get("content-type"))
        if media_type != b"multipart/form-data" or b"boundary" not in params:
            raise StagingError(400, "The upload must be multipart/form-data.")
        form = _Form()
        parser = MultipartParser(params[b"boundary"], form.callbacks())
        async with _Incoming(self, None) as incoming:
            try:
                async for chunk in request.stream():
                    parser.write(chunk)
                    await incoming.write(form.take_file_bytes())
                parser.finalize()
            except FormParserError:
                raise StagingError(400, "The upload is not a valid multipart form.") from None
            await incoming.write(form.take_file_bytes())
            if not form.file_done:
                raise StagingError(400, f"The upload has no {_FILE_FIELD!r} field.")
            path = _check_path(f"{prefix_path}/{form.key}")
            return await self._commit(incoming, self._file(path), path)

    def _file(self, path: str) -> Path:
        return self._objects / _check_path(path)

    async def _reserve(self, size_bytes: int) -> None:
        async with self._lock:
            used = await self._usage()
            if used + size_bytes > self.max_bytes:
                raise StagingError(413, "Staging is full.")
            self._used = used + size_bytes

    async def _release(self, size_bytes: int) -> None:
        async with self._lock:
            self._used = await self._usage() - size_bytes

    async def _usage(self) -> int:
        if self._used is None:
            self._used = await asyncio.to_thread(_tree_bytes, self._objects)
        return self._used

    async def _commit(self, incoming: _Incoming, target: Path, path: str) -> Staged:
        await incoming.close()
        async with self._lock:
            used = await self._usage()
            try:
                replaced = target.stat().st_size if target.is_file() else 0
                _private(self._objects)
                target.parent.mkdir(parents=True, exist_ok=True)
                self._stamp(incoming.path)
                os.replace(incoming.path, target)
            except (FileExistsError, IsADirectoryError, NotADirectoryError):
                raise StagingError(409, "The path conflicts with one already staged.") from None
            incoming.committed()
            self._used = used - replaced
            stat = target.stat()
        return Staged(path, stat.st_size, _etag(stat), incoming.sha256)

    def _stamp(self, file: Path) -> None:
        stamp = max(time.time_ns(), self._last_stamp + 1000)
        self._last_stamp = stamp
        os.utime(file, ns=(stamp, stamp))

    def _prune(self, directory: Path) -> None:
        while directory != self._objects and self._objects in directory.parents:
            try:
                directory.rmdir()
            except OSError:
                return
            directory = directory.parent


class _Incoming:
    """A write on its way in: a staging file, the bytes it reserves, and their digest."""

    def __init__(self, store: StagingStore, size_bytes: int | None) -> None:
        self._store = store
        self._declared = size_bytes
        self._reserved = 0
        self._file: BinaryIO | None = None
        self._committed = False
        self.path = Path()
        self.size_bytes = 0
        self._digest = hashlib.sha256()
        self._buffer = bytearray()

    async def __aenter__(self) -> _Incoming:
        if self._declared is not None:
            await self._store._reserve(self._declared)
            self._reserved = self._declared
        try:
            await asyncio.to_thread(_private, self._store._incoming)
            fd, name = tempfile.mkstemp(dir=self._store._incoming)
        except BaseException:
            await self._store._release(self._reserved)
            raise
        self._file = os.fdopen(fd, "wb")
        self.path = Path(name)
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if self._file is not None:
            self._file.close()
        if not self._committed:
            self.path.unlink(missing_ok=True)
            await self._store._release(self._reserved)

    @property
    def sha256(self) -> str:
        return self._digest.hexdigest()

    async def write(self, data: bytes) -> None:
        if not data:
            return
        self.size_bytes += len(data)
        if self._declared is not None and self.size_bytes > self._declared:
            raise StagingError(400, "The body is longer than its Content-Length.")
        if self.size_bytes > self._reserved:
            await self._store._reserve(self.size_bytes - self._reserved)
            self._reserved = self.size_bytes
        self._digest.update(data)
        self._buffer += data
        if len(self._buffer) >= _CHUNK_BYTES:
            await self._flush()

    async def _flush(self) -> None:
        data, self._buffer = bytes(self._buffer), bytearray()
        await asyncio.to_thread(self._file.write, data)

    async def close(self) -> None:
        await self._flush()
        await asyncio.to_thread(self._file.close)
        self._file = None

    def committed(self) -> None:
        """The staging file is now the object: its bytes stay counted, and it stays."""
        self._committed = True


class _Form:
    """A multipart form read as an upload policy is: fields, then the file, whose bytes are handed
    out as they arrive. Parts after the file are ignored."""

    def __init__(self) -> None:
        self._fields: dict[str, bytes] = {}
        self._headers: dict[bytes, bytes] = {}
        self._header_bytes = 0
        self._header_name = b""
        self._header_value = b""
        self._part_name: str | None = None
        self._part_data = bytearray()
        self._file_bytes = bytearray()
        self.key = ""
        self.file_done = False

    def callbacks(self) -> dict:
        return {
            "on_part_begin": self._on_part_begin,
            "on_header_field": self._on_header_field,
            "on_header_value": self._on_header_value,
            "on_header_end": self._on_header_end,
            "on_headers_finished": self._on_headers_finished,
            "on_part_data": self._on_part_data,
            "on_part_end": self._on_part_end,
        }

    def take_file_bytes(self) -> bytes:
        data, self._file_bytes = bytes(self._file_bytes), bytearray()
        return data

    def _on_part_begin(self) -> None:
        self._headers = {}
        self._header_bytes = 0
        self._part_name = None
        self._part_data = bytearray()

    def _on_header_field(self, data: bytes, start: int, end: int) -> None:
        self._count_header_bytes(end - start)
        self._header_name += data[start:end]

    def _on_header_value(self, data: bytes, start: int, end: int) -> None:
        self._count_header_bytes(end - start)
        self._header_value += data[start:end]

    def _count_header_bytes(self, count: int) -> None:
        self._header_bytes += count
        if self._header_bytes > _MAX_PART_HEADER_BYTES:
            raise StagingError(400, "A form part's headers are too large.")

    def _on_header_end(self) -> None:
        self._headers[self._header_name.lower()] = self._header_value
        self._header_name = self._header_value = b""

    def _on_headers_finished(self) -> None:
        if self.file_done:
            return
        _, options = parse_options_header(self._headers.get(b"content-disposition"))
        name = options.get(b"name")
        if name is None:
            raise StagingError(400, "A form field has no name.")
        self._part_name = name.decode("utf-8", "replace")
        if self._part_name == _FILE_FIELD:
            key = self._fields.get(_KEY_FIELD)
            if key is None:
                raise StagingError(400, f"The upload must name its {_KEY_FIELD!r} before its file.")
            self.key = key.decode("utf-8", "replace")
        elif len(self._fields) >= _MAX_FIELDS:
            raise StagingError(400, "The upload has too many form fields.")

    def _on_part_data(self, data: bytes, start: int, end: int) -> None:
        if self.file_done:
            return
        if self._part_name == _FILE_FIELD:
            self._file_bytes += data[start:end]
            return
        if len(self._part_data) + end - start > _MAX_FIELD_BYTES:
            raise StagingError(400, "A form field is too large.")
        self._part_data += data[start:end]

    def _on_part_end(self) -> None:
        if self.file_done:
            return
        if self._part_name == _FILE_FIELD:
            self.file_done = True
        elif self._part_name is not None:
            self._fields[self._part_name] = bytes(self._part_data)


def _check_path(path: str) -> str:
    """``path`` when it is a normalized relative path whose first segment is a long enough id."""
    parts = path.split("/")
    if (
        len(path) > _MAX_PATH_LENGTH
        or "\\" in path
        or "\x00" in path
        or any(part in ("", ".", "..") for part in parts)
    ):
        raise StagingError(400, "The staging path is not a normalized relative path.")
    if len(parts[0]) < MIN_ID_LENGTH:
        raise StagingError(404, "Nothing is staged at this path.")
    return path


def _private(directory: Path) -> None:
    """Make ``directory`` readable by its owner alone, so other users on the host cannot list the ids
    below it, however it was left. Only staging's own directories, never the one it was given."""
    directory.mkdir(parents=True, exist_ok=True)
    if directory.stat().st_mode & 0o077:
        directory.chmod(0o700)


def _etag(stat: os.stat_result) -> str:
    return f'"{stat.st_mtime_ns:x}-{stat.st_size:x}"'


def _tree_bytes(top: Path) -> int:
    total = 0
    for directory, _, files in os.walk(top):
        for name in files:
            try:
                total += (Path(directory) / name).stat().st_size
            except FileNotFoundError:
                pass
    return total


def staging_routes(store: StagingStore, endpoint: str = STAGING_ENDPOINT) -> list[Route]:
    """The staging extension's routes, below ``endpoint``."""

    async def handle(request: Request) -> Response:
        path: str = request.path_params["path"]
        listing = path.endswith("/")
        try:
            if listing:
                path = path[:-1]
                if request.method == "GET":
                    found = await asyncio.to_thread(store.list_below, path)
                    return JSONResponse({"objects": [staged.to_json() for staged in found]})
                if request.method == "DELETE":
                    await store.delete_below(path)
                    return Response(status_code=204)
                raise StagingError(405, "A prefix can only be listed or deleted.")
            if request.method == "GET":
                file, staged = await asyncio.to_thread(store.open, path)
                return StreamingResponse(
                    _read(file),
                    media_type="application/octet-stream",
                    headers={"content-length": str(staged.size_bytes), "etag": staged.etag},
                )
            if request.method == "PUT":
                length = request.headers.get("content-length")
                staged = await store.put(
                    path, request.stream(), size_bytes=int(length) if length is not None else None
                )
                return JSONResponse(staged.to_json(), status_code=201)
            if request.method == "POST":
                staged = await store.post(path, request)
                return JSONResponse(staged.to_json(), status_code=201)
            await store.delete(path, if_match=request.headers.get("if-match"))
            return Response(status_code=204)
        except StagingError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=exc.status_code)
        except ValueError:
            return JSONResponse({"detail": "The request is malformed."}, status_code=400)
        except ClientDisconnect:
            return Response(status_code=400)
        except OSError as exc:
            status = 507 if exc.errno == 28 else 500  # ENOSPC: the sandbox's disk is full
            return JSONResponse({"detail": "Staging could not store the object."}, status_code=status)

    return [Route(f"{endpoint}/{{path:path}}", handle, methods=["GET", "PUT", "POST", "DELETE"])]


async def _read(file: BinaryIO) -> AsyncIterator[bytes]:
    try:
        while chunk := await asyncio.to_thread(file.read, _CHUNK_BYTES):
            yield chunk
    finally:
        file.close()
