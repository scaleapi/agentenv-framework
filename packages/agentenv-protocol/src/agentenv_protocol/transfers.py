"""Provider-neutral object-transfer types and exact-object helpers."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import random
import re
import io
import tempfile
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Annotated, BinaryIO, Literal, TypeVar
from urllib.parse import urlsplit, urlunsplit

import httpx
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    model_validator,
)

_MIME_TYPE = re.compile(r"^[!#$&^_.+\-|~0-9A-Za-z]+/[!#$&^_.+\-|~0-9A-Za-z]+$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CHUNK_BYTES = 64 * 1024
_UTC = timezone.utc  # noqa: UP017 -- datetime.UTC requires Python 3.11.
# Idle limits, not totals, as long as botocore's: long enough for a slow uplink to drain its
# buffers before the store answers, short enough that a stalled connection is retried within
# the minutes agent-env waits for an extension call.
TRANSFER_IDLE_TIMEOUT_SECONDS = 60.0
TRANSFER_ATTEMPTS = 3
_CONNECT_TIMEOUT_SECONDS = 30.0
_RETRY_BACKOFF_SECONDS = 1.0
_TRANSFER_TIMEOUT = httpx.Timeout(
    connect=_CONNECT_TIMEOUT_SECONDS,
    read=TRANSFER_IDLE_TIMEOUT_SECONDS,
    write=TRANSFER_IDLE_TIMEOUT_SECONDS,
    pool=_CONNECT_TIMEOUT_SECONDS,
)
# The longest the helpers take to give up on a transfer whose every connection stalls.
TRANSFER_STALL_BUDGET_SECONDS = TRANSFER_ATTEMPTS * (
    _CONNECT_TIMEOUT_SECONDS + TRANSFER_IDLE_TIMEOUT_SECONDS
) + sum(_RETRY_BACKOFF_SECONDS * 2**attempt for attempt in range(TRANSFER_ATTEMPTS - 1))
_T = TypeVar("_T")
# A grant naming a path on its holder's own staging carries this header, whose value is that path on
# the holder's server: a sandbox can't always call its own public URL, so the helpers reach the server
# over loopback.
STAGING_PATH_HEADER = "AgentEnv-Staging-Path"
_SERVER_PORT_ENV = "A2A_PORT"
_SERVER_PORT: ContextVar[int | None] = ContextVar("agentenv_protocol_server_port", default=None)


def _https_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("must be an absolute HTTPS URL")
    return value


def _utc_timestamp(value: datetime) -> datetime:
    offset = value.utcoffset()
    if offset is None:
        raise ValueError("must be an RFC 3339 UTC timestamp")
    if offset.total_seconds() != 0:
        raise ValueError("must use UTC")
    return value


def _media_type(value: str) -> str:
    if not _MIME_TYPE.fullmatch(value):
        raise ValueError("must be a concrete MIME type")
    return value


def _sha256(value: str) -> str:
    if not _SHA256.fullmatch(value):
        raise ValueError("must be 64 lowercase hexadecimal characters")
    return value


def _relative_path(value: str) -> str:
    if "\\" in value or any(part in ("", ".", "..") for part in value.split("/")):
        raise ValueError("must be a normalized relative POSIX path")
    return value


HttpsUrl = Annotated[str, AfterValidator(_https_url)]
UtcTimestamp = Annotated[datetime, AfterValidator(_utc_timestamp)]
MediaType = Annotated[str, AfterValidator(_media_type)]
Sha256 = Annotated[str, AfterValidator(_sha256)]
RelativePath = Annotated[str, AfterValidator(_relative_path)]


class TransferModel(BaseModel):
    """Closed, immutable base model for transfer wire types."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class HttpGetGrant(TransferModel):
    kind: Literal["http-get"]
    url: HttpsUrl = Field(repr=False)
    expires_at: UtcTimestamp
    headers: dict[str, str] | None = Field(default=None, repr=False)


class HttpPutGrant(TransferModel):
    kind: Literal["http-put"]
    url: HttpsUrl = Field(repr=False)
    expires_at: UtcTimestamp
    headers: dict[str, str] | None = Field(default=None, repr=False)


class HttpPostPolicyGrant(TransferModel):
    kind: Literal["http-post-policy"]
    url: HttpsUrl = Field(repr=False)
    fields: dict[str, str] = Field(repr=False)
    path_field: str = Field(min_length=1)
    file_field: str = Field(min_length=1)
    headers: dict[str, str] | None = Field(default=None, repr=False)


class WriteNamespaceGrant(TransferModel):
    root_path: RelativePath
    expires_at: UtcTimestamp
    max_objects: StrictInt = Field(gt=0)
    max_object_bytes: StrictInt = Field(gt=0)
    max_total_bytes: StrictInt = Field(gt=0)
    write: HttpPostPolicyGrant


class ReadObject(TransferModel):
    media_type: MediaType
    max_bytes: StrictInt = Field(gt=0)
    size_bytes: StrictInt | None = Field(default=None, ge=0)
    sha256: Sha256 | None = None
    read: HttpGetGrant

    @model_validator(mode="after")
    def _known_size_is_bounded(self) -> ReadObject:
        if self.size_bytes is not None and self.size_bytes > self.max_bytes:
            raise ValueError("size_bytes must not exceed max_bytes")
        return self


class WriteObject(TransferModel):
    media_type: MediaType
    max_bytes: StrictInt = Field(gt=0)
    write: HttpPutGrant


class Uploaded(TransferModel):
    size_bytes: StrictInt = Field(ge=0)
    sha256: Sha256 | None = None


_ErrorCode = Literal[
    "invalid_transfer",
    "grant_expired",
    "transfer_too_large",
    "integrity_mismatch",
    "transfer_rejected",
    "transfer_unavailable",
    "transfer_timeout",
]
_ERROR_STATUS: dict[str, tuple[int, bool]] = {
    "invalid_transfer": (400, False),
    "grant_expired": (410, False),
    "transfer_too_large": (413, False),
    "integrity_mismatch": (422, False),
    "transfer_rejected": (502, False),
    "transfer_unavailable": (502, True),
    "transfer_timeout": (504, True),
}
_TOO_LARGE = "The transfer exceeds its configured size limit."
_UNREADABLE = "The transfer source could not be read."
_UNAVAILABLE = "The object store is temporarily unavailable."


class TransferError(RuntimeError):
    """Sanitized object-transfer failure suitable for an extension error body."""

    def __init__(
        self, code: _ErrorCode, message: str, *, retryable: bool | None = None
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code, default_retryable = _ERROR_STATUS[code]
        self.retryable = default_retryable if retryable is None else retryable

    def body(self) -> dict[str, object]:
        return {
            "error": {
                "code": self.code,
                "message": str(self),
                "retryable": self.retryable,
            }
        }


_HTTPX_LOGGER = logging.getLogger("httpx")
_REDACT_REQUEST_URLS: ContextVar[bool] = ContextVar(
    "agentenv_protocol_redact_request_urls", default=False
)
_URL = re.compile(
    r"(?P<scheme>https?)://(?:[^/?#@\s\"']*@)?(?P<host>[^/?#\s\"']+)[^\s\"']*"
)


class _RequestUrlRedactor(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # The formatted message, not httpx's arguments: it holds however httpx logs.
        if _REDACT_REQUEST_URLS.get():
            record.msg = _URL.sub(
                r"\g<scheme>://\g<host>/<redacted>", record.getMessage()
            )
            record.args = None
        return True


_HTTPX_LOGGER.addFilter(_RequestUrlRedactor())


@contextmanager
def redacting_request_urls() -> Iterator[None]:
    """httpx logs only the origin of requests made in this scope: grant URLs carry credentials."""

    token = _REDACT_REQUEST_URLS.set(True)
    try:
        yield
    finally:
        _REDACT_REQUEST_URLS.reset(token)


@contextmanager
def _as_transfer_errors(expires_at: datetime) -> Iterator[None]:
    """Transport failures in this scope become sanitized ``TransferError``s."""

    try:
        yield
    except httpx.TimeoutException as exc:
        raise TransferError(
            "transfer_timeout",
            "The object transfer timed out.",
            retryable=datetime.now(_UTC) < expires_at,
        ) from exc
    except httpx.HTTPError as exc:
        raise TransferError("transfer_unavailable", _UNAVAILABLE) from exc


@contextmanager
def _transfer_request(expires_at: datetime) -> Iterator[None]:
    """One request made with a grant."""

    with redacting_request_urls(), _as_transfer_errors(expires_at):
        yield


def _raise_for_transfer_status(response: httpx.Response) -> None:
    if response.is_success:
        return
    if (
        response.status_code in (408, 429)
        or response.status_code >= 500
        or _is_s3_request_timeout(response)
    ):
        raise TransferError("transfer_unavailable", _UNAVAILABLE)
    raise TransferError("transfer_rejected", "The object store rejected the transfer.")


def _is_s3_request_timeout(response: httpx.Response) -> bool:
    """S3 answers an upload that stalled with a 400 whose error code is RequestTimeout."""
    if response.status_code != 400:
        return False
    try:
        return b"<Code>RequestTimeout</Code>" in response.content[:1024]
    except httpx.ResponseNotRead:
        return False


@contextmanager
def serving_on(port: int) -> Iterator[None]:
    """Transfers made in this scope serve a request the holder's own server took on ``port``, so they
    reach its staging over loopback there. The SDK's server sets it for every request it handles."""

    token = _SERVER_PORT.set(port)
    try:
        yield
    finally:
        _SERVER_PORT.reset(token)


def loopback_url(url: str, headers: Mapping[str, str] | None) -> str | None:
    """Where the holder's own server answers a grant's ``url`` over loopback: when the grant names a path
    on the holder's staging (an ``AgentEnv-Staging-Path`` header ending the URL's path) and the server's
    port is known, from the request being served (``serving_on``) or else ``A2A_PORT``. None otherwise."""
    path = httpx.Headers(headers or {}).get(STAGING_PATH_HEADER)
    port = str(_SERVER_PORT.get() or os.environ.get(_SERVER_PORT_ENV, ""))
    if not path or not path.startswith("/") or not port.isdigit() or not 0 < int(port) < 65536:
        return None
    parsed = urlsplit(url)
    if not parsed.path.endswith(path):
        return None
    return urlunsplit(("http", f"127.0.0.1:{int(port)}", path, parsed.query, ""))


def _destinations(url: str, headers: Mapping[str, str] | None) -> tuple[str, ...]:
    """Where a grant's request goes: the holder's own server over loopback first when the grant names a
    path on its staging, then the grant's URL, for a server that isn't listening there."""
    loopback = loopback_url(url, headers)
    return (url,) if loopback is None else (loopback, url)


async def _send_to_first_listening(
    urls: tuple[str, ...], send: Callable[[str], Awaitable[_T]]
) -> _T:
    """``send`` to the first of ``urls`` whose server takes the connection."""
    *earlier, last = urls
    for url in earlier:
        try:
            return await send(url)
        except httpx.ConnectError:
            pass
    return await send(last)


def _send_to_first_listening_sync(urls: tuple[str, ...], send: Callable[[str], _T]) -> _T:
    """``_send_to_first_listening`` for a sync client."""
    *earlier, last = urls
    for url in earlier:
        try:
            return send(url)
        except httpx.ConnectError:
            pass
    return send(last)


def _check_unexpired(expires_at: datetime) -> None:
    if expires_at <= datetime.now(_UTC):
        raise TransferError("grant_expired", "The object-transfer grant has expired.")


async def _retrying(expires_at: datetime, attempt: Callable[[], Awaitable[_T]]) -> _T:
    """Run ``attempt``, retrying a retryable failure with jittered backoff up to
    three attempts in all, and only while the grant is unexpired."""

    attempts = 1
    while True:
        _check_unexpired(expires_at)
        try:
            return await attempt()
        except TransferError as exc:
            if (
                attempts == TRANSFER_ATTEMPTS
                or not exc.retryable
                or datetime.now(_UTC) >= expires_at
            ):
                raise
        delay = _RETRY_BACKOFF_SECONDS * 2 ** (attempts - 1)
        await asyncio.sleep(random.uniform(delay / 2, delay))
        attempts += 1


def _check_source(source: Path | bytes, max_bytes: int) -> int:
    if isinstance(source, bytes):
        size_bytes = len(source)
    else:
        try:
            size_bytes = source.stat().st_size
        except OSError as exc:
            raise TransferError(
                "invalid_transfer", "The transfer source is unavailable."
            ) from exc
        if not source.is_file():
            raise TransferError(
                "invalid_transfer", "The transfer source must be a file."
            )
    if size_bytes > max_bytes:
        raise TransferError("transfer_too_large", _TOO_LARGE)
    return size_bytes


class _SourceStream:
    """An open upload source that counts and hashes what it sends and fails unless
    it sends exactly the size declared for the request.

    It is the async body of a PUT and the file of a multipart POST.
    """

    def __init__(self, source: Path | bytes, size_bytes: int) -> None:
        if isinstance(source, bytes):
            self._file: BinaryIO = io.BytesIO(source)
        else:
            try:
                self._file = source.open("rb")
            except OSError as exc:
                raise TransferError("invalid_transfer", _UNREADABLE) from exc
        self._declared_bytes = size_bytes
        self.size_bytes = 0
        self._digest = hashlib.sha256()

    def __enter__(self) -> _SourceStream:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._file.close()

    def fileno(self) -> int:
        return self._file.fileno()

    # httpx sizes a multipart file part it cannot fstat by seeking to its end and back, and
    # rewinds it before sending; a rewind restarts the count.
    def tell(self) -> int:
        return self._file.tell()

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        position = self._file.seek(offset, whence)
        if position == 0:
            self.size_bytes = 0
            self._digest = hashlib.sha256()
        return position

    def read(self, size: int = -1) -> bytes:
        try:
            chunk = self._file.read(size)
        except OSError as exc:
            raise TransferError("invalid_transfer", _UNREADABLE) from exc
        self.size_bytes += len(chunk)
        if self.size_bytes > self._declared_bytes or (
            not chunk and self.size_bytes < self._declared_bytes
        ):
            raise TransferError(
                "invalid_transfer", "The transfer source changed during upload."
            )
        self._digest.update(chunk)
        return chunk

    async def __aiter__(self) -> AsyncIterator[bytes]:
        while chunk := await asyncio.to_thread(self.read, _CHUNK_BYTES):
            yield chunk

    def uploaded(self) -> Uploaded:
        return Uploaded(size_bytes=self.size_bytes, sha256=self._digest.hexdigest())


def _as_source(source: Path | str | bytes) -> Path | bytes:
    return source if isinstance(source, bytes) else Path(source)


async def upload(target: WriteObject, source: Path | bytes) -> Uploaded:
    """Upload one file, or bytes already in memory, using an opaque exact-object PUT grant."""

    source = _as_source(source)
    size_bytes = _check_source(source, target.max_bytes)
    headers = httpx.Headers(target.write.headers or {})
    headers.setdefault("content-type", target.media_type)
    headers.setdefault("content-length", str(size_bytes))
    urls = _destinations(target.write.url, target.write.headers)

    async def put() -> Uploaded:
        with _transfer_request(target.write.expires_at):
            async with httpx.AsyncClient(
                follow_redirects=False, timeout=_TRANSFER_TIMEOUT
            ) as client:

                async def to(url: str) -> Uploaded:
                    with _SourceStream(source, size_bytes) as stream:
                        response = await client.put(url, headers=headers, content=stream)
                    _raise_for_transfer_status(response)
                    return stream.uploaded()

                return await _send_to_first_listening(urls, to)

    return await _retrying(target.write.expires_at, put)


async def download(source: ReadObject, destination: Path) -> None:
    """Download one object atomically, checking the stored bytes against its size
    and checksum; the store is asked not to transcode them."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, filename = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".part",
    )
    os.close(handle)
    temporary = Path(filename)
    headers = httpx.Headers(source.read.headers or {})
    headers["accept-encoding"] = "identity"
    urls = _destinations(source.read.url, source.read.headers)

    async def get() -> tuple[int, str]:
        size_bytes = 0
        digest = hashlib.sha256()
        with _transfer_request(source.read.expires_at):
            async with httpx.AsyncClient(
                follow_redirects=False, timeout=_TRANSFER_TIMEOUT
            ) as client:
                response = await _send_to_first_listening(
                    urls,
                    lambda url: client.send(
                        client.build_request("GET", url, headers=headers), stream=True
                    ),
                )
                try:
                    _raise_for_transfer_status(response)
                    stream = await asyncio.to_thread(temporary.open, "wb")
                    try:
                        async for chunk in response.aiter_raw(_CHUNK_BYTES):
                            size_bytes += len(chunk)
                            if size_bytes > source.max_bytes:
                                raise TransferError("transfer_too_large", _TOO_LARGE)
                            digest.update(chunk)
                            await asyncio.to_thread(stream.write, chunk)
                    finally:
                        await asyncio.to_thread(stream.close)
                finally:
                    await response.aclose()
        return size_bytes, digest.hexdigest()

    try:
        size_bytes, sha256 = await _retrying(source.read.expires_at, get)
        if source.size_bytes is not None and size_bytes != source.size_bytes:
            raise TransferError(
                "integrity_mismatch",
                "The downloaded object size does not match its descriptor.",
            )
        if source.sha256 is not None and sha256 != source.sha256:
            raise TransferError(
                "integrity_mismatch",
                "The downloaded object checksum does not match its descriptor.",
            )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _post_to_namespace(
    target: WriteNamespaceGrant, object_path: str, source: Path | bytes, size_bytes: int
) -> Uploaded:
    """A multipart POST through the namespace's policy. httpx encodes multipart bodies by
    reading the file synchronously, so this runs on a sync client in a worker thread rather
    than blocking the event loop."""
    grant = target.write
    fields = {**grant.fields, grant.path_field: object_path}
    fields.setdefault("Content-Type", "application/octet-stream")
    with (
        _transfer_request(target.expires_at),
        httpx.Client(follow_redirects=False, timeout=_TRANSFER_TIMEOUT) as client,
    ):

        def to(url: str) -> Uploaded:
            with _SourceStream(source, size_bytes) as stream:
                response = client.post(
                    url,
                    data=fields,
                    files={
                        grant.file_field: (
                            object_path.rsplit("/", 1)[-1],
                            stream,
                            "application/octet-stream",
                        )
                    },
                    headers=grant.headers,
                )
            _raise_for_transfer_status(response)
            return stream.uploaded()

        return _send_to_first_listening_sync(_destinations(grant.url, grant.headers), to)


class NamespaceUploader:
    """Uploads objects under one namespace grant within its object and byte limits.

    One instance must make every upload for the grant; it runs them one at a time.
    """

    def __init__(self, target: WriteNamespaceGrant) -> None:
        self.target = target
        self._committed: dict[str, int] = {}
        self._lock = asyncio.Lock()

    async def upload(self, relative_path: str, source: Path | bytes) -> Uploaded:
        """Upload or overwrite one relative path under the namespace root."""

        try:
            path = _relative_path(relative_path)
        except ValueError as exc:
            raise TransferError(
                "invalid_transfer", "The namespace upload path is invalid."
            ) from exc
        source = _as_source(source)
        async with self._lock:
            size_bytes = _check_source(source, self.target.max_object_bytes)
            prospective = {**self._committed, path: size_bytes}
            if len(prospective) > self.target.max_objects:
                raise TransferError(
                    "transfer_too_large",
                    "The upload exceeds the namespace object limit.",
                )
            if sum(prospective.values()) > self.target.max_total_bytes:
                raise TransferError(
                    "transfer_too_large",
                    "The upload exceeds the namespace total-byte limit.",
                )
            uploaded = await _retrying(
                self.target.expires_at,
                partial(
                    asyncio.to_thread,
                    _post_to_namespace,
                    self.target,
                    f"{self.target.root_path}/{path}",
                    source,
                    size_bytes,
                ),
            )
            self._committed[path] = uploaded.size_bytes
            return uploaded
