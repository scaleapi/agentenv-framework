"""The in-process HTTPS server that honours local object-transfer grants.

One route, ``/v1/grants/{token}``, with the semantics of the S3 requests the grants stand in for:
GET returns an object's exact bytes; PUT writes one object of the claimed type and size; a
multipart POST writes one object anywhere under the claimed prefix. A token that does not verify,
has expired or names another method is a 403; a request outside the grant's bounds is a 4xx.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import ipaddress
import logging
import os
import platform
import socket
import subprocess
import threading
import time
from pathlib import Path
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, BinaryIO

import uvicorn
from python_multipart.exceptions import FormParserError
from python_multipart.multipart import MultipartParser, parse_options_header
from starlette.applications import Starlette
from starlette.requests import ClientDisconnect, Request
from starlette.responses import PlainTextResponse, Response, StreamingResponse
from starlette.routing import Route
from uvicorn.protocols.http.h11_impl import H11Protocol

from agent_env.store.base import GrantUnavailableError
from agent_env.store.object_store.local.tls import check_local_host, local_ca, server_context
from agent_env.store.object_store.local.tokens import (
    GrantClaims,
    GrantSigner,
    InvalidGrantError,
    Op,
    store_id,
)
from agent_env.store.object_store.object_store import DEFAULT_CONTENT_TYPE

if TYPE_CHECKING:
    from agent_env.store.object_store.local.store import LocalFilesystemObjectStore

logger = logging.getLogger(__name__)

DEFAULT_ADVERTISE_HOST = "host.docker.internal"
_START_TIMEOUT_SECONDS = 10
_OPS_BY_METHOD = {"GET": "get", "HEAD": "get", "PUT": "put", "POST": "post"}
_UPLOAD_KEY_FIELD = "key"  # the form fields a policy upload names its key and file by, as S3's do
_UPLOAD_FILE_FIELD = "file"
_MAX_FORM_FIELDS = 32
_MAX_FORM_FIELD_BYTES = 64 * 1024
_MAX_PART_HEADER_BYTES = 16 * 1024
_DOWNLOAD_CHUNK_BYTES = 1024 * 1024

_servers: dict[tuple[str, str], GrantServer] = {}
_servers_lock = threading.Lock()


class _UnloggedH11Protocol(H11Protocol):
    """HTTP/1.1 without access logs, which would record grant tokens. uvicorn's own switch clears
    the process-wide access logger, silencing any other server in this process."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.access_log = False


class _Rejected(Exception):
    def __init__(self, status: int, reason: str) -> None:
        super().__init__(reason)
        self.status = status
        self.reason = reason


def grant_server(bind_host: str | None, advertise_host: str | None) -> GrantServer:
    """This process's grant server for a bind and advertised host, created on first use; neither
    given means this platform's default bind address and ``host.docker.internal``."""
    key = (bind_host or default_bind_host(), advertise_host or DEFAULT_ADVERTISE_HOST)
    with _servers_lock:
        server = _servers.get(key)
        if server is None:
            server = _servers[key] = GrantServer(*key)
        return server


@functools.cache
def default_bind_host() -> str:
    """Where the grant server listens unless told otherwise. On macOS and Windows, Docker reaches
    the host's loopback as ``host.docker.internal``; native Linux maps that name to the default
    bridge's gateway (``--add-host host.docker.internal:host-gateway``), so the server listens
    there, reachable from containers and not beyond the host."""
    if platform.system() != "Linux":
        return "127.0.0.1"
    try:
        gateway = subprocess.run(
            ["docker", "network", "inspect", "bridge", "--format", "{{range .IPAM.Config}}{{.Gateway}} {{end}}"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.split()
        return str(ipaddress.ip_address(gateway[0]))
    except (OSError, subprocess.SubprocessError, IndexError, ValueError) as e:
        logger.warning(
            "Could not find the Docker bridge gateway (%s); the local grant server listens on 127.0.0.1, "
            "which containers may not reach. Set grant_bind_host under [stores.object] to override.", e,
        )
        return "127.0.0.1"


class GrantServer:
    """Serves the grants of the local stores registered with it, over HTTPS on one port. Started by
    the first grant it issues; it runs in a daemon thread for the rest of the process."""

    def __init__(self, bind_host: str, advertise_host: str) -> None:
        self.bind_host = bind_host
        self.advertise_host = advertise_host
        self._signer = GrantSigner()
        self._stores: dict[str, LocalFilesystemObjectStore] = {}
        self._lock = threading.Lock()
        self._port: int | None = None
        self._uvicorn: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self._app = Starlette(routes=[Route("/v1/grants/{token}", self._handle, methods=["GET", "PUT", "POST"])])

    def issue(
        self,
        store: LocalFilesystemObjectStore,
        op: Op,
        expires: int,
        *,
        key: str | None = None,
        prefix: str | None = None,
        max_bytes: int | None = None,
        content_type: str | None = None,
    ) -> str:
        """The URL of a new grant for ``op`` on ``store``, valid until ``expires`` (epoch seconds)."""
        claims = GrantClaims(
            op=op, store=self._register(store), expires=expires, key=key, prefix=prefix,
            max_bytes=max_bytes, content_type=content_type,
        )
        port = self._start()
        host = f"[{self.advertise_host}]" if ":" in self.advertise_host else self.advertise_host
        return f"https://{host}:{port}/v1/grants/{self._signer.sign(claims)}"

    def close(self) -> None:
        """Stop serving; grants already issued stop working."""
        with self._lock:
            if self._uvicorn is not None:
                self._uvicorn.should_exit = True
            if self._thread is not None:
                self._thread.join(timeout=_START_TIMEOUT_SECONDS)
            self._port = self._uvicorn = self._thread = None

    def _register(self, store: LocalFilesystemObjectStore) -> str:
        name = store_id(str(store.root.resolve()))
        with self._lock:
            self._stores[name] = store
        return name

    def _start(self) -> int:
        with self._lock:
            if self._port is not None:
                return self._port
            address = ipaddress.ip_address(self.bind_host)
            sock = socket.socket(socket.AF_INET6 if address.version == 6 else socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.bind((self.bind_host, 0))
                hosts = ["localhost", "host.docker.internal", "127.0.0.1", self.advertise_host]
                with contextlib.suppress(ValueError):  # a name the CA cannot vouch for would void the certificate
                    check_local_host(self.bind_host)
                    hosts.append(self.bind_host)
                context = server_context(local_ca(), hosts)
            except (OSError, ValueError) as e:
                sock.close()
                raise GrantUnavailableError(f"the local grant server cannot listen on {self.bind_host}: {e}") from e
            config = uvicorn.Config(
                self._app, lifespan="off", log_config=None, http=_UnloggedH11Protocol,
                ssl_context_factory=lambda _config, _default: context,
            )
            server = uvicorn.Server(config)
            failure: list[BaseException] = []

            def serve() -> None:
                try:
                    asyncio.run(server.serve(sockets=[sock]))
                except BaseException as e:  # uvicorn ends a failed startup with SystemExit
                    failure.append(e)

            thread = threading.Thread(target=serve, name="agent-env-grant-server", daemon=True)
            thread.start()
            deadline = time.monotonic() + _START_TIMEOUT_SECONDS
            while not server.started:
                if failure or not thread.is_alive() or time.monotonic() > deadline:
                    server.should_exit = True
                    sock.close()
                    raise GrantUnavailableError(
                        f"the local grant server did not start on {self.bind_host}"
                        + (f": {failure[0]!r}" if failure else "")
                    )
                time.sleep(0.01)
            self._uvicorn, self._thread = server, thread
            self._port = sock.getsockname()[1]
            logger.info("Local grant server listening on %s:%d", self.bind_host, self._port)
            return self._port

    async def _handle(self, request: Request) -> Response:
        try:
            claims = self._signer.verify(request.path_params["token"], now=time.time())
        except (InvalidGrantError, ValueError, KeyError, TypeError):
            return PlainTextResponse("The grant is invalid or has expired.", status_code=403)
        store = self._stores.get(claims.store)
        if store is None or claims.op != _OPS_BY_METHOD.get(request.method):
            return PlainTextResponse("The grant does not allow this request.", status_code=403)
        try:
            if claims.op == "get":
                return _get(store, claims)
            if claims.op == "put":
                await _put(store, claims, request)
                return Response(status_code=200)
            await _post(store, claims, request)
            return Response(status_code=204)
        except _Rejected as r:
            return PlainTextResponse(r.reason, status_code=r.status)
        except ClientDisconnect:
            return PlainTextResponse("The upload ended early.", status_code=400)
        except Exception:
            logger.exception("The local grant server failed a %s request", request.method)
            return PlainTextResponse("The object store failed the transfer.", status_code=500)


def _path(store: LocalFilesystemObjectStore, key: str) -> Path:
    try:
        return store._resolve(key)
    except ValueError:
        raise _Rejected(403, "The grant's key is not an object key of this store.") from None


def _get(store: LocalFilesystemObjectStore, claims: GrantClaims) -> Response:
    path = _path(store, claims.key)
    # Opened under the key's lock and typed from that open file, so the bytes sent and their type are one write's.
    with store._locked(path, shared=True):
        try:
            f = path.open("rb")
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            raise _Rejected(404, "No object exists at this grant's key.") from None
        st = os.fstat(f.fileno())
        content_type = store._read_content_type(path, st)
    return StreamingResponse(
        _chunks(f), media_type=content_type or DEFAULT_CONTENT_TYPE, headers={"Content-Length": str(st.st_size)}
    )


async def _chunks(f: BinaryIO) -> AsyncIterator[bytes]:
    try:
        while chunk := await asyncio.to_thread(f.read, _DOWNLOAD_CHUNK_BYTES):
            yield chunk
    finally:
        f.close()


async def _put(store: LocalFilesystemObjectStore, claims: GrantClaims, request: Request) -> None:
    if request.headers.get("content-type") != claims.content_type:
        raise _Rejected(403, f"The upload must declare Content-Type {claims.content_type}.")
    declared = request.headers.get("content-length")
    try:
        if declared is not None and int(declared) > claims.max_bytes:
            raise _Rejected(400, "The upload is larger than the grant allows.")
    except ValueError:
        raise _Rejected(400, "The upload's Content-Length is not a number.") from None
    path = _path(store, claims.key)
    with store._staged() as staged:
        received = 0
        with staged.open("wb") as out:
            async for chunk in request.stream():
                received += len(chunk)
                if received > claims.max_bytes:
                    raise _Rejected(400, "The upload is larger than the grant allows.")
                out.write(chunk)
        store._commit(staged, path, claims.content_type, allow_overwrite=True)


async def _post(store: LocalFilesystemObjectStore, claims: GrantClaims, request: Request) -> None:
    media_type, params = parse_options_header(request.headers.get("content-type"))
    if media_type != b"multipart/form-data" or b"boundary" not in params:
        raise _Rejected(400, "The upload must be multipart/form-data.")
    with contextlib.ExitStack() as stack:
        upload = _PolicyUpload(store, claims, stack)
        parser = MultipartParser(params[b"boundary"], upload.callbacks())
        try:
            async for chunk in request.stream():
                parser.write(chunk)
            parser.finalize()
        except FormParserError:
            raise _Rejected(400, "The upload is not a valid multipart form.") from None
        upload.commit()


class _PolicyUpload:
    """One multipart POST under a policy grant, read as S3 reads one: form fields, then the file,
    which is written to a staged file as it arrives. Fields after the file are ignored."""

    def __init__(self, store: LocalFilesystemObjectStore, claims: GrantClaims, stack: contextlib.ExitStack) -> None:
        self._store = store
        self._claims = claims
        self._stack = stack
        self._fields: dict[str, bytes] = {}
        self._headers: dict[bytes, bytes] = {}
        self._header_bytes = 0
        self._header_name = b""
        self._header_value = b""
        self._part_name: str | None = None
        self._part_data = bytearray()
        self._path: Path | None = None
        self._staged: Path | None = None
        self._out: BinaryIO | None = None
        self._received = 0
        self._file_done = False

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

    def commit(self) -> None:
        if self._staged is None or not self._file_done:
            raise _Rejected(400, f"The upload has no {_UPLOAD_FILE_FIELD!r} field.")
        self._out.close()
        content_type = self._field("Content-Type") or DEFAULT_CONTENT_TYPE
        self._store._commit(self._staged, self._path, content_type, allow_overwrite=True)

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
            raise _Rejected(400, "A form part's headers are too large.")

    def _on_header_end(self) -> None:
        self._headers[self._header_name.lower()] = self._header_value
        self._header_name = self._header_value = b""

    def _on_headers_finished(self) -> None:
        if self._file_done:
            return
        _, options = parse_options_header(self._headers.get(b"content-disposition"))
        name = options.get(b"name")
        if name is None:
            raise _Rejected(400, "A form field has no name.")
        self._part_name = name.decode("utf-8", "replace")
        if self._part_name == _UPLOAD_FILE_FIELD:
            self._path = _path(self._store, self._upload_key())
            self._staged = self._stack.enter_context(self._store._staged())
            self._out = self._stack.enter_context(self._staged.open("wb"))
        elif len(self._fields) >= _MAX_FORM_FIELDS:
            raise _Rejected(400, "The upload has too many form fields.")

    def _on_part_data(self, data: bytes, start: int, end: int) -> None:
        if self._file_done:
            return
        if self._part_name == _UPLOAD_FILE_FIELD:
            self._received += end - start
            if self._received > self._claims.max_bytes:
                raise _Rejected(400, "The upload is larger than the grant allows.")
            self._out.write(data[start:end])
            return
        if len(self._part_data) + end - start > _MAX_FORM_FIELD_BYTES:
            raise _Rejected(400, "A form field is too large.")
        self._part_data += data[start:end]

    def _on_part_end(self) -> None:
        if self._file_done:
            return
        if self._part_name == _UPLOAD_FILE_FIELD:
            self._file_done = True
        elif self._part_name is not None:
            self._fields[self._part_name] = bytes(self._part_data)

    def _field(self, name: str) -> str | None:
        value = self._fields.get(name)
        return None if value is None else value.decode("utf-8", "replace")

    def _upload_key(self) -> str:
        """The key the upload names, which must come before the file, be a plain relative path,
        and lie under the grant's prefix."""
        key = self._field(_UPLOAD_KEY_FIELD)
        if key is None:
            raise _Rejected(400, f"The upload must name its {_UPLOAD_KEY_FIELD!r} before its file.")
        parts = key.split("/")
        if key.startswith("/") or "\\" in key or "\x00" in key or any(p in ("", ".", "..") for p in parts):
            raise _Rejected(403, "The upload key is not a normalized relative path.")
        if not key.startswith(self._claims.prefix):
            raise _Rejected(403, "The upload key is outside the grant's prefix.")
        return key
