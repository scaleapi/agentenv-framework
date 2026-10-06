"""Objects moved through an agent's staging extension, for agents the object store's grants cannot reach.

A local object store's grants reach only agents on this machine. An agent elsewhere that serves
``urn:agentenv:staging/v1`` (every SDK agent does) gets ordinary HTTPS grants naming paths on its own
staging routes instead: agent-env pushes each object the agent reads there before the call, and pulls each
object it writes into the store after it. A changelog namespace stays staged for the agent's life, so its
increments are drained into the store while the agent works, when a prompt ends, and before the agent is
torn down. Every byte moves over a connection agent-env opens, so nothing has to reach this machine.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import secrets
import tempfile
import threading
from collections.abc import AsyncIterator, Iterable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import quote, urlsplit

import httpx
from agentenv_protocol.a2a_agent import STAGING_V1_URI
from agentenv_protocol.transfers import HttpGetGrant, HttpPostPolicyGrant, HttpPutGrant, redacting_request_urls

from agent_env.config import get_config
from agent_env.store.base import ObjectAlreadyExistsError
from agent_env.store.object_store import DEFAULT_CONTENT_TYPE, ObjectMetadata, ObjectStore, UploadPolicy

logger = logging.getLogger(__name__)

# How often a running prompt's staged changelog increments are moved into the store: what is lost if
# the agent's sandbox dies mid-prompt is at most this much of its work.
DRAIN_INTERVAL_SECONDS = 5.0
# The longest a last drain waits on an agent before giving up on what it still holds, so a stalled agent can't
# hold up the end of a prompt or a teardown.
LAST_DRAIN_SECONDS = 120.0
_CHUNK_BYTES = 1024 * 1024
_ATTEMPTS = 3
_CONCURRENCY = 4
_TIMEOUT = httpx.Timeout(connect=30.0, read=60.0, write=60.0, pool=30.0)
_REPLACING = threading.Lock()


class StagingError(RuntimeError):
    """Moving an object through an agent's staging failed. The message never names a staged path, whose id
    is what keeps it private."""


def staging_endpoint(a2a_url: str, card: Mapping[str, Any] | None) -> str | None:
    """The absolute URL of the agent's staging routes, when its card advertises them on an HTTPS URL a
    grant can name; None otherwise."""
    params = _staging_params(card)
    endpoint = params.get("endpoint")
    url = f"{a2a_url.rstrip('/')}{endpoint}" if isinstance(endpoint, str) and endpoint.startswith("/") else None
    return url.rstrip("/") if url is not None and urlsplit(url).scheme == "https" else None


def _staging_params(card: Mapping[str, Any] | None) -> Mapping[str, Any]:
    for extension in ((card or {}).get("capabilities") or {}).get("extensions") or ():
        if isinstance(extension, Mapping) and extension.get("uri") == STAGING_V1_URI:
            return extension.get("params") or {}
    return {}


def transfer_store(
    store: ObjectStore, a2a_url: str, card: Mapping[str, Any] | None, *, sandbox_type: str | None
) -> ObjectStore:
    """The store an extension call's grants come from: ``store`` when its grants reach the agent, else a
    store that stages them through the agent when the agent serves staging, else ``store`` itself, so the
    call falls back to the forms that carry no grants."""
    endpoint = staging_endpoint(a2a_url, card)
    if endpoint is None or (store.supports_transfer_grants and store.grants_reach(sandbox_type)):
        return store
    return _staged(store, endpoint, card)


def staged_store(store: ObjectStore, a2a_url: str, card: Mapping[str, Any] | None) -> StagedObjectStore | None:
    """``store``, issuing grants staged through the agent at ``a2a_url``; None when the agent serves no staging."""
    endpoint = staging_endpoint(a2a_url, card)
    return None if endpoint is None else _staged(store, endpoint, card)


def _staged(store: ObjectStore, endpoint: str, card: Mapping[str, Any] | None) -> StagedObjectStore:
    limit = _staging_params(card).get("max_bytes")
    return StagedObjectStore(store, endpoint, max_bytes=limit if isinstance(limit, int) and limit > 0 else None)


@dataclass(frozen=True)
class StagedNamespace:
    """A changelog namespace staged on an agent: where its increments wait, and where they go."""

    staging_url: str  # holds an id no one else knows: kept with the run, never logged
    namespace_url: str
    max_object_bytes: int | None = None


@dataclass(frozen=True)
class _Staged:
    path: str
    object_url: str
    media_type: str = DEFAULT_CONTENT_TYPE
    max_bytes: int | None = None  # the most the grant let the agent write


class StagedObjectStore:
    """``store``, issuing grants that name paths on the agent's staging routes at ``endpoint``.

    One serves one extension call: build the call with it, then pass it to ``invoke_transfer``, which
    pushes the reads before the call, pulls the writes after it, and clears the call's staging either
    way. A namespace it issues outlives the call; ``namespaces`` names them for the caller to keep.
    """

    supports_transfer_grants = True

    def __init__(self, store: ObjectStore, endpoint: str, *, max_bytes: int | None = None) -> None:
        self.store = store
        self.endpoint = endpoint
        self.max_bytes = max_bytes  # what the agent's card says its staging holds
        self._call = secrets.token_urlsafe(24)
        self._reads: list[_Staged] = []
        self._writes: list[_Staged] = []
        self._sizes: dict[str, int | None] = {}  # each object's size when the call described it
        self.namespaces: list[StagedNamespace] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.store, name)

    def grants_reach(self, sandbox_type: str | None) -> bool:
        return True

    def get_object_metadata_at(self, object_url: str) -> ObjectMetadata | None:
        metadata = self.store.get_object_metadata_at(object_url)
        self._sizes[object_url] = None if metadata is None else metadata.size
        return metadata

    def issue_read_grant(self, object_url: str, *, expires_in: int | None = None) -> HttpGetGrant:
        staged = self._stage(self._reads, object_url)
        return HttpGetGrant(kind="http-get", url=self._url(staged.path), expires_at=self._expiry(expires_in))

    def issue_write_grant(
        self, object_url: str, *, media_type: str, max_bytes: int, expires_in: int | None = None
    ) -> HttpPutGrant:
        staged = self._stage(self._writes, object_url, media_type, max_bytes)
        return HttpPutGrant(kind="http-put", url=self._url(staged.path), expires_at=self._expiry(expires_in))

    def issue_upload_policy(self, prefix_url: str, *, max_object_bytes: int, expires_in: int) -> UploadPolicy:
        url = f"{self.endpoint}/{secrets.token_urlsafe(24)}"
        self.namespaces.append(StagedNamespace(url, prefix_url, max_object_bytes))
        return UploadPolicy(
            write=HttpPostPolicyGrant(kind="http-post-policy", url=url, fields={}, path_field="key", file_field="file"),
            expires_at=self._expiry(expires_in),
        )

    async def push(self) -> None:
        """Put each object the call reads into the agent's staging, at the size the call described it with; objects
        that together outgrow the limit its card advertises are refused before any is sent."""
        sizes = [
            self._sizes[staged.object_url] if staged.object_url in self._sizes
            else await asyncio.to_thread(_size, self.store, staged.object_url)
            for staged in self._reads
        ]
        if self.max_bytes is not None and sum(size or 0 for size in sizes) > self.max_bytes:
            raise StagingError("staging an object on the agent failed: its staging is full")
        async with _client("staging an object on the agent") as client:
            await _each(
                list(zip(self._reads, sizes)),
                lambda item: _push(client, self.store, item[0].object_url, self._url(item[0].path), item[1]),
            )

    async def pull(self) -> None:
        """Move each object the agent wrote into the store; one it did not write is skipped."""
        async with _client("collecting an object the agent staged") as client:
            await _each(self._writes, lambda staged: _pull(client, self.store, self._url(staged.path), staged))

    async def release(self) -> None:
        """Clear what the call staged. Best-effort: the agent's sandbox ending clears it too."""
        if not self._reads and not self._writes:
            return
        try:
            async with _client("clearing the call's staged objects") as client:
                (await client.delete(self._url("") + "/")).raise_for_status()
        except StagingError as exc:
            logger.warning("%s", exc)

    def _stage(
        self, staged: list[_Staged], object_url: str, media_type: str = DEFAULT_CONTENT_TYPE, max_bytes: int | None = None
    ) -> _Staged:
        entry = _Staged(str(len(self._reads) + len(self._writes)), object_url, media_type, max_bytes)
        staged.append(entry)
        return entry

    def _url(self, path: str) -> str:
        return f"{self.endpoint}/{self._call}" + (f"/{quote(path)}" if path else "")

    def _expiry(self, expires_in: int | None) -> datetime:
        seconds = expires_in if expires_in is not None else self.store.grant_lifetime_seconds
        return datetime.now(UTC) + timedelta(seconds=seconds)


async def drain(namespace: StagedNamespace, store: ObjectStore) -> int:
    """Move the increments staged in ``namespace`` into ``store``, and return how many. An increment is
    removed from staging only once it is in the store, and only if the agent has not replaced it since."""
    root = store.get_object_key(namespace.namespace_url).rstrip("/") + "/"
    async with _client("draining staged changelog increments") as client:
        listed = await client.get(namespace.staging_url + "/")
        listed.raise_for_status()
        entries = [e for e in listed.json().get("objects") or () if str(e.get("path", "")).startswith(root)]

        async def move(entry: Mapping[str, Any]) -> None:
            staged = _Staged(entry["path"], store.object_url(entry["path"]), max_bytes=namespace.max_object_bytes)
            await _pull(client, store, f"{namespace.staging_url}/{quote(entry['path'])}", staged)

        await _each(entries, move)
    return len(entries)


def staged_changelogs(metadata: Mapping[str, Any], *, agent_name: str | None = None) -> list[StagedNamespace]:
    """The changelog namespaces a run's metadata records as staged on its agents, ``agent_name``'s alone
    when one is given."""
    return [
        StagedNamespace(entry["staging_url"], entry["object_url"], entry.get("staging_max_object_bytes"))
        for entry in metadata.get("agent_changelog") or ()
        if entry.get("staging_url") and agent_name in (None, entry.get("agent_name"))
    ]


async def drain_all(namespaces: Iterable[StagedNamespace], *, within: float | None = None, quiet: bool = False) -> list[str]:
    """Drain each of ``namespaces`` into the store its namespace URL is in, giving up after ``within``
    seconds when given, and return why any could not be. One that cannot be drained is left for the next
    drain to try again, and logged unless ``quiet``."""
    failures = []
    try:
        async with asyncio.timeout(within):
            for namespace in namespaces:
                try:
                    await drain(namespace, get_config().get_object_store_at(namespace.namespace_url))
                except Exception as exc:  # an agent that is gone or busy must not fail the run
                    failures.append(str(exc) if isinstance(exc, StagingError) else type(exc).__name__)
    except TimeoutError:
        failures.append(f"staged changelog increments were not all drained within {within:g}s")
    for failure in failures if not quiet else ():
        logger.warning("%s", failure)
    return failures


@asynccontextmanager
async def draining(namespaces: Iterable[StagedNamespace]) -> AsyncIterator[None]:
    """Drain ``namespaces`` every ``DRAIN_INTERVAL_SECONDS`` for the length of the block, and once more
    when it ends."""
    namespaces = list(namespaces)
    if not namespaces:
        yield
        return

    async def loop() -> None:
        reported: list[str] = []
        while True:
            await asyncio.sleep(DRAIN_INTERVAL_SECONDS)
            failures = await drain_all(namespaces, quiet=True)
            if failures and failures != reported:  # when the reason changes, not every few seconds
                logger.warning("%s; retrying every %gs", "; ".join(failures), DRAIN_INTERVAL_SECONDS)
            elif reported and not failures:
                logger.info("staged changelog increments are draining again")
            reported = failures

    task = asyncio.create_task(loop())
    try:
        yield
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await drain_all(namespaces, within=LAST_DRAIN_SECONDS)


def _size(store: ObjectStore, object_url: str) -> int | None:
    metadata = store.get_object_metadata_at(object_url)
    return metadata.size if metadata is not None else None


async def _push(client: httpx.AsyncClient, store: ObjectStore, object_url: str, url: str, size: int | None) -> None:
    async def attempt() -> None:
        source = await asyncio.to_thread(store.open, object_url)
        headers = {"content-type": "application/octet-stream"}
        if size is not None:
            headers["content-length"] = str(size)
        response = await client.put(url, content=_chunks(source), headers=headers)
        response.raise_for_status()

    await _retrying(attempt)


async def _pull(client: httpx.AsyncClient, store: ObjectStore, url: str, staged: _Staged) -> None:
    fd, name = tempfile.mkstemp(prefix="agentenv-staged-")
    try:
        with os.fdopen(fd, "wb") as out:
            fetched = await _retrying(lambda: _fetch(client, url, out, staged.max_bytes))
        if fetched is None:
            return
        etag, digest = fetched
        try:
            await asyncio.to_thread(store.put_file_at, staged.object_url, name, staged.media_type)
        except ObjectAlreadyExistsError:
            # Rewritten after an earlier copy went into the store: the latest write wins, as through a grant.
            await asyncio.to_thread(_replace, store, staged, Path(name), digest)
        removed = await client.delete(url, headers={"if-match": etag})
        if removed.status_code not in (204, 404, 412):  # 412: rewritten since, so the next drain takes it
            removed.raise_for_status()
    finally:
        os.unlink(name)


async def _fetch(client: httpx.AsyncClient, url: str, out: BinaryIO, max_bytes: int | None) -> tuple[str, str] | None:
    """Stream the object at ``url`` into ``out``, no more than ``max_bytes`` of it, and return its tag and
    sha256; None when nothing is staged there."""
    out.seek(0)
    out.truncate()
    digest, size = hashlib.sha256(), 0
    async with client.stream("GET", url, headers={"accept-encoding": "identity"}) as response:
        if response.status_code == 404:
            return None
        response.raise_for_status()
        async for chunk in response.aiter_raw(_CHUNK_BYTES):
            size += len(chunk)
            if max_bytes is not None and size > max_bytes:
                raise StagingError("the agent staged an object larger than its grant allows")
            digest.update(chunk)
            await asyncio.to_thread(out.write, chunk)
    await asyncio.to_thread(out.flush)
    return response.headers.get("etag", ""), digest.hexdigest()


def _replace(store: ObjectStore, staged: _Staged, file: Path, digest: str) -> None:
    """Put ``file`` over the stored copy of ``staged`` unless the two hold the same bytes.

    The store overwrites only from memory, so replacements go one at a time: an agent rewrites a drained
    increment with new bytes only on purpose, and at most one such increment is held at once."""
    stored = hashlib.sha256()
    with store.open(staged.object_url) as current:
        while chunk := current.read(_CHUNK_BYTES):
            stored.update(chunk)
    if stored.hexdigest() != digest:
        with _REPLACING:
            store.put(store.get_object_key(staged.object_url), file.read_bytes(), staged.media_type, allow_overwrite=True)


async def _chunks(source: BinaryIO) -> AsyncIterator[bytes]:
    try:
        while chunk := await asyncio.to_thread(source.read, _CHUNK_BYTES):
            yield chunk
    finally:
        await asyncio.to_thread(source.close)


async def _retrying(attempt):
    for tried in range(1, _ATTEMPTS + 1):
        try:
            return await attempt()
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            transient = isinstance(exc, httpx.TransportError) or exc.response.status_code >= 500
            if tried == _ATTEMPTS or not transient:
                raise
            await asyncio.sleep(tried)


async def _each(items, act) -> None:
    """``act`` on each of ``items``, a few at a time; the first failure stops the rest."""
    gate = asyncio.Semaphore(_CONCURRENCY)

    async def one(item) -> None:
        async with gate:
            await act(item)

    tasks = [asyncio.ensure_future(one(item)) for item in items]
    try:
        await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


@asynccontextmanager
async def _client(doing: str) -> AsyncIterator[httpx.AsyncClient]:
    """A client for staging requests, which are logged by origin alone and fail as a StagingError that says
    what was being done."""
    with redacting_request_urls():
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=False) as client:
                yield client
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            why = {413: "its staging is full", 507: "its disk is full"}.get(status, f"it answered {status}")
            raise StagingError(f"{doing} failed: {why}") from None
        except httpx.HTTPError:
            raise StagingError(f"{doing} failed: the agent could not be reached") from None
