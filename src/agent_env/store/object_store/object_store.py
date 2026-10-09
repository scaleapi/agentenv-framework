"""Backend-agnostic object store abstraction over write-once keyed blobs."""

from __future__ import annotations

import io
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import BinaryIO

from agentenv_protocol.transfers import (
    HttpGetGrant,
    HttpPartsPutGrant,
    HttpPostPolicyGrant,
    HttpPutGrant,
    Uploaded,
    WriteObject,
)

from agent_env.store.base import GrantUnavailableError, UploadFailedError

DEFAULT_CONTENT_TYPE = "application/octet-stream"
DEFAULT_GRANT_LIFETIME_SECONDS = 12 * 60 * 60
# The shortest a store's grants may be set to last: longer than the ten minutes agent-env waits for an agent
# to move an object through one.
MIN_GRANT_LIFETIME_SECONDS = 15 * 60


def grant_lifetime(seconds: object, *, most: int | None = None) -> int:
    """``seconds`` checked as a store's ``grant_lifetime_seconds`` setting: a whole number of seconds, at
    least ``MIN_GRANT_LIFETIME_SECONDS``, and no more than ``most`` where the store cannot sign for longer."""
    if isinstance(seconds, bool) or not isinstance(seconds, int):
        raise ValueError(f"grant_lifetime_seconds must be a whole number of seconds, got {seconds!r}")
    if seconds < MIN_GRANT_LIFETIME_SECONDS:
        raise ValueError(
            f"grant_lifetime_seconds must be at least {MIN_GRANT_LIFETIME_SECONDS}, to outlast a transfer; got {seconds}"
        )
    if most is not None and seconds > most:
        raise ValueError(f"grant_lifetime_seconds can be at most {most} for this store, got {seconds}")
    return seconds


@dataclass(frozen=True)
class UploadPolicy:
    """A signed multipart POST that uploads any object under one prefix, and when it stops working."""
    write: HttpPostPolicyGrant
    expires_at: datetime


class PendingWrite(ABC):
    """An object a remote party uploads through ``grant``, for the caller to use only once ``complete``
    accepts what the party says it uploaded. Leaving the ``with`` block without completing aborts the write:
    a staged upload (an S3 multipart upload) is discarded, but a one-PUT write lands as the PUT finishes and
    stays, since object stores do not delete, so begin a write at a fresh URL nothing else refers to."""

    def __init__(self, object_url: str, grant: WriteObject) -> None:
        self.object_url = object_url
        self.grant = grant
        self._settled = False

    def complete(self, uploaded: Uploaded) -> None:
        """Make the object from what was uploaded; raise UploadFailedError, aborting the write, when the
        party reports more than ``grant.max_bytes``, the stored bytes are not the ``uploaded.size_bytes`` it
        reported, or the store cannot finish it."""
        try:
            if uploaded.size_bytes > self.grant.max_bytes:
                raise UploadFailedError(
                    f"{uploaded.size_bytes} bytes were reported uploaded to {self.object_url}, "
                    f"over its {self.grant.max_bytes}-byte limit"
                )
            self._complete(uploaded)
        except BaseException:
            self.abort()
            raise
        self._settled = True

    def abort(self) -> None:
        if not self._settled:
            self._settled = True
            self._abort()

    @abstractmethod
    def _complete(self, uploaded: Uploaded) -> None: ...

    @abstractmethod
    def _abort(self) -> None: ...

    def __enter__(self) -> PendingWrite:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.abort()


class _OnePutWrite(PendingWrite):
    """A write through one PUT, which makes the object as it lands, so aborting it leaves the object. Its grant
    stays good until it expires, so the receiver, which chose the bytes anyway, can replace an accepted object
    until then; an S3 parts write cannot, since completing it ends the upload its part URLs name."""

    def __init__(self, store: ObjectStore, object_url: str, grant: WriteObject) -> None:
        super().__init__(object_url, grant)
        self._store = store

    def _complete(self, uploaded: Uploaded) -> None:
        metadata = self._store.get_object_metadata_at(self.object_url)
        if metadata is None or metadata.size != uploaded.size_bytes:
            stored = "nothing" if metadata is None else f"{metadata.size} bytes"
            raise UploadFailedError(f"{self.object_url} holds {stored}, not the {uploaded.size_bytes} bytes reported")

    def _abort(self) -> None:
        pass


@dataclass(frozen=True)
class ObjectMetadata:
    """Metadata of a stored object. ``content_encoding`` is how the stored bytes are encoded (such
    as ``gzip``); reads return the bytes as stored, so ``size`` counts the encoded bytes."""
    content_type: str | None = None
    size: int | None = None
    last_modified: datetime | None = None
    content_encoding: str | None = None


class ObjectStore(ABC):
    """Object store interface. Objects are addressed by a logical ``key``.

    Write ops return an ``object_url`` — an opaque, backend-specific locator
    (persisted on documents) that read ops accept back; ``object_url(key)``
    reproduces it without writing. A read of a missing object raises ObjectNotFoundError.
    """

    # Whether this store can serve the three issue_* grant methods; callers check this, not
    # the NotImplementedError the defaults raise, and grants_reach before handing a grant on.
    supports_transfer_grants: bool = False
    # The largest object one upload through a grant can create, where the provider caps it.
    max_single_upload_bytes: int | None = None
    # How long a read or write grant lasts when its caller names no lifetime; the built-in stores
    # take it as their grant_lifetime_seconds setting.
    grant_lifetime_seconds: int = DEFAULT_GRANT_LIFETIME_SECONDS

    @classmethod
    def from_config(cls, **config) -> ObjectStore:
        """Construct from a resolved config table; backends that build a client override this."""
        return cls(**config)

    @abstractmethod
    def put(
        self,
        key: str,
        data: bytes,
        content_type: str = DEFAULT_CONTENT_TYPE,
        allow_overwrite: bool = False,
    ) -> str:
        """Write ``data`` at ``key`` and return its object_url; raises ObjectAlreadyExistsError unless allow_overwrite."""

    @abstractmethod
    def put_file(
        self, key: str, file_path: str, content_type: str = DEFAULT_CONTENT_TYPE
    ) -> str:
        """Write-once upload of a local file at ``key`` and return its object_url."""

    @abstractmethod
    def put_file_at(
        self, object_url: str, file_path: str, content_type: str = DEFAULT_CONTENT_TYPE
    ) -> str:
        """Write-once upload of a local file to an explicit object_url (any location)."""

    @abstractmethod
    def get(self, object_url: str) -> bytes:
        """Read the object at an object_url previously returned by put/put_file."""

    def exists(self, key: str) -> bool:
        """Whether an object exists at ``key`` (derived from get_object_metadata)."""
        return self.get_object_metadata(key) is not None

    @abstractmethod
    def get_object_metadata(self, key: str) -> ObjectMetadata | None:
        """Metadata (content_type, size, last_modified, content_encoding) for the object at
        ``key``, or None if it does not exist. Unpersisted fields come back None."""

    @abstractmethod
    def get_object_metadata_at(self, object_url: str) -> ObjectMetadata | None:
        """Metadata for the object at an explicit object_url (any location), or None if absent."""

    @abstractmethod
    def list(self, prefix: str) -> list[str]:
        """Logical keys under ``prefix`` (store-relative; excludes directory markers)."""

    @abstractmethod
    def list_at(self, url_prefix: str) -> list[str]:
        """Object_urls under an explicit url_prefix (any location; excludes directory markers)."""

    @abstractmethod
    def object_url(self, key: str) -> str:
        """The object_url ``put(key)`` would return, without writing."""

    @abstractmethod
    def get_object_key(self, object_url: str) -> str:
        """The logical key ``object_url`` addresses — inverse of ``object_url(key)``.
        Only valid for object_urls in this store (raises ValueError otherwise)."""

    def owns(self, object_url: str) -> bool:
        """Whether ``object_url`` is one of this store's own urls (what ``object_url(key)`` can
        return), so callers need not read its scheme; the explicit-url reads may reach further."""
        try:
            self.get_object_key(object_url)
        except ValueError:
            return False
        return True

    @abstractmethod
    def download_to_file(self, object_url: str, dest_path: str) -> None:
        """Stream the object at an object_url to a local path, creating parent dirs."""

    def open(self, object_url: str) -> BinaryIO:
        """The object at an object_url as a readable binary stream. The default reads it whole;
        a backend that can stream overrides this."""
        return io.BytesIO(self.get(object_url))

    def read(self, key: str) -> bytes:
        """Read by key — convenience for the list→read flow (``get(object_url(key))``)."""
        return self.get(self.object_url(key))

    def signed_get_url(self, object_url: str, expires_in: int = 3600) -> str | None:
        """A URL a remote party (e.g. a sandbox VM) can GET this object from directly,
        or None if the backend can't produce one (e.g. local filesystem)."""
        return None

    def signed_put_url(self, object_url: str, expires_in: int = 3600) -> str | None:
        """A URL a remote party can PUT (upload) this object to directly,
        or None if the backend can't produce one (e.g. local filesystem)."""
        return None

    def shared_credentials_env(self) -> dict[str, str]:
        """Environment variables handing credentials to the agents and env services agent-env
        deploys, so they can use this store directly. None by default; a deployment that wants
        it overrides this, and the workloads get whatever scope those credentials carry."""
        return {}

    def grants_reach(self, sandbox_type: str | None) -> bool:
        """Whether an agent in a sandbox of ``sandbox_type`` (None: unknown) can use this store's
        grants. True by default, for grants that are public HTTPS URLs."""
        return True

    def issue_read_grant(
        self, object_url: str, *, expires_in: int | None = None
    ) -> HttpGetGrant:
        """An HTTPS GET grant for the existing object at ``object_url``. ``expires_at`` is at most
        ``expires_in`` away (None: ``grant_lifetime_seconds``), and earlier if the store's signing
        credentials expire first; raise GrantUnavailableError for an ``expires_in`` beyond what the
        store can sign."""
        raise NotImplementedError(
            f"{type(self).__name__} cannot issue remote object-transfer grants"
        )

    def issue_write_grant(
        self,
        object_url: str,
        *,
        media_type: str,
        max_bytes: int,
        expires_in: int | None = None,
    ) -> HttpPutGrant:
        """An HTTPS PUT grant for one object at ``object_url``, signed for ``media_type`` and
        bounded to ``max_bytes`` where the provider can enforce that; expiry as for a read grant."""
        raise NotImplementedError(
            f"{type(self).__name__} cannot issue remote object-transfer grants"
        )

    def begin_write(
        self,
        object_url: str,
        *,
        media_type: str,
        max_bytes: int,
        kinds: frozenset[str],
        expires_in: int | None = None,
    ) -> PendingWrite:
        """Start a write a remote party makes to ``object_url`` through a grant of one of ``kinds``,
        preferring ``http-put-parts``. This default issues one PUT, bounded by one upload; a store that
        can take an object in parallel parts overrides it."""
        if self.max_single_upload_bytes is not None:
            max_bytes = min(max_bytes, self.max_single_upload_bytes)
        put = self.issue_write_grant(object_url, media_type=media_type, max_bytes=max_bytes, expires_in=expires_in)
        if "http-put-parts" in kinds:
            write = HttpPartsPutGrant(
                kind="http-put-parts", part_bytes=max_bytes, urls=[put.url], expires_at=put.expires_at, headers=put.headers
            )
        elif "http-put" in kinds:
            write = put
        else:
            raise GrantUnavailableError(f"{type(self).__name__} issues no write grant of the kinds {sorted(kinds)}")
        return _OnePutWrite(self, object_url, WriteObject(media_type=media_type, max_bytes=max_bytes, write=write))

    def issue_upload_policy(
        self, prefix_url: str, *, max_object_bytes: int, expires_in: int
    ) -> UploadPolicy:
        """A multipart POST policy for uploads of at most ``max_object_bytes`` each to any key
        below ``prefix_url``. It is handed over once for a long capture, so it must last all of
        ``expires_in``: raise GrantUnavailableError when it might not, judged by the kind of
        credentials that sign it rather than by how long they have left. Object counts and total
        size are the uploader's to enforce."""
        raise NotImplementedError(
            f"{type(self).__name__} cannot issue remote object-transfer grants"
        )


def issues_grants_to(store: ObjectStore, sandbox_type: str | None) -> bool:
    """Whether ``store`` can hand a transfer grant to a remote party on the ``sandbox_type`` sandbox
    provider (None: unknown): it issues grants, and they reach that provider."""
    return store.supports_transfer_grants and store.grants_reach(sandbox_type)


def readable_url(store: ObjectStore, object_url: str, *, sandbox_type: str | None, expires_in: int) -> str | None:
    """An HTTPS URL that a remote party on the ``sandbox_type`` sandbox provider (None: unknown) can GET the
    object at ``object_url`` from: a read grant when the store's grants reach it, else a URL the store
    signs; None when the store offers neither. It lasts at least ``expires_in`` seconds and at least the
    store's grant lifetime, unless the store's signing credentials or limits end it sooner."""
    expires_in = max(expires_in, store.grant_lifetime_seconds)
    if issues_grants_to(store, sandbox_type):
        try:
            grant = store.issue_read_grant(object_url, expires_in=expires_in)
        except GrantUnavailableError:  # it cannot last that long; a signed URL may
            grant = None
        if grant is not None and not grant.headers:
            return str(grant.url)
    return store.signed_get_url(object_url, expires_in=expires_in)
