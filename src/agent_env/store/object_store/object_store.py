"""Backend-agnostic object store abstraction over write-once keyed blobs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

DEFAULT_CONTENT_TYPE = "application/octet-stream"


@dataclass(frozen=True)
class ObjectMetadata:
    """Metadata of a stored object."""
    content_type: str | None = None
    size: int | None = None
    last_modified: datetime | None = None


class ObjectStore(ABC):
    """Object store interface. Objects are addressed by a logical ``key``.

    Write ops return an ``object_url`` — an opaque, backend-specific locator
    (persisted on documents) that read ops accept back; ``object_url(key)``
    reproduces it without writing.
    """

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
        """Metadata (content_type, size, last_modified) for the object at ``key``,
        or None if it does not exist. Unpersisted fields come back None."""

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
        Only valid for object_urls in this store (raises otherwise)."""

    @abstractmethod
    def download_to_file(self, object_url: str, dest_path: str) -> None:
        """Stream the object at an object_url to a local path, creating parent dirs."""

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

    def signed_post(
        self, url_prefix: str, *, expires_in: int = 3600, max_bytes: int | None = None
    ) -> dict | None:
        """Credentials letting a remote party upload objects it names itself, anywhere
        under ``url_prefix``. None if the backend can't produce them.

        Prefix-scoped so the uploader owns its file layout — the caller bounds where
        it may write, not what it may call things. That is also why the grant is a
        dict and not a url: a signed url covers one key, since the key is part of
        what is signed, so a prefix needs a policy the uploader submits as form
        fields. Pass the whole value on, not just ``url``.
        """
        return None
