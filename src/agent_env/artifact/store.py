"""Artifact store for versioned documents and binary objects via the pluggable stores."""

from __future__ import annotations

import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional, Self, TypeVar

from agent_env.config import get_config
from agent_env.plugins import _registration

logger = logging.getLogger(__name__)
from agent_env.artifact.registry import canonical_type, equivalent_types, get_artifact_registry
from agent_env.store.base import NotFoundError
from agent_env.store.document_store import (
    DuplicateKeyError,
    Filter,
    In,
    UpdateSpec,
    VersionedEntityStore,
    VersionedEntityStoreCache,
)
from agent_env.store.ids import key_segment
from agent_env.store.query import QueryBuilder, to_document_query

if TYPE_CHECKING:
    from agent_env.artifact.artifact import Artifact

_A = TypeVar("_A", bound="Artifact")

ARTIFACTS_COLLECTION = "artifacts"

# Retry cap for the version-allocation race in put_document.
_MAX_VERSION_RETRIES = 3

_write_locks: dict[str, threading.Lock] = {}
_write_locks_guard = threading.Lock()


def artifact_write_lock(artifact_id: str) -> threading.Lock:
    """A lock for writing ``artifact_id`` from this process. A write that allocates a version, puts
    objects at paths named by it and then records the document takes the same version as any
    other write of that id running alongside it; hold this around the whole write to take turns."""
    with _write_locks_guard:
        return _write_locks.setdefault(artifact_id, threading.Lock())

# Fields that address the document rather than describe it, so `set_field` must not touch them.
_IDENTITY_FIELDS = frozenset({"id", "version", "type"})


# Hex digits of the attempt part of an object prefix: enough that two attempts never meet.
_ATTEMPT_ID_LENGTH = 8


class ArtifactQuery(QueryBuilder["Artifact"]):
    """Chainable query builder for artifacts.

    Example:
        artifact = Artifact.query().id("my_artifact").latest().first()
        artifacts = Artifact.query().type("docker_image").execute()
        artifacts = Artifact.query().version_gte(2).execute()
    """

    def __init__(self, store: Optional["ArtifactStore"] = None) -> None:
        super().__init__()
        self._store = store

    def _clone(self) -> Self:
        clone = ArtifactQuery(self._store)
        clone._filters = self._filters.copy()
        clone._sort_field = self._sort_field
        clone._sort_desc = self._sort_desc
        clone._limit_value = self._limit_value
        clone._offset_value = self._offset_value
        return clone

    def type(self, artifact_type: str) -> Self:
        spellings = equivalent_types(artifact_type)
        if len(spellings) == 1:
            return self._add_filter("type", artifact_type)
        return self._add_filter("type_in", spellings)

    def execute(self) -> list["Artifact"]:
        if self._store is None:
            self._store = get_artifact_store()
        return self._store.execute_query(self)

    def _execute_count(self) -> int:
        if self._store is None:
            self._store = get_artifact_store()
        return self._store.execute_count(self)


class ArtifactStore:
    """Store for artifact persistence in the configured document and object stores."""

    def __init__(self) -> None:
        self._versioned_cache: VersionedEntityStoreCache["Artifact"] = VersionedEntityStoreCache(
            ARTIFACTS_COLLECTION, None, self._deserialize, secondary_indexes=[["type"]],
        )

    @property
    def _doc_store(self):
        return get_config().get_document_store()

    @property
    def _versioned(self) -> VersionedEntityStore["Artifact"]:
        return self._versioned_cache.for_store(self._doc_store)

    def get(self, id: str, version: Optional[int] = None) -> "Artifact":
        """Get an artifact by id. Returns latest version if version not specified."""
        artifact = self._versioned.get(id, version)
        if artifact is None:
            version_str = f" version={version}" if version is not None else ""
            raise NotFoundError(f"Artifact {id}{version_str} not found")
        return artifact

    def next_version(self, id: str) -> int:
        # Every put asks for its version before writing an object or pushing an image, so an id
        # the store refuses, or a bare one while an @local task runs, is refused here rather than
        # after those side effects, in put_document.
        self._doc_store.check_id(id)
        get_config().check_local_run_write(id)
        return self._versioned.next_version(id)

    def attempt_prefix(self, artifact_type: str, id: str) -> str:
        """An object-store prefix of its own for one write of ``id``'s next version. Objects are written
        once, so a write that fails partway must not leave its objects where the next attempt writes."""
        attempt = f"{self.next_version(id)}-{uuid.uuid4().hex[:_ATTEMPT_ID_LENGTH]}"
        key = f"{get_config().get_artifact_key_prefix()}artifacts/{artifact_type}/{key_segment(id)}/{attempt}"
        return get_config().get_object_store_for(id).object_url(key).rstrip("/") + "/"

    def put_object(
        self,
        artifact_type: str,
        id: str,
        version: int,
        object_name: str,
        data: bytes,
        content_type: str = "application/json",
        allow_overwrite: bool = False,
    ) -> str:
        """Upload an artifact object; return its object URL.

        Write-once by default (raises ObjectAlreadyExistsError on collision);
        set allow_overwrite=True for mutable data (e.g. rolling run snapshots).
        """
        key = self._object_key(artifact_type, id, version, object_name)
        return get_config().get_object_store_for(id).put(key, data, content_type, allow_overwrite)

    def put_object_file(
        self,
        artifact_type: str,
        id: str,
        version: int,
        object_name: str,
        file_path: str,
        content_type: str = "application/octet-stream",
    ) -> str:
        """Write-once upload of a local file (multipart for large files); return its object URL."""
        key = self._object_key(artifact_type, id, version, object_name)
        return get_config().get_object_store_for(id).put_file(key, file_path, content_type)

    @staticmethod
    def _object_key(artifact_type: str, id: str, version: int, object_name: str) -> str:
        return f"{get_config().get_artifact_key_prefix()}artifacts/{artifact_type}/{key_segment(id)}/{version}/{object_name}"

    def get_object(self, object_url: str) -> bytes:
        """Download an artifact object by its object_url."""
        return get_config().get_object_store_at(object_url).get(object_url)

    def put_document(self, artifact: _A) -> _A:
        """Insert an artifact document, retrying the version-allocation race.

        Concurrent writers to one id can allocate the same next_version; the
        unique index rejects the losers, so we re-allocate and retry rather
        than fail the caller (which drops a pass@k rollout).
        """

        self._doc_store.check_id(artifact.id)
        if artifact.version < 0:
            raise ValueError(f"Artifact version must be non-negative, got {artifact.version}")

        store = self._doc_store
        versioned = self._versioned_cache.for_store(store)
        version = artifact.version if artifact.version > 0 else versioned.next_version(artifact.id)
        doc = artifact.model_dump(by_alias=True)
        doc["created_at_utc"] = datetime.now(timezone.utc)
        last_exc: Optional[DuplicateKeyError] = None
        for attempt in range(_MAX_VERSION_RETRIES):
            doc["version"] = version
            try:
                store.insert(ARTIFACTS_COLLECTION, dict(doc))
                return artifact.model_copy(update={"version": version})
            except DuplicateKeyError as e:
                last_exc = e
                if attempt == _MAX_VERSION_RETRIES - 1:
                    break
                version = versioned.next_version(artifact.id)
                logger.warning(
                    "artifact %s version taken (concurrent write); retrying at v%s (%s/%s)",
                    artifact.id, version, attempt + 2, _MAX_VERSION_RETRIES,
                )
        assert last_exc is not None
        raise last_exc

    def latest_by_type(
        self, artifact_type: str, ids: Optional[list[str]] = None, limit: int = 50
    ) -> list["Artifact"]:
        """Return the latest version of each artifact of a type, newest version first.

        Optionally restricted to ``ids``. Groups all matching versions in Python
        (the DocumentStore has no aggregation), so this is a browse/CLI helper,
        not a hot path.
        """
        spellings = equivalent_types(artifact_type)
        filt = (
            Filter.of(type=artifact_type) if len(spellings) == 1
            else Filter({}).where("type", In(spellings))
        )
        if ids is not None:
            filt = filt.where("id", In(ids))
        latest: dict[str, dict] = {}
        for doc in self._doc_store.query(ARTIFACTS_COLLECTION, filt):
            current = latest.get(doc["id"])
            if current is None or doc["version"] > current["version"]:
                latest[doc["id"]] = doc
        ordered = sorted(latest.values(), key=lambda d: (-d["version"], d["id"]))[:limit]
        return [self._deserialize(d) for d in ordered]

    def set_field(self, artifact_type: str, id: str, version: int, field: str, value) -> None:
        """Persist a single mutable side-channel field on an already-stored artifact doc.

        Identity is not a side channel: `id`, `version` and `type` address the document being
        updated, so setting one would rename a stored artifact out from under the filter that
        found it — and would reach the store without passing the id reservation.
        """
        if field in _IDENTITY_FIELDS:
            raise ValueError(
                f"set_field cannot change {field!r}: it identifies the document being updated, "
                f"not a side-channel field (identity: {', '.join(sorted(_IDENTITY_FIELDS))})"
            )
        self._doc_store.update(
            ARTIFACTS_COLLECTION,
            Filter.of(id=id, version=version, type=artifact_type),
            UpdateSpec(set={field: value}),
        )

    def _deserialize(self, doc: dict) -> "Artifact":
        # Remove MongoDB-specific fields not in Pydantic model
        doc.pop("_id", None)
        doc.pop("created_at_utc", None)

        artifact_type = canonical_type(doc["type"])
        cls = get_artifact_registry().get(artifact_type)
        if cls is None:
            # A plugin that claims the name says why it is missing; the config remedies are for a name none claims.
            if note := _registration.failure_note(_registration.ARTIFACTS, artifact_type):
                raise ValueError(f"Unknown artifact type {doc['type']!r}{note}")
            raise ValueError(
                f"Unknown artifact type {doc['type']!r}; register it under [artifacts].impls "
                f"in .agentenv/config.toml, or map a renamed type to its current spelling "
                f"under [artifacts].type_aliases"
            )
        return cls.model_validate(doc)

    def execute_query(self, query: ArtifactQuery) -> list["Artifact"]:
        filt, sort = to_document_query(query)
        return self._versioned.query(filt, sort, query._limit_value, query._offset_value)

    def execute_count(self, query: ArtifactQuery) -> int:
        filt, _ = to_document_query(query)
        return self._versioned.count(filt)


_artifact_store: Optional[ArtifactStore] = None


def get_artifact_store() -> ArtifactStore:
    global _artifact_store
    if _artifact_store is None:
        _artifact_store = ArtifactStore()
    return _artifact_store


def set_artifact_store(store: ArtifactStore) -> None:
    global _artifact_store
    _artifact_store = store


def reset_artifact_store() -> None:
    global _artifact_store
    _artifact_store = None
