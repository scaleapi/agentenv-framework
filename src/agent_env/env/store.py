"""Env store for MongoDB persistence."""

from __future__ import annotations

import dataclasses
import logging
import random
import string
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Optional, Self

from agent_env.plugins import _registration
from agent_env.store.base import ConcurrentModificationError, NotFoundError
from agent_env.config import get_config
from agent_env.store.document_store import (
    AbsentOrNull,
    DocumentStore,
    Filter,
    LteOrAbsent,
    Sort,
    UpdateSpec,
    VersionedEntityStore,
    VersionedEntityStoreCache,
)
from agent_env.store.query import QueryBuilder, to_document_query

if TYPE_CHECKING:
    from agent_env.env.env import DeployedEnv, Env

logger = logging.getLogger(__name__)

ENVS_COLLECTION = "envs"
ENV_INSTANCES_COLLECTION = "env_instances"


class EnvQuery(QueryBuilder["Env"]):
    def __init__(self, store: Optional["EnvStore"] = None) -> None:
        super().__init__()
        self._store = store

    def _clone(self) -> Self:
        clone = EnvQuery(self._store)
        clone._filters = self._filters.copy()
        clone._sort_field = self._sort_field
        clone._sort_desc = self._sort_desc
        clone._limit_value = self._limit_value
        clone._offset_value = self._offset_value
        return clone

    def type(self, env_type: str) -> Self:
        return self._add_filter("type", env_type)

    def execute(self) -> list[Env]:
        if self._store is None:
            self._store = get_env_store()
        return self._store.execute_query(self)

    def _execute_count(self) -> int:
        if self._store is None:
            self._store = get_env_store()
        return self._store.execute_count(self)


class EnvStore:
    def __init__(self) -> None:
        self._versioned_cache: VersionedEntityStoreCache[Env] = VersionedEntityStoreCache(
            ENVS_COLLECTION, self._serialize, self._deserialize, secondary_indexes=[["type"]],
        )

    @property
    def _doc_store(self):
        return get_config().get_document_store()

    @property
    def _versioned(self) -> VersionedEntityStore[Env]:
        return self._versioned_cache.for_store(self._doc_store)

    def _serialize(self, env: Env) -> dict:
        doc = env.to_dict()
        doc["created_at_utc"] = datetime.now(timezone.utc)
        return doc

    def get(self, id: str, version: Optional[int] = None) -> Env:
        env = self._versioned.get(id, version)
        if env is None:
            version_str = f" version={version}" if version is not None else ""
            raise NotFoundError(f"Env {id}{version_str} not found")
        return env

    def next_version(self, id: str) -> int:
        return self._versioned.next_version(id)

    def put_document(self, env: Env) -> Env:
        store = self._doc_store
        existing = store.find_one(
            ENVS_COLLECTION, Filter.of(id=env.id), sort=Sort.by("version", descending=True)
        )
        if existing and existing.get("type") != env.type:
            raise ValueError(f"Env id '{env.id}' already exists as type '{existing.get('type')}'; cannot put as type '{env.type}'.")
        env.version = self._versioned_cache.for_store(store).put(env)
        return env

    def update_metadata(
        self,
        id: str,
        version: int,
        old_metadata: dict[str, Any],
        new_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Replace an env's metadata using a timestamp-based compare-and-swap.

        The update succeeds if the document's ``metadata.updated_at`` is older
        than or equal to the caller's snapshot, or if the field does not yet
        exist.  On success both ``metadata`` and its ``updated_at`` field are
        written atomically.

        Args:
            id: Env ID.
            version: Env version.
            old_metadata: Previously loaded metadata (used to read the old timestamp).
            new_metadata: The new metadata to set.

        Raises:
            NotFoundError: If (id, version) does not exist.
            ConcurrentModificationError: If metadata was modified by a newer writer.
        """
        now = datetime.now(timezone.utc).isoformat()
        old_ts = old_metadata.get("updated_at")
        new_metadata = {**new_metadata, "updated_at": now}

        ts_pred = LteOrAbsent(old_ts) if old_ts is not None else AbsentOrNull()
        store = self._doc_store
        matched = store.update(
            ENVS_COLLECTION,
            Filter.of(id=id, version=version).where("metadata.updated_at", ts_pred),
            UpdateSpec(set={"metadata": new_metadata}),
        )

        if matched == 0:
            if store.find_one(ENVS_COLLECTION, Filter.of(id=id, version=version)) is None:
                raise NotFoundError(f"Env {id} version={version} not found")
            raise ConcurrentModificationError(
                f"Env {id} version={version} metadata was modified concurrently"
            )

        return new_metadata

    def _deserialize(self, doc: dict) -> Env:
        from agent_env.env.registry import get_env_registry

        doc.pop("_id", None)
        doc.pop("created_at_utc", None)
        env_type = doc["type"]
        registry = get_env_registry()
        cls = registry.get(env_type)
        if cls is None:
            raise ValueError(f"Unknown env type: {env_type}{_registration.failure_note(_registration.ENVS, env_type)}")
        return cls.from_dict(doc)

    def execute_query(self, query: EnvQuery) -> list[Env]:
        filt, sort = to_document_query(query)
        return self._versioned.query(filt, sort, query._limit_value, query._offset_value)

    def execute_count(self, query: EnvQuery) -> int:
        filt, _ = to_document_query(query)
        return self._versioned.count(filt)


_env_store: Optional[EnvStore] = None


def get_env_store() -> EnvStore:
    global _env_store
    if _env_store is None:
        _env_store = EnvStore()
    return _env_store


def set_env_store(store: EnvStore) -> None:
    global _env_store
    _env_store = store


def reset_env_store() -> None:
    global _env_store
    _env_store = None


# --- EnvInstance store (non-versioned, tracks deployed env instances) ---


class EnvInstanceStore:
    def __init__(self) -> None:
        self._indexed = None

    @property
    def _doc_store(self):
        # Resolved per call: a cached store outlives reset_config(), so a process that
        # re-pointed would read the new config and write the old backend.
        store = get_config().get_document_store()
        if self._indexed is not store:
            store.ensure_index(ENV_INSTANCES_COLLECTION, ["instance_id"], unique=True)
            self._indexed = store
        return store

    def create_instance(self, deployed_env: DeployedEnv, ttl_seconds: int) -> DeployedEnv:
        """Persist a deployed env as an instance record. Returns a new DeployedEnv with instance_id set."""
        from agent_env.env.env import DeployedEnv as DeployedEnvCls

        suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
        instance_id = f"{deployed_env.env_id}-{suffix}"
        now = datetime.now(timezone.utc)
        created_at_utc = now.strftime("%Y-%m-%d %H:%M UTC")
        expires_at_utc = (now + timedelta(seconds=ttl_seconds)).strftime("%Y-%m-%d %H:%M UTC")
        doc = dataclasses.asdict(deployed_env) | {
            "instance_id": instance_id,
            "created_at_utc": created_at_utc,
            "expires_at_utc": expires_at_utc,
        }
        self._doc_store.insert(ENV_INSTANCES_COLLECTION, doc)
        return DeployedEnvCls.from_dict(doc)

    def get(self, instance_id: str) -> DeployedEnv:
        """Look up a deployed env by instance ID."""
        from agent_env.env.env import DeployedEnv as DeployedEnvCls

        doc = self._doc_store.find_one(ENV_INSTANCES_COLLECTION, Filter.of(instance_id=instance_id))
        if not doc:
            raise NotFoundError(f"EnvInstance '{instance_id}' not found")
        return DeployedEnvCls.from_dict(doc)

    def set_environment_universe(self, instance_id: str, artifact_id: str, artifact_version: int) -> None:
        """Record which EnvironmentUniverseArtifact was loaded into this instance.

        Writes both spellings (dual-write). Unlike every other
        renamed surface — all of which are versioned immutable puts, where an
        old-build writer merely omits the new key and a backfill catch-up repairs
        it — this one mutates in place. An old-build writer here overwrites only
        ``service_universe``, leaving the pair *present but disagreeing*, which no
        ``{$exists: false}`` check can detect. Hence the dual-read below.
        """
        subdoc = {"id": artifact_id, "version": artifact_version}
        self._doc_store.update(
            ENV_INSTANCES_COLLECTION,
            Filter.of(instance_id=instance_id),
            UpdateSpec(set={"service_universe": subdoc, "environment_universe": subdoc}),
        )

    def get_loaded_environments(self, instance_id: str, universe_id: str, universe_version: int) -> list[str]:
        """Which of a universe's services are already loaded into this instance.

        Durable rather than context-carried on purpose. The Temporal worker snapshots the
        step context BEFORE running the step and re-sends that frozen copy on every
        heartbeat, so progress recorded mid-step never reaches the heartbeat a retry
        restores from. Mongo is the only place this survives an activity retry.

        Keyed by universe id AND version, so loading a different universe -- or a new
        version of the same one -- correctly resumes from nothing.
        """
        return self._loaded_environments(self._doc_store, instance_id, universe_id, universe_version)

    def _loaded_environments(
        self, store: DocumentStore, instance_id: str, universe_id: str, universe_version: int,
    ) -> list[str]:
        doc = store.find_one(ENV_INSTANCES_COLLECTION, Filter.of(instance_id=instance_id))
        if not doc:
            return []
        progress = doc.get("loaded_environments") or {}
        if not isinstance(progress, dict):
            return []
        names = progress.get(f"{universe_id}:{universe_version}") or []
        return [n for n in names if isinstance(n, str)]

    def record_loaded_environment(
        self, instance_id: str, universe_id: str, universe_version: int, environment_name: str,
    ) -> None:
        """Mark one service as loaded. Called only after its data add has succeeded.

        "After" is load-bearing: a service whose reset succeeded but whose re-add failed is
        EMPTY, and recording it earlier would let a retry skip it and leave the env silently
        missing data.
        """
        key = f"loaded_environments.{universe_id}:{universe_version}"
        store = self._doc_store
        existing = set(self._loaded_environments(store, instance_id, universe_id, universe_version))
        existing.add(environment_name)
        store.update(
            ENV_INSTANCES_COLLECTION,
            Filter.of(instance_id=instance_id),
            UpdateSpec(set={key: sorted(existing)}),
        )

    def clear_loaded_environments(self, instance_id: str) -> None:
        """Drop all resume state for an instance (a snapshot restore replaces everything)."""
        self._doc_store.update(
            ENV_INSTANCES_COLLECTION,
            Filter.of(instance_id=instance_id),
            UpdateSpec(set={"loaded_environments": {}}),
        )

    def merge_metadata(self, instance_id: str, values: dict) -> None:
        """Merge key/values into an instance record's ``metadata``.

        ``create_instance`` runs inside ``Env.deploy()``, so anything set on the
        DeployedEnv *after* deploy() returns lands only on the in-memory object
        and never reaches the record. Callers that need a value persisted use
        this instead of assigning to ``deployed_env.metadata``.

        Two writes, one of which applies. ``create_instance`` persists
        ``asdict(deployed_env)``, so an unset metadata field is stored as **null**
        rather than being absent — and MongoDB refuses to create a child under a
        scalar, failing ``$set {"metadata.x": ...}`` with PathNotViable. Callers
        treat this helper as best-effort, so that error would be swallowed and the
        annotation lost precisely on the records that need it. Backends disagree
        here too (the local sqlite store rewrites the null, mongomock silently
        no-ops), so the null shape is normalised explicitly instead of relying on
        any of them.

        The seed writes the whole subdocument, and only matches when metadata is
        null or absent. Otherwise the dotted merge runs, which sets keys
        individually so concurrent writers of different keys don't clobber each
        other.
        """
        if not values:
            return
        store = self._doc_store
        seeded = store.update(
            ENV_INSTANCES_COLLECTION,
            Filter.of(instance_id=instance_id).where("metadata", AbsentOrNull()),
            UpdateSpec(set={"metadata": dict(values)}),
        )
        if seeded:
            return
        store.update(
            ENV_INSTANCES_COLLECTION,
            Filter.of(instance_id=instance_id),
            UpdateSpec(set={f"metadata.{key}": value for key, value in values.items()}),
        )

    def get_environment_universe(self, instance_id: str) -> dict | None:
        """Return the loaded-universe dict for an instance, or None if not set.

        Reads the legacy key first, deliberately: every writer at every SDK
        version sets it, so it is never the stale half of a divergent pair. The
        new key is a fallback for docs written once the old one is dropped.
        """
        doc = self._doc_store.find_one(ENV_INSTANCES_COLLECTION, Filter.of(instance_id=instance_id))
        if not doc:
            raise NotFoundError(f"EnvInstance '{instance_id}' not found")
        return doc.get("service_universe") or doc.get("environment_universe")


def register_env_instance(deployed_env: DeployedEnv, ttl_seconds: int) -> DeployedEnv:
    try:
        return get_env_instance_store().create_instance(deployed_env, ttl_seconds=ttl_seconds)
    except Exception:
        logger.warning("Failed to create env instance record", exc_info=True)
        return deployed_env


def update_env_instance_metadata(instance_id: str, values: dict) -> None:
    """Persist metadata onto an existing env instance record (best-effort).

    Non-fatal like ``register_env_instance``: instance records are an
    observability/attribution surface, so failing to annotate one must not fail a
    deploy that otherwise succeeded.
    """
    try:
        get_env_instance_store().merge_metadata(instance_id, values)
    except Exception:
        logger.warning("Failed to update env instance metadata", exc_info=True)


def update_env_instance_environment_universe(instance_id: str, artifact_id: str, artifact_version: int) -> None:
    try:
        get_env_instance_store().set_environment_universe(instance_id, artifact_id, artifact_version)
    except Exception:
        logger.warning("Failed to update env instance service_universe", exc_info=True)


_env_instance_store: Optional[EnvInstanceStore] = None


def get_env_instance_store() -> EnvInstanceStore:
    global _env_instance_store
    if _env_instance_store is None:
        _env_instance_store = EnvInstanceStore()
    return _env_instance_store


def set_env_instance_store(store: EnvInstanceStore) -> None:
    global _env_instance_store
    _env_instance_store = store


def reset_env_instance_store() -> None:
    global _env_instance_store
    _env_instance_store = None
