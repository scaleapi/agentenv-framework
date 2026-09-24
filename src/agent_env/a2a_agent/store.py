"""A2A Agent store for MongoDB persistence."""
from __future__ import annotations

import dataclasses
import logging
import random
import string
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Optional, Self

from agent_env.store.base import ConcurrentModificationError, NotFoundError
from agent_env.config import get_config
from agent_env.store.document_store import (
    AbsentOrNull,
    Filter,
    LteOrAbsent,
    UpdateSpec,
    VersionedEntityStore,
    VersionedEntityStoreCache,
)
from agent_env.store.query import QueryBuilder, to_document_query

if TYPE_CHECKING:
    from agent_env.a2a_agent.a2a_agent import A2AAgent

logger = logging.getLogger(__name__)

A2A_AGENTS_COLLECTION = "a2a_agents"
A2A_AGENT_INSTANCES_COLLECTION = "a2a_agent_instances"


class A2AAgentQuery(QueryBuilder["A2AAgent"]):
    def __init__(self, store: Optional[A2AAgentStore] = None) -> None:
        super().__init__()
        self._store = store

    def _clone(self) -> Self:
        clone = A2AAgentQuery(self._store)
        clone._filters = self._filters.copy()
        clone._sort_field = self._sort_field
        clone._sort_desc = self._sort_desc
        clone._limit_value = self._limit_value
        clone._offset_value = self._offset_value
        return clone

    def execute(self) -> list[A2AAgent]:
        if self._store is None:
            self._store = get_a2a_agent_store()
        return self._store.execute_query(self)

    def _execute_count(self) -> int:
        if self._store is None:
            self._store = get_a2a_agent_store()
        return self._store.execute_count(self)


class A2AAgentStore:
    def __init__(self) -> None:
        self._versioned_cache: VersionedEntityStoreCache[A2AAgent] = VersionedEntityStoreCache(
            A2A_AGENTS_COLLECTION, self._serialize, self._deserialize, secondary_indexes=[["type"]],
        )

    @property
    def _doc_store(self):
        return get_config().get_document_store()

    @property
    def _versioned(self) -> VersionedEntityStore[A2AAgent]:
        return self._versioned_cache.for_store(self._doc_store)

    def _serialize(self, agent: A2AAgent) -> dict:
        doc = agent.to_dict()
        doc["created_at_utc"] = datetime.now(timezone.utc)
        return doc

    def get(self, id: str, version: Optional[int] = None) -> A2AAgent:
        agent = self._versioned.get(id, version)
        if agent is None:
            version_str = f" version={version}" if version is not None else ""
            raise NotFoundError(f"A2AAgent {id}{version_str} not found")
        return agent

    def next_version(self, id: str) -> int:
        return self._versioned.next_version(id)

    def put_document(self, agent: A2AAgent) -> A2AAgent:
        agent.version = self._versioned.put(agent)
        return agent

    def update_metadata(self, id: str, version: int, old_metadata: dict[str, Any], new_metadata: dict[str, Any]) -> dict[str, Any]:
        """Replace metadata using timestamp-based CAS (same pattern as EnvStore)."""
        now = datetime.now(timezone.utc).isoformat()
        old_ts = old_metadata.get("updated_at")
        new_metadata = {**new_metadata, "updated_at": now}

        ts_pred = LteOrAbsent(old_ts) if old_ts is not None else AbsentOrNull()
        store = self._doc_store
        matched = store.update(
            A2A_AGENTS_COLLECTION,
            Filter.of(id=id, version=version).where("metadata.updated_at", ts_pred),
            UpdateSpec(set={"metadata": new_metadata}),
        )

        if matched == 0:
            if store.find_one(A2A_AGENTS_COLLECTION, Filter.of(id=id, version=version)) is None:
                raise NotFoundError(f"A2AAgent {id} version={version} not found")
            raise ConcurrentModificationError(f"A2AAgent {id} version={version} metadata was modified concurrently")

        return new_metadata

    def _deserialize(self, doc: dict) -> A2AAgent:
        from agent_env.a2a_agent.a2a_agent import A2AAgent

        doc.pop("_id", None)
        doc.pop("created_at_utc", None)
        return A2AAgent.from_dict(doc)

    def execute_query(self, query: A2AAgentQuery) -> list[A2AAgent]:
        filt, sort = to_document_query(query)
        return self._versioned.query(filt, sort, query._limit_value, query._offset_value)

    def execute_count(self, query: A2AAgentQuery) -> int:
        filt, _ = to_document_query(query)
        return self._versioned.count(filt)


_a2a_agent_store: Optional[A2AAgentStore] = None


def get_a2a_agent_store() -> A2AAgentStore:
    global _a2a_agent_store
    if _a2a_agent_store is None:
        _a2a_agent_store = A2AAgentStore()
    return _a2a_agent_store


def set_a2a_agent_store(store: A2AAgentStore) -> None:
    global _a2a_agent_store
    _a2a_agent_store = store


def reset_a2a_agent_store() -> None:
    global _a2a_agent_store
    _a2a_agent_store = None


class A2AAgentInstanceStore:
    def __init__(self) -> None:
        self._indexed = None

    @property
    def _doc_store(self):
        # Resolved per call: a cached store outlives reset_config(), so a process that
        # re-pointed would read the new config and write the old backend.
        store = get_config().get_document_store()
        if self._indexed is not store:
            store.ensure_index(A2A_AGENT_INSTANCES_COLLECTION, ["instance_id"], unique=True)
            self._indexed = store
        return store

    def create_instance(self, deployed: "DeployedA2AAgent", ttl_seconds: int) -> "DeployedA2AAgent":
        from agent_env.a2a_agent.a2a_agent import DeployedA2AAgent
        suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
        instance_id = f"{deployed.agent_id}-{suffix}"
        now = datetime.now(timezone.utc)
        created_at_utc = now.strftime("%Y-%m-%d %H:%M UTC")
        expires_at_utc = (now + timedelta(seconds=ttl_seconds)).strftime("%Y-%m-%d %H:%M UTC")
        doc = dataclasses.asdict(deployed) | {
            "instance_id": instance_id,
            "created_at_utc": created_at_utc,
            "expires_at_utc": expires_at_utc,
        }
        self._doc_store.insert(A2A_AGENT_INSTANCES_COLLECTION, doc)
        return DeployedA2AAgent.from_dict(doc)

    def get(self, instance_id: str) -> "DeployedA2AAgent":
        from agent_env.a2a_agent.a2a_agent import DeployedA2AAgent
        doc = self._doc_store.find_one(A2A_AGENT_INSTANCES_COLLECTION, Filter.of(instance_id=instance_id))
        if not doc:
            raise NotFoundError(f"A2AAgentInstance '{instance_id}' not found")
        return DeployedA2AAgent.from_dict(doc)


_a2a_agent_instance_store: Optional[A2AAgentInstanceStore] = None


def get_a2a_agent_instance_store() -> A2AAgentInstanceStore:
    global _a2a_agent_instance_store
    if _a2a_agent_instance_store is None:
        _a2a_agent_instance_store = A2AAgentInstanceStore()
    return _a2a_agent_instance_store


def set_a2a_agent_instance_store(store: A2AAgentInstanceStore) -> None:
    global _a2a_agent_instance_store
    _a2a_agent_instance_store = store


def reset_a2a_agent_instance_store() -> None:
    global _a2a_agent_instance_store
    _a2a_agent_instance_store = None
