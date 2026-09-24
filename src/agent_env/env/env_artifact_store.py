"""Store for env-artifact relationship data (e.g. universe compatibility results)."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from agent_env.config import get_config
from agent_env.store.document_store import Filter, UpdateSpec


class EnvArtifactType(str, Enum):
    UNIVERSE_COMPATIBILITY = "universe_compatibility"

ENV_ARTIFACTS_COLLECTION = "env_artifacts"


class EnvArtifactStore:
    def __init__(self) -> None:
        self._indexed = None

    @property
    def _doc_store(self):
        # Resolved per call: a cached store outlives reset_config(), so a process that
        # re-pointed would read the new config and write the old backend.
        store = get_config().get_document_store()
        if self._indexed is not store:
            store.ensure_index(
                ENV_ARTIFACTS_COLLECTION,
                ["env_id", "env_version", "artifact_id", "artifact_version", "type"],
                unique=True,
            )
            store.ensure_index(ENV_ARTIFACTS_COLLECTION, ["artifact_id", "type"])
            self._indexed = store
        return store

    def put(self, env_id: str, env_version: int, artifact_id: str, artifact_version: int, type: str, data: dict[str, Any]) -> None:
        filter_key = {"env_id": env_id, "env_version": env_version, "artifact_id": artifact_id, "artifact_version": artifact_version, "type": type}
        self._doc_store.update(
            ENV_ARTIFACTS_COLLECTION,
            Filter.of(**filter_key),
            UpdateSpec(set={**filter_key, "data": data, "created_at_utc": datetime.now(timezone.utc).isoformat()}),
            upsert=True,
        )

    def get(self, env_id: str, env_version: int, artifact_id: str, artifact_version: int, type: str) -> Optional[dict[str, Any]]:
        return self._doc_store.find_one(
            ENV_ARTIFACTS_COLLECTION,
            Filter.of(env_id=env_id, env_version=env_version, artifact_id=artifact_id, artifact_version=artifact_version, type=type),
        )

    def get_by_env(self, env_id: str, env_version: int | None = None, type: str | None = None) -> list[dict[str, Any]]:
        query: dict[str, Any] = {"env_id": env_id}
        if env_version is not None:
            query["env_version"] = env_version
        if type is not None:
            query["type"] = type
        return self._doc_store.query(ENV_ARTIFACTS_COLLECTION, Filter.of(**query))

    def get_by_artifact(self, artifact_id: str, artifact_version: int | None = None, type: str | None = None) -> list[dict[str, Any]]:
        query: dict[str, Any] = {"artifact_id": artifact_id}
        if artifact_version is not None:
            query["artifact_version"] = artifact_version
        if type is not None:
            query["type"] = type
        return self._doc_store.query(ENV_ARTIFACTS_COLLECTION, Filter.of(**query))


_env_artifact_store: Optional[EnvArtifactStore] = None


def get_env_artifact_store() -> EnvArtifactStore:
    global _env_artifact_store
    if _env_artifact_store is None:
        _env_artifact_store = EnvArtifactStore()
    return _env_artifact_store


def set_env_artifact_store(store: EnvArtifactStore) -> None:
    global _env_artifact_store
    _env_artifact_store = store


def reset_env_artifact_store() -> None:
    global _env_artifact_store
    _env_artifact_store = None
