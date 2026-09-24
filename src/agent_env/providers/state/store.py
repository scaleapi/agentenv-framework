"""Store for ``EnvStateInstance`` records (collection ``env_state_instances``)."""

from __future__ import annotations

import logging
import random
import string
from datetime import datetime, timedelta, timezone
from typing import Optional

from agent_env.providers.state.env_state_provider import EnvStateInstance, TS_FMT
from agent_env.store.base import NotFoundError
from agent_env.config import get_config
from agent_env.store.document_store import Filter, UpdateSpec

logger = logging.getLogger(__name__)

ENV_STATE_INSTANCES_COLLECTION = "env_state_instances"
_TS_FMT = TS_FMT


class EnvStateInstanceStore:
    def __init__(self) -> None:
        self._indexed = None

    @property
    def _doc_store(self):
        # Resolved per call: a cached store outlives reset_config(), so a process that
        # re-pointed would read the new config and write the old backend.
        store = get_config().get_document_store()
        if self._indexed is not store:
            store.ensure_index(
                ENV_STATE_INSTANCES_COLLECTION, ["instance_id"], unique=True
            )
            self._indexed = store
        return store

    def create_instance(self, instance: EnvStateInstance, ttl_seconds: int) -> EnvStateInstance:
        now = datetime.now(timezone.utc)
        if not instance.instance_id:
            suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
            instance.instance_id = f"esi-{suffix}"
        instance.created_at_utc = now.strftime(_TS_FMT)
        instance.expires_at_utc = (now + timedelta(seconds=ttl_seconds)).strftime(_TS_FMT)
        self._doc_store.insert(ENV_STATE_INSTANCES_COLLECTION, instance.to_dict())
        return instance

    def get(self, instance_id: str) -> EnvStateInstance:
        doc = self._doc_store.find_one(
            ENV_STATE_INSTANCES_COLLECTION, Filter.of(instance_id=instance_id)
        )
        if not doc:
            raise NotFoundError(f"EnvStateInstance '{instance_id}' not found")
        return EnvStateInstance.from_dict(doc)

    def set_expired_now(self, instance_id: str) -> None:
        """Retire by pulling ``expires_at_utc`` forward to now (RETIRED ⟺ expires <= now)."""
        now = datetime.now(timezone.utc).strftime(_TS_FMT)
        self._doc_store.update(
            ENV_STATE_INSTANCES_COLLECTION,
            Filter.of(instance_id=instance_id),
            UpdateSpec(set={"expires_at_utc": now}),
        )

    def set_metadata(self, instance_id: str, metadata: dict) -> None:
        self._doc_store.update(
            ENV_STATE_INSTANCES_COLLECTION,
            Filter.of(instance_id=instance_id),
            UpdateSpec(set={"metadata": metadata}),
        )


_store: Optional[EnvStateInstanceStore] = None


def get_env_state_instance_store() -> EnvStateInstanceStore:
    global _store
    if _store is None:
        _store = EnvStateInstanceStore()
    return _store


def set_env_state_instance_store(store: EnvStateInstanceStore) -> None:
    global _store
    _store = store


def reset_env_state_instance_store() -> None:
    global _store
    _store = None


def register_env_state_instance(instance: EnvStateInstance, ttl_seconds: int) -> EnvStateInstance:
    try:
        return get_env_state_instance_store().create_instance(instance, ttl_seconds)
    except Exception:
        logger.warning("Failed to persist env state instance %s", instance.instance_id, exc_info=True)
        return instance


def retire_env_state_instance(instance_id: str) -> None:
    try:
        get_env_state_instance_store().set_expired_now(instance_id)
    except Exception:
        logger.warning("Failed to retire env state instance %s", instance_id, exc_info=True)


def update_env_state_instance_metadata(instance_id: str, metadata: dict) -> None:
    try:
        get_env_state_instance_store().set_metadata(instance_id, metadata)
    except Exception:
        logger.warning("Failed to update metadata for env state instance %s", instance_id, exc_info=True)
