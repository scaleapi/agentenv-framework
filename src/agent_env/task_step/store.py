"""TaskStep store for MongoDB persistence."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional, Self

from agent_env.plugins import _registration
from agent_env.store.base import NotFoundError
from agent_env.config import get_config
from agent_env.store.document_store import VersionedEntityStore, VersionedEntityStoreCache
from agent_env.store.query import QueryBuilder, to_document_query

if TYPE_CHECKING:
    from agent_env.task_step.task_step import TaskStep

logger = logging.getLogger(__name__)

TASK_STEPS_COLLECTION = "task_steps"


class TaskStepQuery(QueryBuilder["TaskStep"]):
    def __init__(self, store: Optional["TaskStepStore"] = None) -> None:
        super().__init__()
        self._store = store

    def _clone(self) -> Self:
        clone = TaskStepQuery(self._store)
        clone._filters = self._filters.copy()
        clone._sort_field = self._sort_field
        clone._sort_desc = self._sort_desc
        clone._limit_value = self._limit_value
        clone._offset_value = self._offset_value
        return clone

    def type(self, task_step_type: str) -> Self:
        return self._add_filter("type", task_step_type)

    def execute(self) -> list[TaskStep]:
        if self._store is None:
            self._store = get_task_step_store()
        return self._store.execute_query(self)

    def _execute_count(self) -> int:
        if self._store is None:
            self._store = get_task_step_store()
        return self._store.execute_count(self)


class TaskStepStore:
    def __init__(self) -> None:
        self._versioned_cache: VersionedEntityStoreCache[TaskStep] = VersionedEntityStoreCache(
            TASK_STEPS_COLLECTION, self._serialize, self._deserialize, secondary_indexes=[["type"]],
        )

    @property
    def _doc_store(self):
        return get_config().get_document_store()

    @property
    def _versioned(self) -> VersionedEntityStore[TaskStep]:
        return self._versioned_cache.for_store(self._doc_store)

    def _serialize(self, task_step: TaskStep) -> dict:
        doc = task_step.to_dict()
        doc["created_at_utc"] = datetime.now(timezone.utc)
        return doc

    def get(self, id: str, version: Optional[int] = None) -> TaskStep:
        task_step = self._versioned.get(id, version)
        if task_step is None:
            version_str = f" version={version}" if version is not None else ""
            raise NotFoundError(f"TaskStep {id}{version_str} not found")
        return task_step

    def next_version(self, id: str) -> int:
        return self._versioned.next_version(id)

    def put_document(self, task_step: TaskStep) -> TaskStep:
        task_step.version = self._versioned.put(task_step)
        return task_step

    def _deserialize(self, doc: dict) -> TaskStep:
        from agent_env.task_step.registry import get_task_step_registry

        doc.pop("_id", None)
        doc.pop("created_at_utc", None)
        task_step_type = doc["type"]
        registry = get_task_step_registry()
        cls = registry.get(task_step_type)
        if cls is None:
            raise ValueError(
                f"Unknown task step type: {task_step_type}"
                f"{_registration.failure_note(_registration.TASK_STEPS, task_step_type)}"
            )
        from agent_env.task_step.task_step import attach_retry_config

        return attach_retry_config(cls.from_dict(doc), doc)

    def execute_query(self, query: TaskStepQuery) -> list[TaskStep]:
        filt, sort = to_document_query(query)
        return self._versioned.query(filt, sort, query._limit_value, query._offset_value)

    def execute_count(self, query: TaskStepQuery) -> int:
        filt, _ = to_document_query(query)
        return self._versioned.count(filt)


_task_step_store: Optional[TaskStepStore] = None


def get_task_step_store() -> TaskStepStore:
    global _task_step_store
    if _task_step_store is None:
        _task_step_store = TaskStepStore()
    return _task_step_store


def set_task_step_store(store: TaskStepStore) -> None:
    global _task_step_store
    _task_step_store = store


def reset_task_step_store() -> None:
    global _task_step_store
    _task_step_store = None
