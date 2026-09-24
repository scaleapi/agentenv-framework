"""Eval store for MongoDB persistence."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional, Self

from agent_env.store.base import NotFoundError
from agent_env.config import get_config
from agent_env.store.document_store import VersionedEntityStore, VersionedEntityStoreCache
from agent_env.store.query import QueryBuilder, to_document_query

if TYPE_CHECKING:
    from agent_env.eval.eval import Eval

logger = logging.getLogger(__name__)

EVALS_COLLECTION = "evals"


class EvalQuery(QueryBuilder["Eval"]):
    def __init__(self, store: Optional["EvalStore"] = None) -> None:
        super().__init__()
        self._store = store

    def _clone(self) -> Self:
        clone = EvalQuery(self._store)
        clone._filters = self._filters.copy()
        clone._sort_field = self._sort_field
        clone._sort_desc = self._sort_desc
        clone._limit_value = self._limit_value
        clone._offset_value = self._offset_value
        return clone

    def type(self, eval_type: str) -> Self:
        return self._add_filter("type", eval_type)

    def execute(self) -> list[Eval]:
        if self._store is None:
            self._store = get_eval_store()
        return self._store.execute_query(self)

    def _execute_count(self) -> int:
        if self._store is None:
            self._store = get_eval_store()
        return self._store.execute_count(self)


class EvalStore:
    def __init__(self) -> None:
        self._versioned_cache: VersionedEntityStoreCache[Eval] = VersionedEntityStoreCache(
            EVALS_COLLECTION, self._serialize, self._deserialize, secondary_indexes=[["type"]],
        )

    @property
    def _doc_store(self):
        return get_config().get_document_store()

    @property
    def _versioned(self) -> VersionedEntityStore[Eval]:
        return self._versioned_cache.for_store(self._doc_store)

    def _serialize(self, eval_obj: Eval) -> dict:
        doc = eval_obj.to_dict()
        doc["created_at_utc"] = datetime.now(timezone.utc)
        return doc

    def get(self, id: str, version: Optional[int] = None) -> Eval:
        eval_obj = self._versioned.get(id, version)
        if eval_obj is None:
            version_str = f" version={version}" if version is not None else ""
            raise NotFoundError(f"Eval {id}{version_str} not found")
        return eval_obj

    def next_version(self, id: str) -> int:
        return self._versioned.next_version(id)

    def put_document(self, eval_obj: Eval) -> Eval:
        eval_obj.version = self._versioned.put(eval_obj)
        return eval_obj

    def _deserialize(self, doc: dict) -> Eval:
        from agent_env.eval.eval import Eval

        doc.pop("_id", None)
        doc.pop("created_at_utc", None)
        return Eval.from_dict(doc)

    def execute_query(self, query: EvalQuery) -> list[Eval]:
        filt, sort = to_document_query(query)
        return self._versioned.query(filt, sort, query._limit_value, query._offset_value)

    def execute_count(self, query: EvalQuery) -> int:
        filt, _ = to_document_query(query)
        return self._versioned.count(filt)


_eval_store: Optional[EvalStore] = None


def get_eval_store() -> EvalStore:
    global _eval_store
    if _eval_store is None:
        _eval_store = EvalStore()
    return _eval_store


def set_eval_store(store: EvalStore) -> None:
    global _eval_store
    _eval_store = store


def reset_eval_store() -> None:
    global _eval_store
    _eval_store = None
