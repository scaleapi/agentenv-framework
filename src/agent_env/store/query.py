"""Base query builder for AgentEnv stores."""

from abc import ABC, abstractmethod
from typing import Generic, Optional, Self, TypeVar

from agent_env.store.document_store.document_store import (
    Eq,
    Filter,
    Gte,
    In,
    Lte,
    Predicate,
    Sort,
)

T = TypeVar("T")


class QueryBuilder(Generic[T], ABC):
    """Base query builder with common operations. Immutable - each method returns a new instance.

    Example:
        results = (
            Artifact.query()
            .type("docker_image")
            .sort("version", descending=True)
            .limit(10)
            .execute()
        )
    """

    def __init__(self) -> None:
        self._filters: dict = {}
        self._sort_field: Optional[str] = None
        self._sort_desc: bool = True
        self._limit_value: Optional[int] = None
        self._offset_value: Optional[int] = None

    def _clone(self) -> Self:
        clone = self.__class__()
        clone._filters = self._filters.copy()
        clone._sort_field = self._sort_field
        clone._sort_desc = self._sort_desc
        clone._limit_value = self._limit_value
        clone._offset_value = self._offset_value
        return clone

    def _add_filter(self, key: str, value) -> Self:
        clone = self._clone()
        clone._filters[key] = value
        return clone

    def id(self, id: str) -> Self:
        return self._add_filter("id", id)

    def version(self, version: int) -> Self:
        return self._add_filter("version", version)

    def version_gte(self, version: int) -> Self:
        return self._add_filter("version_gte", version)

    def version_lte(self, version: int) -> Self:
        return self._add_filter("version_lte", version)

    def sort(self, field: str, descending: bool = True) -> Self:
        clone = self._clone()
        clone._sort_field = field
        clone._sort_desc = descending
        return clone

    def limit(self, n: int) -> Self:
        clone = self._clone()
        clone._limit_value = n
        return clone

    def offset(self, n: int) -> Self:
        clone = self._clone()
        clone._offset_value = n
        return clone

    def latest(self) -> Self:
        return self.sort("version", descending=True).limit(1)

    def first(self) -> Optional[T]:
        results = self.limit(1).execute()
        return results[0] if results else None

    def count(self) -> int:
        return self._execute_count()

    @abstractmethod
    def execute(self) -> list[T]:
        pass

    @abstractmethod
    def _execute_count(self) -> int:
        pass


_SUFFIX_PREDICATES = {"_gte": Gte, "_lte": Lte, "_in": In}


def to_document_query(qb: "QueryBuilder") -> tuple[Filter, Optional[Sort]]:
    """Compile a QueryBuilder's suffix-encoded filters + sort into a Filter/Sort."""
    conditions: dict[str, list[Predicate]] = {}
    for key, value in qb._filters.items():
        suffix = next((s for s in _SUFFIX_PREDICATES if key.endswith(s)), None)
        field = key.removesuffix(suffix) if suffix else key
        pred = _SUFFIX_PREDICATES.get(suffix, Eq)
        conditions.setdefault(field, []).append(pred(value))
    sort = Sort.by(qb._sort_field, descending=qb._sort_desc) if qb._sort_field else None
    return Filter(conditions), sort
