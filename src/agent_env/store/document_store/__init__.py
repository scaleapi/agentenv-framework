"""Backend-agnostic document store abstraction + its implementations.

``document_store`` defines the abstraction (Filter/UpdateSpec/Sort/DocumentStore/
VersionedEntityStore); ``mongo_document_store`` and ``sqlite_document_store`` are
implementations. Future backends (DynamoDB, ...) live alongside as new modules.
"""

from agent_env.store.document_store.document_store import (
    AbsentOrNull,
    DocumentStore,
    DuplicateKeyError,
    Eq,
    Exists,
    Filter,
    Gte,
    In,
    Lte,
    LteOrAbsent,
    Ne,
    Predicate,
    Sort,
    SortKey,
    UpdateSpec,
    VersionedEntityStore,
    VersionedEntityStoreCache,
    compare_and_swap,
    reject_reserved_id,
    rev_precondition,
)
from agent_env.store.document_store.mongo_document_store import MongoDocumentStore
from agent_env.store.document_store.sqlite_document_store import LocalSqliteDocumentStore

__all__ = [
    "DocumentStore",
    "DuplicateKeyError",
    "MongoDocumentStore",
    "LocalSqliteDocumentStore",
    "VersionedEntityStore",
    "VersionedEntityStoreCache",
    "compare_and_swap",
    "reject_reserved_id",
    "rev_precondition",
    "Filter",
    "Predicate",
    "Eq",
    "Ne",
    "Gte",
    "Lte",
    "In",
    "Exists",
    "LteOrAbsent",
    "AbsentOrNull",
    "Sort",
    "SortKey",
    "UpdateSpec",
]
