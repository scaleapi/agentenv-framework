"""Backend-agnostic document store abstraction + its implementations.

``document_store`` defines the abstraction (Filter/UpdateSpec/Sort/DocumentStore/
VersionedEntityStore); ``mongo_document_store``, ``sqlite_document_store`` and
``dynamodb_document_store`` are implementations. ``dynamodb_document_store`` needs the ``aws``
extra, so DynamoDbDocumentStore is imported on first use and left out of ``__all__``.
``firestore_mongo_document_store`` is one too, not re-exported here because it needs the ``gcp``
extra. Future backends (Cosmos DB, ...) live alongside as new modules.
"""

from typing import TYPE_CHECKING

from agent_env.store._lazy import lazy_backends
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

if TYPE_CHECKING:  # type checkers see the class; at runtime __getattr__ imports it on first use
    from agent_env.store.document_store.dynamodb_document_store import DynamoDbDocumentStore as DynamoDbDocumentStore

__getattr__ = lazy_backends(__name__, {"DynamoDbDocumentStore": "agent_env.store.document_store.dynamodb_document_store"})
