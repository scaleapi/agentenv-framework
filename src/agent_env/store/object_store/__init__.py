"""Backend-agnostic object store abstraction + its implementations.

``object_store`` defines the abstraction (ObjectStore); ``local``, ``s3_object_store``
and ``gcs_object_store`` are implementations. ``s3_object_store`` needs the ``aws`` extra,
so S3ObjectStore is imported on first use and left out of ``__all__``. ``gcs_object_store``
needs the ``gcp`` extra, so it is not re-exported here: an impl pointer names its module.
"""

from typing import TYPE_CHECKING

from agent_env.store._lazy import lazy_backends
from agent_env.store.object_store.local.store import LocalFilesystemObjectStore
from agent_env.store.object_store.object_store import (
    DEFAULT_CONTENT_TYPE,
    DEFAULT_GRANT_LIFETIME_SECONDS,
    MIN_GRANT_LIFETIME_SECONDS,
    ObjectMetadata,
    ObjectStore,
    UploadPolicy,
)

__all__ = [
    "ObjectStore",
    "ObjectMetadata",
    "UploadPolicy",
    "DEFAULT_CONTENT_TYPE",
    "DEFAULT_GRANT_LIFETIME_SECONDS",
    "MIN_GRANT_LIFETIME_SECONDS",
    "LocalFilesystemObjectStore",
]

if TYPE_CHECKING:  # type checkers see the class; at runtime __getattr__ imports it on first use
    from agent_env.store.object_store.s3_object_store import S3ObjectStore as S3ObjectStore

__getattr__ = lazy_backends(__name__, {"S3ObjectStore": "agent_env.store.object_store.s3_object_store"})
