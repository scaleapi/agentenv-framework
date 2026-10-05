"""Backend-agnostic object store abstraction + its implementations.

``object_store`` defines the abstraction (ObjectStore); ``s3_object_store``,
``local`` and ``gcs_object_store`` are implementations. The last needs
the ``gcp`` extra, so it is not re-exported here: an impl pointer names its module.
"""

from agent_env.store.object_store.local.store import LocalFilesystemObjectStore
from agent_env.store.object_store.object_store import (
    DEFAULT_CONTENT_TYPE,
    ObjectMetadata,
    ObjectStore,
    UploadPolicy,
)
from agent_env.store.object_store.s3_object_store import S3ObjectStore

__all__ = [
    "ObjectStore",
    "ObjectMetadata",
    "UploadPolicy",
    "DEFAULT_CONTENT_TYPE",
    "S3ObjectStore",
    "LocalFilesystemObjectStore",
]
