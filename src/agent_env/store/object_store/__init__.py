"""Backend-agnostic object store abstraction + its implementations.

``object_store`` defines the abstraction (ObjectStore); ``s3_object_store`` and
``local_object_store`` are implementations. Future backends (GCS, MinIO, ...)
live alongside as new modules.
"""

from agent_env.store.object_store.local_object_store import LocalFilesystemObjectStore
from agent_env.store.object_store.object_store import DEFAULT_CONTENT_TYPE, ObjectMetadata, ObjectStore
from agent_env.store.object_store.s3_object_store import S3ObjectStore

__all__ = [
    "ObjectStore",
    "ObjectMetadata",
    "DEFAULT_CONTENT_TYPE",
    "S3ObjectStore",
    "LocalFilesystemObjectStore",
]
