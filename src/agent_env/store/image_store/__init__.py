"""Backend-agnostic image (registry) store abstraction + its implementations.

``image_store`` defines the abstractions and generic OCI implementation;
``ecr_image_store`` and ``local_registry_image_store`` retain compatibility names.
"""

from agent_env.store.image_store.ecr_image_store import EcrCredentials, EcrImageStore
from agent_env.store.image_store.image_store import (
    ImageStore,
    OciRegistryImageStore,
)
from agent_env.store.image_store.local_registry_image_store import (
    LocalRegistryImageStore,
)
from agent_env.store.image_store.oci_registry_credentials import (
    OciRegistryCredentials,
    RegistryAuth,
    SecretStoreCredentials,
    normalize_registry_host,
    registry_host_from_ref,
)

__all__ = [
    "ImageStore",
    "OciRegistryImageStore",
    "RegistryAuth",
    "OciRegistryCredentials",
    "normalize_registry_host",
    "registry_host_from_ref",
    "EcrCredentials",
    "SecretStoreCredentials",
    "EcrImageStore",
    "LocalRegistryImageStore",
]
