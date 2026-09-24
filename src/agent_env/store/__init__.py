"""Store infrastructure for AgentEnv persistence layer"""

from agent_env.store.base import (
    ConcurrentModificationError,
    ConfigError,
    NotFoundError,
    ObjectAlreadyExistsError,
)
from agent_env.config import (
    Config,
    configure,
    get_config,
    reset_config,
    set_document_store,
    set_image_store,
    set_object_store,
    set_secret_store,
)
from agent_env.store.document_store import (
    AbsentOrNull,
    DocumentStore,
    DuplicateKeyError,
    Eq,
    Exists,
    Filter,
    Gte,
    In,
    LocalSqliteDocumentStore,
    Lte,
    LteOrAbsent,
    MongoDocumentStore,
    Ne,
    Predicate,
    Sort,
    SortKey,
    UpdateSpec,
    VersionedEntityStore,
    compare_and_swap,
    rev_precondition,
)
from agent_env.store.image_store import (
    EcrCredentials,
    EcrImageStore,
    ImageStore,
    LocalRegistryImageStore,
    OciRegistryCredentials,
    OciRegistryImageStore,
    RegistryAuth,
    SecretStoreCredentials,
)
from agent_env.store.object_store import (
    LocalFilesystemObjectStore,
    ObjectMetadata,
    ObjectStore,
    S3ObjectStore,
)
from agent_env.store.query import QueryBuilder
from agent_env.store.secret_store import (
    AwsSecretsManagerSecretStore,
    LocalSecretStore,
    SecretStore,
)

__all__ = [
    # Configuration
    "Config",
    "configure",
    "get_config",
    "reset_config",
    "set_document_store",
    "set_image_store",
    "set_object_store",
    "set_secret_store",
    # Base classes
    "QueryBuilder",
    # Exceptions
    "NotFoundError",
    "ObjectAlreadyExistsError",
    "ConcurrentModificationError",
    "ConfigError",
    "DuplicateKeyError",
    # Document store abstraction
    "DocumentStore",
    "MongoDocumentStore",
    "LocalSqliteDocumentStore",
    "VersionedEntityStore",
    "compare_and_swap",
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
    # Image store abstraction
    "ImageStore",
    "OciRegistryImageStore",
    "RegistryAuth",
    "OciRegistryCredentials",
    "EcrCredentials",
    "SecretStoreCredentials",
    "EcrImageStore",
    "LocalRegistryImageStore",
    # Object store abstraction
    "ObjectStore",
    "ObjectMetadata",
    "S3ObjectStore",
    "LocalFilesystemObjectStore",
    # Secret store abstraction
    "SecretStore",
    "AwsSecretsManagerSecretStore",
    "LocalSecretStore",
]
