"""Store infrastructure for AgentEnv persistence layer"""

from typing import TYPE_CHECKING

from agent_env.store.base import (
    ConcurrentModificationError,
    ConfigError,
    GrantUnavailableError,
    NotFoundError,
    ObjectAlreadyExistsError,
    ObjectNotFoundError,
    UploadFailedError,
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
from agent_env.store._lazy import lazy_backends
from agent_env.store.object_store import (
    LocalFilesystemObjectStore,
    ObjectMetadata,
    ObjectStore,
)
from agent_env.store.query import QueryBuilder
from agent_env.store.secret_store import (
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
    "ObjectNotFoundError",
    "GrantUnavailableError",
    "UploadFailedError",
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
    "LocalFilesystemObjectStore",
    # Secret store abstraction
    "SecretStore",
    "LocalSecretStore",
]

if TYPE_CHECKING:  # type checkers see the classes; at runtime __getattr__ imports them on first use
    from agent_env.store.document_store.dynamodb_document_store import DynamoDbDocumentStore as DynamoDbDocumentStore
    from agent_env.store.object_store.s3_object_store import S3ObjectStore as S3ObjectStore
    from agent_env.store.secret_store.aws_secrets_manager_secret_store import (
        AwsSecretsManagerSecretStore as AwsSecretsManagerSecretStore,
    )

# The backends that need the aws extra: imported on first use, and left out of __all__.
__getattr__ = lazy_backends(__name__, {
    "DynamoDbDocumentStore": "agent_env.store.document_store.dynamodb_document_store",
    "S3ObjectStore": "agent_env.store.object_store.s3_object_store",
    "AwsSecretsManagerSecretStore": "agent_env.store.secret_store.aws_secrets_manager_secret_store",
})
