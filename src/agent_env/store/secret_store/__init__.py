"""Backend-agnostic secret store abstraction + its implementations.

``secret_store`` defines the abstraction (SecretStore); ``aws_secrets_manager_secret_store``
and ``local_secret_store`` are implementations. Future backends (Vault, GCP/Azure
secret managers, K8s secrets, ...) live alongside as new modules.
"""

from agent_env.store.secret_store.aws_secrets_manager_secret_store import (
    AwsSecretsManagerSecretStore,
)
from agent_env.store.secret_store.local_secret_store import LocalSecretStore
from agent_env.store.secret_store.secret_store import SecretStore

__all__ = [
    "SecretStore",
    "AwsSecretsManagerSecretStore",
    "LocalSecretStore",
]
