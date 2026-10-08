"""Backend-agnostic secret store abstraction + its implementations.

``secret_store`` defines the abstraction (SecretStore); ``aws_secrets_manager_secret_store``
and ``local_secret_store`` are implementations. The first needs the ``aws`` extra, so
AwsSecretsManagerSecretStore is imported on first use and left out of ``__all__``.
``gcp_secret_manager_secret_store`` is one too, not re-exported here because it needs the
``gcp`` extra. Future backends (Vault, Azure Key Vault, K8s secrets, ...) live alongside as new
modules.
"""

from typing import TYPE_CHECKING

from agent_env.store._lazy import lazy_backends
from agent_env.store.secret_store.local_secret_store import LocalSecretStore
from agent_env.store.secret_store.secret_store import SecretStore

__all__ = [
    "SecretStore",
    "LocalSecretStore",
]

if TYPE_CHECKING:  # type checkers see the class; at runtime __getattr__ imports it on first use
    from agent_env.store.secret_store.aws_secrets_manager_secret_store import (
        AwsSecretsManagerSecretStore as AwsSecretsManagerSecretStore,
    )

__getattr__ = lazy_backends(
    __name__, {"AwsSecretsManagerSecretStore": "agent_env.store.secret_store.aws_secrets_manager_secret_store"}
)
