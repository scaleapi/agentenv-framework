"""ECR implementation of the ImageStore interface."""

from __future__ import annotations

import base64
import binascii
import logging
from typing import Any

import boto3
from botocore.exceptions import ClientError

from agent_env.config.errors import ConfigError
from agent_env.store.image_store.image_store import OciRegistryImageStore
from agent_env.store.image_store.oci_registry_credentials import (
    OciRegistryCredentials,
    RegistryAuth,
    normalize_registry_host,
)

logger = logging.getLogger(__name__)


class EcrCredentials(OciRegistryCredentials):
    """Mint short-lived Docker credentials through the AWS ECR API."""

    def __init__(
        self,
        region: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        *,
        client: Any = None,
    ) -> None:
        if (access_key is None) != (secret_key is None):
            raise ValueError("access_key and secret_key must be provided together")
        self._region = region
        self._access_key = access_key
        self._secret_key = secret_key
        self._client = client

    @classmethod
    def from_config(
        cls,
        *,
        region: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
    ) -> EcrCredentials:
        """Build explicitly configured ECR credentials.

        The no-argument constructor remains available only for the legacy
        ``EcrImageStore`` defaults; generic OCI configuration must name its
        credential inputs instead of relying on the secret bundle's export keys.
        """
        if not region or not access_key or not secret_key:
            raise ValueError("region, access_key, and secret_key must be configured")
        return cls(region=region, access_key=access_key, secret_key=secret_key)

    @property
    def client(self):
        if self._client is None:
            if self._region is None:
                raise ConfigError(
                    "EcrCredentials has no region: configure it on the store's "
                    "credentials (e.g. [stores.image.config] region)."
                )
            if self._access_key is None:
                from agent_env.config import get_config

                self._client = get_config().build_ecr_client(self._region)
            else:
                self._client = boto3.client(
                    "ecr",
                    region_name=self._region,
                    aws_access_key_id=self._access_key,
                    aws_secret_access_key=self._secret_key,
                )
        return self._client

    def mint(self, host: str) -> RegistryAuth:
        token = self.client.get_authorization_token()["authorizationData"][0][
            "authorizationToken"
        ]
        try:
            username, password = (
                base64.b64decode(token, validate=True).decode().split(":", 1)
            )
        except (binascii.Error, UnicodeDecodeError, ValueError):
            raise ValueError("ECR returned an invalid authorization token") from None
        return RegistryAuth(
            registry=normalize_registry_host(host), username=username, password=password
        )


class EcrImageStore(OciRegistryImageStore):
    """ImageStore backed by AWS ECR; the client is built lazily on the first ECR op."""

    def __init__(
        self,
        registry_host: str,
        repository_prefix: str = "",
        *,
        credentials: EcrCredentials | None = None,
    ) -> None:
        if credentials is not None and not isinstance(credentials, EcrCredentials):
            raise TypeError(
                "EcrImageStore credentials must be an EcrCredentials instance"
            )
        self._ecr_credentials = credentials if credentials is not None else EcrCredentials()
        super().__init__(
            registry_host=registry_host,
            repository_prefix=repository_prefix,
            credentials=self._ecr_credentials,
        )

    @property
    def _ecr(self):
        return self._ecr_credentials.client

    def ensure_repository(self, repository: str) -> None:
        name = self._repo_name(repository)
        try:
            self._ecr.create_repository(repositoryName=name)
            logger.info(f"Created ECR repository {name}")
        except self._ecr.exceptions.RepositoryAlreadyExistsException:
            pass
        except ClientError as e:
            if e.response["Error"]["Code"] != "AccessDeniedException":
                raise
            logger.info(
                f"No CreateRepository permission for {name}; assuming it exists"
            )
