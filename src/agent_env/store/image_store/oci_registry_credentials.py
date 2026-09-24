"""Credential providers for OCI registries."""

from __future__ import annotations

import base64
import binascii
import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegistryAuth:
    """docker-login material for a registry (the password may be a short-lived token)."""

    registry: str
    username: str
    password: str = field(repr=False)


class OciRegistryCredentials(ABC):
    """Produces docker-login material for a registry.

    Implementations resolve secret or identity inputs inside ``mint`` rather than
    ``__init__`` so a process-cached image store can observe credential rotation.
    """

    @classmethod
    def from_config(cls, **config) -> OciRegistryCredentials:
        return cls(**config)

    @abstractmethod
    def mint(self, host: str) -> RegistryAuth | None:
        """Return credentials for ``host``, or ``None`` when none are available."""


# Docker Hub is special: unqualified image references resolve there, Docker config
# commonly keys its credentials by ``index.docker.io``, and pulls may use
# ``registry-1.docker.io``. Canonicalizing these aliases lets one credential entry
# cover all equivalent forms; every other registry still requires an exact host match.
_DOCKER_HUB_HOSTS = {
    "docker.io",
    "index.docker.io",
    "registry-1.docker.io",
}


def normalize_registry_host(host: str) -> str:
    """Normalize a registry authority without weakening exact host matching."""
    normalized = host.strip().lower()
    if not normalized:
        raise ValueError("registry_host must not be empty")
    if "://" in normalized or "/" in normalized:
        raise ValueError("registry_host must be an authority without a scheme or path")
    return "docker.io" if normalized in _DOCKER_HUB_HOSTS else normalized


def registry_host_from_ref(ref: str) -> str | None:
    """Return the normalized registry authority from a Docker/OCI image reference.

    Docker treats a first path component as a registry only when it is ``localhost``
    or contains a dot or port. Unqualified references therefore resolve to Docker Hub.
    Schemed values are not valid image references and deliberately do not match.
    """
    if not ref or "://" in ref:
        return None
    first, separator, _ = ref.partition("/")
    if not first:
        return None
    if not separator or not (
        first.lower() == "localhost" or "." in first or ":" in first
    ):
        return "docker.io"
    try:
        return normalize_registry_host(first)
    except ValueError:
        return None


class SecretStoreCredentials(OciRegistryCredentials):
    """Read inline Docker-config credentials from the configured secret store."""

    # The default is a secret-store lookup key, not credential material.
    def __init__(self, secret_key: str = "registry_auths") -> None:  # nosec B107
        self._secret_key = secret_key

    def mint(self, host: str) -> RegistryAuth | None:
        from agent_env.config import ConfigError, get_config

        raw = get_config().get_secret_store().get(self._secret_key)
        if raw is None:
            return None
        try:
            document = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            raise ConfigError(
                f"Registry credential secret {self._secret_key!r} must be a JSON string "
                "containing a Docker config 'auths' mapping; nested YAML mappings are not supported"
            ) from None
        if not isinstance(document, Mapping):
            raise ConfigError(
                f"Registry credential secret {self._secret_key!r} must contain a JSON object"
            )

        if "credsStore" in document or "credHelpers" in document:
            logger.debug(
                "Registry credential secret %r contains Docker credential-helper settings; "
                "agent-env only reads inline auth entries",
                self._secret_key,
            )

        auths = document.get("auths", document)
        if not isinstance(auths, Mapping):
            raise ConfigError(
                f"Registry credential secret {self._secret_key!r} has a non-object 'auths' value"
            )

        normalized_host = normalize_registry_host(host)
        entry = next(
            (
                candidate
                for key, candidate in auths.items()
                if _docker_config_registry_host(str(key)) == normalized_host
            ),
            None,
        )
        if entry is None:
            return None
        if not isinstance(entry, Mapping):
            raise ConfigError(
                f"Registry credential secret {self._secret_key!r} has invalid credentials "
                f"for registry {normalized_host!r}"
            )

        if entry.get("auth") is not None:
            username, password = _decode_auth(
                entry["auth"], self._secret_key, normalized_host
            )
        else:
            username = entry.get("username")
            password = entry.get("password")
            if not isinstance(username, str) or not isinstance(password, str):
                raise ConfigError(
                    f"Registry credential secret {self._secret_key!r} must provide either 'auth' "
                    f"or string 'username'/'password' fields for registry {normalized_host!r}"
                )
        return RegistryAuth(
            registry=normalized_host, username=username, password=password
        )


def _docker_config_registry_host(key: str) -> str | None:
    candidate = key.strip()
    if not candidate:
        return None
    if "://" in candidate:
        parsed = urlsplit(candidate)
        if not parsed.hostname:
            return None
        try:
            candidate = parsed.hostname
            if parsed.port is not None:
                candidate = f"{candidate}:{parsed.port}"
        except ValueError:
            return None
    else:
        candidate = candidate.split("/", 1)[0]
    try:
        return normalize_registry_host(candidate)
    except ValueError:
        return None


def _decode_auth(value: object, secret_key: str, host: str) -> tuple[str, str]:
    if not isinstance(value, str):
        raise _invalid_auth(secret_key, host)
    try:
        decoded = base64.b64decode(value, validate=True).decode()
        username, password = decoded.split(":", 1)
    except (binascii.Error, UnicodeDecodeError, ValueError):
        raise _invalid_auth(secret_key, host) from None
    return username, password


def _invalid_auth(secret_key: str, host: str):
    from agent_env.config import ConfigError

    return ConfigError(
        f"Registry credential secret {secret_key!r} has invalid base64 'auth' "
        f"for registry {host!r}"
    )
