"""Backend-agnostic container image (registry) store abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from agent_env.store.image_store.oci_registry_credentials import (
    OciRegistryCredentials,
    RegistryAuth,
    normalize_registry_host,
    registry_host_from_ref,
)


class ImageStore(ABC):
    """Image store interface. Images are addressed by a logical ``repository`` and ``tag``.

    ``image_ref`` builds a fully-qualified, pullable ref; ``auth`` returns the
    docker-login material a remote executor applies — the store never runs docker.
    """

    @classmethod
    def from_config(cls, **config) -> ImageStore:
        """Construct from a resolved config table; backends that build a client override this."""
        return cls(**config)

    @abstractmethod
    def image_ref(self, repository: str, tag: str) -> str:
        """Fully-qualified, pullable ref for ``(repository, tag)`` in this store."""

    def ensure_repository(self, repository: str) -> None:
        """Create the repository if the backend requires pre-creation; no-op otherwise."""

    @abstractmethod
    def auth(self, ref: str) -> RegistryAuth | None:
        """Freshly-minted docker-login material for ``ref``, or None when none is needed
        (anonymous/local registry, a public ref, or a ref not in this store)."""


class OciRegistryImageStore(ImageStore):
    """Generic OCI registry whose repository names and credentials are configurable."""

    def __init__(
        self,
        registry_host: str,
        repository_prefix: str = "",
        credentials: OciRegistryCredentials | None = None,
    ) -> None:
        self._registry = normalize_registry_host(registry_host)
        self._prefix = repository_prefix
        self._credentials = credentials

    @classmethod
    def from_config(
        cls,
        *,
        registry_host: str,
        repository_prefix: str = "",
        credentials: OciRegistryCredentials | Mapping[str, object] | None = None,
    ) -> OciRegistryImageStore:
        if isinstance(credentials, Mapping):
            from agent_env.config import ConfigError, load_impl

            credential_config = dict(credentials)
            impl = credential_config.pop("impl", None)
            if not impl:
                raise ConfigError(
                    "Registry credentials are missing an 'impl' dotted path"
                )
            credential_cls = load_impl(impl, OciRegistryCredentials)
            credentials = credential_cls.from_config(**credential_config)
        elif credentials is not None and not isinstance(
            credentials, OciRegistryCredentials
        ):
            raise TypeError("credentials must implement OciRegistryCredentials")
        return cls(
            registry_host=registry_host,
            repository_prefix=repository_prefix,
            credentials=credentials,
        )

    @property
    def registry_host(self) -> str:
        return self._registry

    @property
    def credentials(self) -> OciRegistryCredentials | None:
        return self._credentials

    def image_ref(self, repository: str, tag: str) -> str:
        return f"{self._registry}/{self._repo_name(repository)}:{tag}"

    def auth(self, ref: str) -> RegistryAuth | None:
        if registry_host_from_ref(ref) != self._registry or self._credentials is None:
            return None
        return self._credentials.mint(self._registry)

    def _repo_name(self, repository: str) -> str:
        return f"{self._prefix}/{repository}" if self._prefix else repository
