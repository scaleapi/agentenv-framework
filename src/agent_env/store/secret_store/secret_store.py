"""Backend-agnostic secret store abstraction over string-keyed secrets."""

from __future__ import annotations

from abc import ABC, abstractmethod


class SecretStore(ABC):
    """Backend-selectable secret source, selected by ``AGENT_ENV_SECRET_STORE`` /
    ``[stores.secret]`` — local env/file (default), AWS Secrets Manager, or a custom backend."""

    @classmethod
    def from_config(cls, **config) -> SecretStore:
        """Construct from a resolved config table; backends that build a client override this."""
        return cls(**config)

    @abstractmethod
    def get(self, name: str) -> str | None:
        """The secret value for ``name``, or ``None`` if it is not present."""
