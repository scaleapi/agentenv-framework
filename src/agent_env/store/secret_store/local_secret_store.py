"""Local secret store backend — env vars + an optional YAML/JSON file, no cloud services."""

from __future__ import annotations

import os
from typing import Mapping, Optional

import yaml

from agent_env.store.secret_store.secret_store import SecretStore


class LocalSecretStore(SecretStore):
    """No-AWS secret backend: process env vars (checked first when ``use_env``) over an
    optional flat YAML/JSON ``file_path`` mapping of ``name -> value``."""

    def __init__(
        self,
        values: Optional[Mapping[str, object]] = None,
        *,
        file_path: Optional[str] = None,
        use_env: bool = True,
    ) -> None:
        merged: dict[str, object] = {}
        if file_path:
            if not os.path.isfile(file_path):
                raise ValueError(f"Secret file {file_path!r} does not exist.")
            with open(file_path) as f:
                loaded = yaml.safe_load(f) or {}
            if not isinstance(loaded, Mapping):
                raise ValueError(f"Secret file {file_path!r} must be a flat mapping of name -> value.")
            merged.update(loaded)
        if values:
            merged.update(values)
        self._values = merged
        self._use_env = use_env

    def get(self, name: str) -> str | None:
        if self._use_env and name in os.environ:
            return os.environ[name]
        value = self._values.get(name)
        return None if value is None else str(value)

    def _load(self) -> dict:
        """The name→value mapping this store was built from (file + values, no env), for
        the whole-bundle reader ``Config._get_secret`` (callers apply their own env overrides)."""
        return {name: str(value) for name, value in self._values.items()}
