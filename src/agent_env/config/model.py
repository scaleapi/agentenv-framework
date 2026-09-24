"""Resolution of the ``[model]`` config.toml block: endpoint, default/per-role models, and
provider params. A ``ModelConfig`` value object parsed from the ``[model]`` table; a
``secret_resolver`` is injected per call so secret references resolve lazily."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from enum import StrEnum
from typing import Optional

from agent_env.config.errors import ConfigError
from agent_env.config.loader import SecretResolver, interpolate

_ENV_BASE_URL = "LITELLM_BASE_URL"
_ENV_API_KEY = "LITELLM_API_KEY"


class ModelParam(StrEnum):
    """The ``litellm`` call kwargs agent-env sets itself; ``[model.params]`` may not override these."""

    MODEL = "model"
    MESSAGES = "messages"
    API_KEY = "api_key"
    API_BASE = "api_base"
    USER = "user"
    METADATA = "metadata"
    TIMEOUT = "timeout"
    RESPONSE_FORMAT = "response_format"


MODEL_PARAMS_RESERVED = frozenset(p.value for p in ModelParam)


@dataclass(frozen=True)
class ModelCallConfig:
    """Resolved api_base / api_key / params for one model call."""

    api_key: Optional[str]
    api_base: Optional[str]
    params: dict

    def client_kwargs(self) -> dict:
        """``api_key`` + ``params``, plus ``api_base`` only when set."""
        kwargs = {ModelParam.API_KEY: self.api_key, **self.params}
        if self.api_base:
            kwargs[ModelParam.API_BASE] = self.api_base
        return kwargs


@dataclass(frozen=True)
class ModelConfig:
    """The parsed ``[model]`` block (every field optional; all-default when unconfigured)."""

    base_url: Optional[str] = None
    api_key: Optional[str] = None  # may hold an unresolved env:/secret: reference
    default: Optional[str] = None
    roles: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)

    @classmethod
    def from_section(cls, section: dict, *, secret_resolver: SecretResolver) -> ModelConfig:
        """Build from the raw ``[model]`` section: reject unknown keys and reserved
        ``[model.params]`` keys, then resolve ``env:``/``secret:`` refs — except
        ``api_key``, which keeps its raw reference and resolves lazily per call
        (an eager secret fetch would make every ``[model]`` read need credentials)."""
        allowed = {f.name for f in fields(cls)}
        unknown = set(section) - allowed
        if unknown:
            raise ConfigError(f"[model] has unknown keys {sorted(unknown)}; allowed: {sorted(allowed)}")
        reserved = set(section.get("params", {})) & MODEL_PARAMS_RESERVED
        if reserved:
            raise ConfigError(
                f"[model.params] may not set reserved keys {sorted(reserved)} "
                "(agent-env sets these on the model call)"
            )
        cfg = interpolate(
            {k: v for k, v in section.items() if k != "api_key"},
            secret_resolver=secret_resolver,
        )
        return cls(
            base_url=cfg.get("base_url"),
            api_key=section.get("api_key"),
            default=cfg.get("default"),
            roles=cfg.get("roles", {}),
            params=cfg.get("params", {}),
        )

    def model_for_role(self, role: str) -> Optional[str]:
        """The ``[model.roles].<role>`` model, else ``default`` (open map)."""
        return self.roles.get(role) or self.default

    def resolve_api_key(self, *, secret_resolver: SecretResolver) -> str:
        """Key: env ``LITELLM_API_KEY`` > ``api_key`` (resolving a reference lazily);
        actionable failure when neither is set."""
        key = os.getenv(_ENV_API_KEY) or self._resolved_api_key(secret_resolver)
        if not key:
            raise ConfigError(
                "No model API key configured: set [model] api_key in .agentenv/config.toml "
                f"(an env:/secret: reference is fine) or the {_ENV_API_KEY} env var."
            )
        return key

    def _resolved_api_key(self, secret_resolver: SecretResolver) -> Optional[str]:
        if self.api_key is None:
            return None
        return interpolate(self.api_key, secret_resolver=secret_resolver)

    def resolve_call(
        self,
        model: str,
        *,
        default_api_key: Optional[str] = None,
        base_override: Optional[str] = None,
        model_overrides: Optional[dict[str, dict[str, str]]] = None,
        secret_resolver: SecretResolver,
    ) -> ModelCallConfig:
        """A provider-prefixed model routes natively when no endpoint is configured; a bare
        model name needs one (override/env/config) and fails actionably without it."""
        if model_overrides:
            override = model_overrides.get(model)
            if override:
                key = secret_resolver(override["secret_key"])
                if key is None:
                    raise ConfigError(
                        f"Model override for {model!r} names secret {override['secret_key']!r}, "
                        "which the configured secret store has no value for"
                    )
                return ModelCallConfig(
                    api_key=key,
                    api_base=base_override or override["base_url"],
                    params={},
                )

        key = default_api_key or os.getenv(_ENV_API_KEY)
        configured_base = os.getenv(_ENV_BASE_URL) or self.base_url
        if base_override:
            api_base = base_override
        elif configured_base:
            api_base = configured_base
        elif "/" in model:
            api_base = None  # provider-prefixed: litellm routes natively
        else:
            raise ConfigError(
                f"No model endpoint configured for {model!r}: set [model] base_url in "
                f".agentenv/config.toml or {_ENV_BASE_URL}, or use a provider-prefixed "
                "model name (e.g. 'openai/gpt-4o') for native routing."
            )
        # The configured key is scoped to the configured endpoint (and native routing):
        # a base_override pointing elsewhere must not inherit it, or a caller-supplied
        # URL would receive the configured proxy's credential.
        if key is None and (api_base is None or api_base == configured_base):
            key = self._resolved_api_key(secret_resolver)
        return ModelCallConfig(api_key=key, api_base=api_base, params=dict(self.params))
