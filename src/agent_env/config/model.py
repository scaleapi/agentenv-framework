"""Resolution of the ``[model]`` config.toml block: endpoint, default/per-role models,
provider params, and the endpoint's headers. A ``ModelConfig`` value object parsed from the
``[model]`` table; a ``secret_resolver`` is injected per call so secret references resolve lazily."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field, fields
from enum import StrEnum
from typing import Optional
from urllib.parse import urlsplit, urlunsplit

from agent_env.config.errors import ConfigError
from agent_env.config.loader import ENV_REF_PREFIX, SECRET_REF_PREFIX, SecretResolver, interpolate

logger = logging.getLogger(__name__)

ENV_BASE_URL = "LITELLM_BASE_URL"
ENV_API_KEY = "LITELLM_API_KEY"
# Agents get ``[model.headers]`` here as a JSON object; unlike its two siblings, agent-env never reads it.
MODEL_HEADERS_ENV = "LITELLM_EXTRA_HEADERS"
# RFC 9110: a header name is a token; a value is visible ASCII, spaces and tabs (no CR or LF to split it on).
_HEADER_NAME = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+\Z")
_HEADER_VALUE = re.compile(r"[\t\x20-\x7e]*\Z")


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


class ModelRole(StrEnum):
    """A role a model plays in a run. ``[model.roles]`` also takes the names it had first: ``agent`` for the solver
    and ``user`` for the user simulator."""

    SOLVER = "solver"
    USER_SIMULATOR = "user_simulator"
    JUDGE = "judge"

    @property
    def record_key(self) -> str:
        """The role's key in ``metadata.models``, which its readers have always seen."""
        return "user simulator" if self is ModelRole.USER_SIMULATOR else self.value


ROLE_SYNONYMS: dict[str, ModelRole] = {"agent": ModelRole.SOLVER, "user": ModelRole.USER_SIMULATOR}
JUDGE_FALLBACK_MODEL = "claude-sonnet-4-6"


class ModelSource(StrEnum):
    """The layer a role's model came from: the first of them, in this order, that names one."""

    STEP_PARAMS = "step_params"
    RUN = "run"
    STEP = "step"
    AGENT = "agent"
    CONFIG_ROLE = "config_role"
    CONFIG_DEFAULT = "config_default"
    FALLBACK = "fallback"


@dataclass(frozen=True)
class ResolvedModel:
    """A role's model for one step and the layer it came from; both None when no layer names one."""

    role: ModelRole
    model: Optional[str]
    source: Optional[ModelSource]


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
    headers: dict = field(default_factory=dict)  # values may hold unresolved env:/secret: references

    @classmethod
    def from_section(cls, section: dict, *, secret_resolver: SecretResolver) -> ModelConfig:
        """Build from the raw ``[model]`` section: reject unknown keys and reserved
        ``[model.params]`` keys, then resolve ``env:``/``secret:`` refs — except
        ``api_key`` and ``[model.headers]``, which keep their raw references and resolve
        lazily per call (an eager secret fetch would make every ``[model]`` read need credentials)."""
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
        headers = section.get("headers", {})
        if not isinstance(headers, dict) or not all(isinstance(v, str) for v in headers.values()):
            raise ConfigError("[model.headers] maps header names to strings (an env:/secret: reference is fine)")
        for name, value in headers.items():
            _check_header(name, None if value.startswith((ENV_REF_PREFIX, SECRET_REF_PREFIX)) else value)
        cfg = interpolate(
            {k: v for k, v in section.items() if k not in ("api_key", "headers")},
            secret_resolver=secret_resolver,
        )
        _check_roles(cfg.get("roles", {}))
        return cls(
            base_url=cfg.get("base_url"),
            api_key=section.get("api_key"),
            default=cfg.get("default"),
            roles=cfg.get("roles", {}),
            params=cfg.get("params", {}),
            headers=dict(headers),
        )

    def role_model(self, role: str) -> Optional[str]:
        """``[model.roles]``'s model for ``role``, under its name or the synonym it had first; the map is open, so a
        plugin's own role is looked up by its name."""
        canonical = ROLE_SYNONYMS.get(role, role)
        names = (canonical, *(synonym for synonym, named in ROLE_SYNONYMS.items() if named == canonical))
        return next((self.roles[name] for name in names if self.roles.get(name)), None)

    def model_for_role(self, role: str, *, or_default: bool = True) -> Optional[str]:
        """The ``[model.roles].<role>`` model, else ``default`` unless ``or_default`` is False."""
        return self.role_model(role) or (self.default if or_default else None)

    def resolve_api_key(self, *, secret_resolver: SecretResolver) -> str:
        """Key: env ``LITELLM_API_KEY`` > ``api_key`` (resolving a reference lazily);
        actionable failure when neither is set."""
        key = os.getenv(ENV_API_KEY) or self._resolved_api_key(secret_resolver)
        if not key:
            raise ConfigError(
                "No model API key configured: set [model] api_key in .agentenv/config.toml "
                f"(an env:/secret: reference is fine) or the {ENV_API_KEY} env var."
            )
        return key

    def _resolved_api_key(self, secret_resolver: SecretResolver) -> Optional[str]:
        if self.api_key is None:
            return None
        return interpolate(self.api_key, secret_resolver=secret_resolver)

    def is_endpoint(self, url: Optional[str]) -> bool:
        """Whether ``url`` names the configured endpoint (``LITELLM_BASE_URL``, else ``base_url``), by
        ``same_endpoint``: the only URL the configured key and headers go to."""
        return same_endpoint(url, os.getenv(ENV_BASE_URL) or self.base_url)

    def prices_apply(self, url: Optional[str]) -> bool:
        """Whether a cost may be priced from list prices for calls to ``url``: the configured endpoint's, or with none
        configured, a native route's (no ``url``), but never another URL's, which may serve anything under a model's
        name, a fine-tune or a free local model among them."""
        configured = os.getenv(ENV_BASE_URL) or self.base_url
        return same_endpoint(url, configured) if url else not configured

    def headers_for(self, base_url: Optional[str], *, secret_resolver: SecretResolver) -> dict[str, str]:
        """``[model.headers]``, resolved, for a call or an agent pointed at ``base_url``: none unless
        that is the configured endpoint, since a header may carry a credential."""
        if not self.headers or not base_url:
            return {}
        if not self.is_endpoint(base_url):
            logger.info("Not sending [model.headers] to %s, which is not the configured model endpoint",
                        _netloc(base_url))
            return {}
        headers = interpolate(dict(self.headers), secret_resolver=secret_resolver)
        for name, value in headers.items():
            _check_header(name, value)
        return headers

    def resolve_call(
        self,
        model: str,
        *,
        default_api_key: Optional[str] = None,
        base_override: Optional[str] = None,
        secret_resolver: SecretResolver,
    ) -> ModelCallConfig:
        """A provider-prefixed model routes natively when no endpoint is configured; a bare
        model name needs one (override/env/config) and fails actionably without it. The configured key, from
        ``LITELLM_API_KEY`` or ``api_key``, goes only to the configured endpoint or a native route; a ``base_override``
        elsewhere needs ``default_api_key``, since without one ``litellm`` would fall back to provider keys in the
        environment."""
        key = default_api_key
        configured_base = os.getenv(ENV_BASE_URL) or self.base_url
        if base_override:
            api_base = base_override
        elif configured_base:
            api_base = configured_base
        elif "/" in model:
            api_base = None  # provider-prefixed: litellm routes natively
        else:
            raise ConfigError(
                f"No model endpoint configured for {model!r}: set [model] base_url in "
                f".agentenv/config.toml or {ENV_BASE_URL}, or use a provider-prefixed "
                "model name (e.g. 'openai/gpt-4o') for native routing."
            )
        if key is None:
            if api_base is not None and not same_endpoint(api_base, configured_base):
                raise ConfigError(no_key_for(api_base, "default_api_key"))
            key = os.getenv(ENV_API_KEY) or self._resolved_api_key(secret_resolver)
        params = dict(self.params)
        headers = self.headers_for(api_base, secret_resolver=secret_resolver)
        if headers:
            explicit = params.get("extra_headers", {})
            overridden = {name.lower() for name in explicit}
            params["extra_headers"] = {**{k: v for k, v in headers.items() if k.lower() not in overridden}, **explicit}
        return ModelCallConfig(api_key=key, api_base=api_base, params=params)


def resolve_model(
    role: ModelRole,
    *,
    step_params: Optional[str] = None,
    run: Optional[str] = None,
    step: Optional[str] = None,
    agent: Optional[str] = None,
    config: Optional[ModelConfig] = None,
) -> ResolvedModel:
    """``role``'s model for one step: the first of the layers its call site has that names one, else ``config``'s
    (``named_model``, then ``config_model``). It reads no environment, key or secret, so a plan can resolve every role
    before its run starts."""
    return (named_model(role, step_params=step_params, run=run, step=step, agent=agent)
            or config_model(role, config))


def named_model(
    role: ModelRole,
    *,
    step_params: Optional[str] = None,
    run: Optional[str] = None,
    step: Optional[str] = None,
    agent: Optional[str] = None,
) -> Optional[ResolvedModel]:
    """The first of these layers that names a model: this step's ``step_params``, the run's override, the step's own
    field, the deployed agent's default; None when none does."""
    for source, model in ((ModelSource.STEP_PARAMS, step_params), (ModelSource.RUN, run),
                          (ModelSource.STEP, step), (ModelSource.AGENT, agent)):
        if model:
            return ResolvedModel(role, model, source)
    return None


def config_model(role: ModelRole, config: Optional[ModelConfig]) -> ResolvedModel:
    """``role``'s model when no layer names one: ``[model.roles]``, then for the solver and the judge ``[model]
    default``; a judge always has one, claude-sonnet-4-6 last."""
    if config is not None:
        if model := config.role_model(role):
            return ResolvedModel(role, model, ModelSource.CONFIG_ROLE)
        if role is not ModelRole.USER_SIMULATOR and config.default:
            return ResolvedModel(role, config.default, ModelSource.CONFIG_DEFAULT)
    if role is ModelRole.JUDGE:
        return ResolvedModel(role, JUDGE_FALLBACK_MODEL, ModelSource.FALLBACK)
    return ResolvedModel(role, None, None)


_DEFAULT_PORTS = {"http": 80, "https": 443}


def same_endpoint(a: Optional[str], b: Optional[str]) -> bool:
    """Whether two URLs name one model endpoint: scheme and host in any case, a default port, a trailing ``/`` and one
    trailing ``/v1`` don't tell them apart, so a client that wants ``/v1`` and one that doesn't reach the same one."""
    return bool(a and b) and _endpoint(a) == _endpoint(b)


def api_root(url: str) -> str:
    """``url`` without a trailing ``/`` and one trailing ``/v1``, for a client that appends its own versioned path
    (Anthropic's ``/v1/messages``): the endpoint may be configured either way, and ``same_endpoint`` compares by the
    same root."""
    parts = urlsplit(url.strip())
    return urlunsplit(parts._replace(path=_root_path(parts.path)))


def _root_path(path: str) -> str:
    return path.rstrip("/").removesuffix("/v1").rstrip("/")


def _endpoint(url: str) -> tuple:
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower()
    try:
        port = parts.port
    except ValueError:
        return (url,)
    path = _root_path(parts.path)
    return scheme, (parts.hostname or "").lower(), None if port == _DEFAULT_PORTS.get(scheme) else port, path, parts.query


def _netloc(url: str) -> str:
    """``url``'s host and port, without any user or password it carries."""
    return urlsplit(url).netloc.rpartition("@")[2] or url


def endpoint_host(url: object) -> Optional[str]:
    """The host, and port when it names one, a model call to ``url`` reaches, for a record: never the user or
    password a URL may carry. None for anything but a URL with a host, never raising, so a record can't fail what it
    records."""
    if not isinstance(url, str):
        return None
    try:
        return urlsplit(url.strip()).netloc.rpartition("@")[2] or None
    except ValueError:
        return None


def no_key_for(url: str, how: str) -> str:
    """Why a model call or an agent pointed at ``url`` with no key of its own is refused, and ``how`` to give it one."""
    return (f"{_netloc(url)} isn't the configured model endpoint, so the configured key isn't sent there, and no key "
            f"was given with it. Pass one with it: {how}. Any non-empty value works for a server that needs no key.")


def _check_roles(roles: object) -> None:
    """Refuse a ``[model.roles]`` that maps a role to anything but a model name, or names one role twice, under its
    name and its synonym."""
    if not isinstance(roles, dict) or not all(isinstance(model, str) for model in roles.values()):
        raise ConfigError("[model.roles] maps each role to a model name, such as judge = \"claude-sonnet-4-6\"")
    for synonym, role in ROLE_SYNONYMS.items():
        if synonym in roles and role.value in roles:
            raise ConfigError(f"[model.roles] sets both {role.value!r} and {synonym!r}, which name the same role; "
                              f"keep {role.value!r}")


def _check_header(name: str, value: Optional[str]) -> None:
    """Refuse a ``[model.headers]`` entry HTTP can't carry (``value`` None checks the name alone): a non-ASCII
    value would fail inside an agent's client, and a CR or LF would split one header into two."""
    if not _HEADER_NAME.match(name):
        raise ConfigError(f"[model.headers] {name!r} is not an HTTP header name")
    if value is not None and not _HEADER_VALUE.match(value):
        raise ConfigError(f"[model.headers] {name!r} has a value HTTP can't carry: only visible ASCII, spaces and tabs")
