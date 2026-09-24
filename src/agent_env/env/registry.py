"""Env registry for type-based deserialization."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent_env.plugins import _registration

if TYPE_CHECKING:
    from agent_env.env.env import Env
    from agent_env.config.runtime import Config


def _get_type(cls: type["Env"]) -> str:
    return cls.type


def _builtin_registry() -> dict[str, type["Env"]]:
    """The built-in env types, by type."""
    from agent_env.env.envs.gateway_server import GatewayEnv
    from agent_env.env.envs.mcp_server import MCPServerEnv
    from agent_env.env.envs.multi_env import MultiEnv
    from agent_env.env.envs.service_db import ServiceDBEnv
    from agent_env.env.envs.website import WebsiteEnv

    return {
        _get_type(GatewayEnv): GatewayEnv,
        _get_type(MCPServerEnv): MCPServerEnv,
        _get_type(MultiEnv): MultiEnv,
        _get_type(ServiceDBEnv): ServiceDBEnv,
        _get_type(WebsiteEnv): WebsiteEnv,
    }


def _build_registry(source: Config | None = None) -> dict[str, type["Env"]]:
    """The built-in env types, then ``agent_env.envs`` plugins, then ``[envs] impls`` in ``source``."""
    from agent_env.env.env import Env

    registry = _builtin_registry()
    validate = _registration.typed_validator(Env, builtins=frozenset(registry))
    from_plugins = _registration.merge(registry, _registration.ENVS, validate, source=source)
    _merge_config_toml_envs(registry, source=source, from_plugins=from_plugins)
    return registry


def get_env_registry() -> dict[str, type["Env"]]:
    from agent_env.config import runtime

    return runtime.get_config().env_registry()


def _merge_config_toml_envs(
    registry: dict[str, type["Env"]],
    *,
    source: Config | None = None,
    from_plugins: _registration.Registrations | None = None,
) -> None:
    from agent_env.config import runtime
    from agent_env.config import ConfigError, load_impl
    from agent_env.env.env import Env

    section = (source or runtime.get_config()).section("envs")
    impls = section.get("impls", [])
    if not isinstance(impls, list):
        raise ConfigError(
            f"[envs] impls must be a list of 'module:Class' strings, got {type(impls).__name__}"
        )
    from_plugins = from_plugins if from_plugins is not None else _registration.Registrations.empty()
    for impl in impls:
        if not isinstance(impl, str):
            raise ConfigError(
                f"[envs] impl must be a 'module:Class' string, got {type(impl).__name__}: {impl!r}"
            )
        cls = load_impl(impl, Env)
        if cls.type == Env.type:
            raise ConfigError(
                f"[envs] impl {impl!r} does not define its own 'type' "
                f"(inherits the base default {Env.type!r}); set a unique 'type' ClassVar"
            )
        if not from_plugins.release(cls.type, f"[envs] impl {impl!r}", cls) and cls.type in registry:
            raise ConfigError(
                f"[envs] impl {impl!r} type {cls.type!r} is already registered "
                f"(conflicts with a built-in or another custom env)"
            )
        registry[cls.type] = cls
